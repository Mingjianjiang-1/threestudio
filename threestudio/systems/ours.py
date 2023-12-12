import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.typing import *
from threestudio.systems.utils import convert_c2w_to_hp


@threestudio.register("ours-system")
class Ours(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        per_editing_step: int = 10
        start_editing_step: int = 1000
        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.edit_frames = {}
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.benchmark_image_path = None

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance2 = None #threestudio.find(self.cfg.guidance_3d_type)(self.cfg.guidance_3d)

    def training_step(self, batch, batch_idx):
        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        origin_gt_rgb = batch["gt_rgb"]
        
        # for key in batch.keys():
        #     if torch.is_tensor(batch[key]):
        #         print(f'Key {key}: Shape {batch[key].shape}')
        #     else:
        #         print(f'Key {key}: Shape {batch[key]}')
        B, H, W, C = origin_gt_rgb.shape
        if batch_index in self.edit_frames:
            gt_rgb = self.edit_frames[batch_index].to(batch["gt_rgb"].device)
            gt_rgb = torch.nn.functional.interpolate(
                gt_rgb.permute(0, 3, 1, 2), (H, W), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            #batch["gt_rgb"] = gt_rgb
        else:
            gt_rgb = origin_gt_rgb
        out = self(batch)
        # for key in out.keys():
        #     print(f'Key {key}: Shape {out[key].shape}')
        guidance_out = {}
        edited = False
        if (
            self.cfg.per_editing_step > 0
            and self.global_step > self.cfg.start_editing_step
        ):
            prompt_utils = self.prompt_processor()
            if (    
                not batch_index in self.edit_frames
                or self.global_step % self.cfg.per_editing_step == 0
            ):
                self.renderer.eval()
                full_out = self(batch)
                self.renderer.train()
                result = self.guidance(   ##apply instruct pix to pix
                    full_out["comp_rgb"], origin_gt_rgb, prompt_utils   
                )
                self.edit_frames[batch_index] = result["edit_images"].detach().cpu()
                guidance_out["loss_sds"] = result['loss_sds']
                edited = True
        else:
            guidance_out["loss_l1"] =  torch.nn.functional.l1_loss(out["comp_rgb"], gt_rgb)
            guidance_out["loss_p"] = self.perceptual_loss(out["comp_rgb"].permute(0, 3, 1, 2).contiguous(),gt_rgb.permute(0, 3, 1, 2).contiguous()).sum()

        loss = 0.0
        loss_zero123 = 0.0
        #todo get azimuth_deg, camera_distance, evaluation for the image
        assert B == 1
        if(self.guidance2 == None and edited):
            result = self.guidance(   ##apply instruct pix to pix
                    origin_gt_rgb, origin_gt_rgb, prompt_utils   
            )
            edited_img = result["edit_images"].detach().cpu()
            # B, H, W, C = edited_img.shape
            # rgb = torch.nn.functional.interpolate(
            #     edited_img.permute(0, 3, 1, 2), (H, W)
            # ).permute(0, 2, 3, 1)[0]
            print(guidance_out)
            self.save_image_grid(
            "cond_img_test.jpg", 
            [   
             {
                    "type": "rgb",
                    "img": origin_gt_rgb[0],
                    "kwargs": {"data_format": "HWC"},
                },
             {
                    "type": "rgb",
                    "img": full_out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": edited_img[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ])
            self.cfg.guidance_3d.cond_image_path = self.save_image_grid(
            "cond_img.jpg", 
            [   
                {
                    "type": "rgb",
                    "img": edited_img[0],
                    "kwargs": {"data_format": "HWC"},
                }
            ])
            print(self.cfg.guidance_3d.cond_image_path)
            # self.cfg.cond_azimuth_deg = 0.0
            # self.cfg.cond_elevation_deg = 0.0
            # self.cfg.cond_camera_distance = 1.0
            hp_results = convert_c2w_to_hp(batch['c2w'][0])
            print(hp_results)
            
            self.cfg.guidance_3d.cond_azimuth_deg = hp_results['azimuth']
            self.cfg.guidance_3d.cond_elevation_deg = hp_results['elevation'] 
            self.cfg.guidance_3d.cond_camera_distance = hp_results['camera_distance']
            self.guidance2 = threestudio.find(self.cfg.guidance_3d_type)(self.cfg.guidance_3d)
        elif edited:
            # loss_zero123 = self.guidance2(full_out["comp_rgb"], origin_gt_rgb, prompt_utils)
            hp_results = convert_c2w_to_hp(batch['c2w'][0])
            batch['azimuth'] = torch.tensor([hp_results['azimuth']], device=batch['c2w'].device)
            batch['elevation'] = torch.tensor([hp_results['elevation']], device=batch['c2w'].device)
            batch['camera_distances'] = torch.tensor([hp_results['camera_distance']], device=batch['c2w'].device)
            loss_zero123 = self.guidance2(full_out["comp_rgb"], **batch, rgb_as_latents=False,)['loss_sd']
            guidance_out["loss_zero123"] = loss_zero123
            #print("loss_zero123",loss_zero123)
            #and self.global_step > self.cfg.start_editing_step)

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        if self.global_step > self.cfg.start_editing_step:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        if batch_index in self.edit_frames:
            B, H, W, C = batch["gt_rgb"].shape
            rgb = torch.nn.functional.interpolate(
                self.edit_frames[batch_index].permute(0, 3, 1, 2), (H, W)
            ).permute(0, 2, 3, 1)[0]
        else:
            rgb = batch["gt_rgb"][0]
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": rgb,
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
