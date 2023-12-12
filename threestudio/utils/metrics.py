import clip
import re
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

device = "cuda"

class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device=device, download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).to(device))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).to(device))

    def encode_text(self, text):
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(
        self, image_0, image_1, text_0, text_1
    ):
        """
        :param image_0: [batch_size, n_channels, H, W]
        :param image_1: [batch_size, n_channels, H, W]
        :param text_0: str
        :param text_1: str
        :return sim_0: [batch_size] similarity between image_0 and text_0
        :return sim_1: [batch_size] similarity between image_1 and text_1
        :return sim_direction: [batch_size] similarity between (image_1 - image_0) and (text_1 - text_0)
        :return sim_consistency: [batch_size - 1] (C(e_i) - C(o_i)) * (C(e_{i + 1}) - C(e_i))
        """
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        print(image_features_0.shape, text_features_0.shape)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_images = F.cosine_similarity(image_features_0, image_features_1)[:-1]
        sim_consistency = F.cosine_similarity(image_features_1[:-1], image_features_1[1:])
        sim_consistency = sim_images * sim_consistency
        return sim_0, sim_1, sim_direction, sim_consistency

def get_img_tensor(matcher, img_dir):
    matcher = re.compile(matcher)
    imgs = []
    # print(matcher)
    for f in os.listdir(img_dir):
        # print(f)
        if matcher.search(f):
            imgs.append(f)
    # print(imgs)
    imgs = sorted(imgs, key=lambda f: int(matcher.search(f).groups()[0]))
    imgs = [torch.tensor(cv2.imread(os.path.join(img_dir, f))).unsqueeze(dim=0) for f in imgs]
    imgs = torch.cat(imgs, dim=0)
    # Here you can select which image among the four to compare
    imgs = torch.split(imgs, imgs.shape[2] // 4, dim=1)[0]
    imgs = imgs.permute(0, 3, 1, 2).to(device)
    return imgs

def calculate_loss(matcher_1, 
                   prompt_1, 
                   img_dir_1,
                   matcher_2, 
                   prompt_2,
                   img_dir_2):
    imgs1, imgs2 = get_img_tensor(matcher_1, img_dir_1), get_img_tensor(matcher_2, img_dir_2)
    m = ClipSimilarity()
    sim_0, sim_1, sim_direction, sim_consistency = m(imgs1, imgs2, prompt_1, prompt_2)
    info = {}
    info["sim_0"] = sim_0.mean().item()
    info["sim_1"] = sim_1.mean().item()
    info["sim_direction"] = sim_direction.mean().item()
    info["sim_consistency"] = sim_consistency.mean().item()
    print(info)
    return info

step = 3000

"""
We compare all images fallen under `img_dir_1` that could be matched with 
regular expression `matcher_1` and could be expressed by `prompt_1` with all 
images fallen under `img_dir_2` that could be matched with regular expression 
`matcher_2` and could be expressed by `prompt_2`.

`sim_direction` measures whether the image content in `img_dir_1` matches the 
prompt transitioning from `prompt_1` to `prompt_2`. Higher is better. If   
`prompt_1` is "Do nothing", and `prompt_2` is "Give him a checkered jacket", 
`img_dir_1` contains original images, `img_dir_2` contains edited images, then
`sim_direction` should measure if the image edit is following the prompt.

`sim_consistency` measures whether the image angle shifting series in 
`img_dir_1` matches the image angle shifting series in `img_dir_2`, as well as 
whether the image angle shifting series themselves are consistent. If 
`img_dir_1` contains original images, `img_dir_2` contains edited images, then
`sim_consistency` should measure whether the edited images in `img_dir_2` are 
consistent with `img_dir_1` and consistent with themselves.
"""
data_dir = "/mnt/disks/disk/Project236/threestudio/outputs/instructnerf2nerf/face_Give_him_a_cowboy_hat@20231210-072926/"
# os.mkdir(data_dir+'origin/save')
# for f in os.listdir(data_dir):
#     if os.path.isfile(f):
#         if(f[0:6] == "it600-"):
#             os.cp(f, data_dir+'origin/')
calculate_loss(matcher_1="it600-(\d+)\.png", 
               matcher_2="it6000-(\d+)\.png",
               img_dir_1=data_dir + "save/",
               img_dir_2 = data_dir + "save/",
               prompt_1 = "Give him a jacket",
               prompt_2 = "Give him a cowboy hat")

