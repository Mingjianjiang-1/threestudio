import sys
import warnings
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np

import threestudio


def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        threestudio.debug(f"Specify optimizer params: {config.params}")
    else:
        params = model.parameters()
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    elif config.name in ["Adan"]:
        from threestudio.systems import optimizers

        optim = getattr(optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler_to_instance(config, optimizer):
    if config.name == "ChainedScheduler":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.ChainedScheduler(schedulers)
    elif config.name == "Sequential":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=config.milestones
        )
    else:
        scheduler = getattr(lr_scheduler, config.name)(optimizer, **config.args)
    return scheduler


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler

def convert_c2w_to_hp(c2w_tensor):
    # Calculating the camera distance, azimuth, and elevation using the tensor
    # Camera position (distance)
    camera_position_tensor = c2w_tensor[:3, 3]
    distance_tensor = torch.norm(camera_position_tensor)

    # Forward vector (negative of third column for right-handed coordinate system)
    forward_vector_tensor = -c2w_tensor[:3, 2]

    # Normalize the forward vector
    forward_vector_normalized_tensor = forward_vector_tensor / torch.norm(forward_vector_tensor)
    x, y, z = forward_vector_normalized_tensor[0], forward_vector_normalized_tensor[1], forward_vector_normalized_tensor[2]
    x_, y_, z_ = -x, -z, -y

    # Compute Elevation (Pitch)
    elevation_tensor = torch.atan2(y_, torch.sqrt(x_**2 + z_**2))

    # Compute Azimuth (Yaw)
    azimuth_tensor = torch.atan2(z_, x_)

    # distance_tensor, torch.degrees(elevation_tensor), torch.degrees(azimuth_tensor)
    radians_to_degrees = 180.0 / np.pi

    # Convert elevation and azimuth from radians to degrees
    elevation_degrees = elevation_tensor * radians_to_degrees
    azimuth_degrees = azimuth_tensor * radians_to_degrees

    return {'azimuth': azimuth_degrees.item(), 
            'elevation': elevation_degrees.item(),
            'camera_distance': distance_tensor.item()
        }