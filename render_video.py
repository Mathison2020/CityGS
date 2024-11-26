#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
import copy, time
import imageio

gen_image = lambda x: (x[0] * x[1] + x[2]) * x[3] 
gen_sun_shade = lambda x: x[0] * x[1]
gen_shade = lambda x: x[0] * x[1] + x[2]
to8b = lambda x: x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def interpolate_views(view0, view1, length):
    #view0 source view
    #view1 target view 
    results = []
    for i in range(length):
        weight = (i+1) / length
        view = copy.deepcopy(view0)
        # view.camera_center = view0.camera_center * (1-weight) + view1.camera_center * weight
        view.FoVy = view0.FoVy * (1-weight) + view1.FoVy * weight
        view.FoVx = view0.FoVx * (1-weight) + view1.FoVx * weight
        view.camera_center = view0.camera_center * (1-weight) + view1.camera_center * weight
        view.projection_matrix = view0.projection_matrix * (1-weight) + view1.projection_matrix * weight
        view.world_view_transform = view0.world_view_transform * (1-weight) + view1.world_view_transform * weight
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        results.append(view)
    return results

import copy

def interpolate_views_quadratic(view0, view1, view2, length):
    # view0: source view, view1: intermediate view, view2: target view
    results = []
    for i in range(length):
        t = (i+1) / length  # 归一化参数
        # 二次插值公式
        a = 1 - 2*t**2 + t**3
        b = t**2 - t**3
        c = t**3 - t**2
        
        view = copy.deepcopy(view0)
        # view.camera_center = a * view0.camera_center + b * view1.camera_center + c * view2.camera_center
        view.FoVy = a * view0.FoVy + b * view1.FoVy + c * view2.FoVy
        view.FoVx = a * view0.FoVx + b * view1.FoVx + c * view2.FoVx
        view.projection_matrix = a * view0.projection_matrix + b * view1.projection_matrix + c * view2.projection_matrix
        view.world_view_transform = a * view0.world_view_transform + b * view1.world_view_transform + c * view2.world_view_transform
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        results.append(view)
    return results


def generate_multi_views(views, length=60):
    generated_views = []
    for i in range(len(views)-1):
        views_temp = interpolate_views(views[i], views[i+1], length)
        generated_views.extend(views_temp)
    return generated_views


def render_video(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        model_path = dataset.model_path
    
        train_views = scene.getTrainCameras()   # 获取所有训练视角信息
        
        # 选定特定几个训练视角作为基准
        base_view_idxs = [1821, 1891] # [45, 92] [714, 737] [1821, 1891]
        base_views = [train_views[i] for i in base_view_idxs] #if i in base_view_idxs
        
        # 在选定的基准视角之间进行视角插值，插帧数量为 length_view
        length_view = 4500
        # 在 base_views 插帧之后的视角数量为 (len(base_views)-1) * length_view
        generated_views = generate_multi_views(base_views, length_view)
        
        video_images_path = os.path.join(model_path, f"video_test_3") #
        makedirs(video_images_path, exist_ok=True)
        
        render_video_out = imageio.get_writer(f'{video_images_path}.mp4', mode='I', fps=60, codec='libx264',quality=10.0) #
        
        for idx, view in enumerate(tqdm(generated_views, desc="Rendering progress")):
                            
            image = render(view, gaussians, pipeline, background)["render"]
            np_image = to8b(image)
            render_video_out.append_data(np_image)
        #   torchvision.utils.save_image(image, os.path.join(video_images_path, f"video_image_{idx:05d}.png"))
            
        render_video_out.close()

        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
        
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_video(model.extract(args), 
                args.iteration, 
                pipeline.extract(args))