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
import time
import yaml
import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from torchmetrics.functional.regression import pearson_corrcoef
from gaussian_renderer import render, render_large, network_gui
import sys
from lightning.pytorch.loggers import (
    TensorBoardLogger,
    WandbLogger,
)
import torch.nn.functional as F
from scene import LargeScene
from scene.datasets import GSDataset, CacheDataLoader
from utils.camera_utils import loadCam
from utils.general_utils import safe_state, parse_cfg
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.log_utils import tensorboard_log_image, wandb_log_image
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import GroupParams
from scene.appearance_network import decouple_appearance

def training(dataset, opt, pipe, testing_iterations, saving_iterations, refilter_iterations, checkpoint_iterations, checkpoint, debug_from, use_depth_loss):
    first_iter = 0
    log_writer, image_logger = prepare_output_and_logger(dataset)
    
    modules = __import__('scene')
    model_config = dataset.model_config
    apply_apperance_decouple = dataset.apply_apperance_decouple
    apply_mask = dataset.apply_mask
    gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_config['kwargs'])
    scene = LargeScene(dataset, gaussians)
    gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipe)
    if len(gs_dataset) > 0:
        data_loader = CacheDataLoader(gs_dataset, max_cache_num=1024, seed=42, batch_size=1, shuffle=True, num_workers=8)
    
    if checkpoint:
        print("Create Gaussians from checkpoint {}".format(checkpoint))
        gaussians.load_ply(os.path.join(checkpoint, 'point_cloud', f"iteration_{opt.iterations}" "point_cloud.ply"))
    gaussians.training_setup(opt, apply_apperance_decouple, args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_time_render = 0.0
    ema_time_loss = 0.0
    ema_time_densify = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    while iteration <= opt.iterations:
        if len(gs_dataset) == 0:
            print("No training data found")
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration, dataset)
            break    
        
        for dataset_index, data in enumerate(data_loader):
            if args.type == '3dgs':
                if len(data) == 4:
                    cam_info, gt_image, mask, gt_obj = data
                else:
                    cam_info, gt_image, mask, gt_obj, gt_depth = data  
            else:
                while True:
                    if len(data) == 4:
                        cam_info, gt_image, mask, gt_obj = data
                    else:
                        cam_info, gt_image, mask, gt_obj, gt_depth = data  
                    if gt_obj is not None:
                        break
            
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Render
            start = time.time()
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render_large(cam_info, gaussians, pipe, background)
            
            if args.type == '3dgs':
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                depth = render_pkg["depth"]
                end = time.time()
                ema_time_render = 0.4 * (end - start) + 0.6 * ema_time_render

                # decouple appearance model
                if(apply_apperance_decouple):
                    decouple_image, transformation_map = decouple_appearance(image, gaussians, cam_info['uid'])
                    # Loss
                    if apply_mask:
                        start = time.time()
                        gt_image = gt_image.cuda()
                        mask = abs(1 - mask.cuda())      
                        Ll1 = l1_loss(decouple_image*mask, gt_image*mask)
                        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image*mask, gt_image*mask))
                    else:
                        start = time.time()
                        gt_image = gt_image.cuda()   
                        Ll1 = l1_loss(decouple_image, gt_image)
                        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                else:
                    # Loss
                    if apply_mask:
                        start = time.time()
                        gt_image = gt_image.cuda()
                        mask = abs(1 - mask.cuda())      
                        Ll1 = l1_loss(image*mask, gt_image*mask)
                        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image*mask, gt_image*mask))
                    else:
                        start = time.time()
                        gt_image = gt_image.cuda()   
                        Ll1 = l1_loss(image, gt_image)
                        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                # depth_loss
                if use_depth_loss:     
                    gt_depth = gt_depth.cuda()
                    rendered_depth = render_pkg["depth"][0]
                    rendered_depth = rendered_depth.reshape(-1, 1)
                    gt_depth = gt_depth.reshape(-1, 1)

                    depth_loss = min(
                        (1 - pearson_corrcoef( - gt_depth, rendered_depth)),
                        (1 - pearson_corrcoef(1 / (gt_depth + 200.), rendered_depth))
                    )
                    loss += opt.lambda_depth * depth_loss

            elif args.type == 'seg':
                objects = render_pkg["render_object"]
                gt_obj_onehot = F.one_hot(gt_obj).permute(2,0,1)    # (C,H,W)
                objects_logit = torch.softmax(objects, dim=0)       # (C,H,W)
                if gt_obj_onehot.shape[0] != objects_logit.shape[0]:
                    margin = objects_logit[:objects_logit.shape[0] - gt_obj_onehot.shape[0]].detach()
                    gt_obj_onehot = torch.cat([gt_obj_onehot, margin], 0)
                loss_sem = F.mse_loss(gt_obj_onehot*mask, objects_logit*mask)
                loss_norm = 100 * ((torch.norm(gaussians._objects_dc, dim=-1, keepdim=True) - 1.0) ** 2).mean()
                loss = loss_sem + loss_norm
            
            loss.backward()
            end = time.time()
            ema_time_loss = 0.4 * (end - start) + 0.6 * ema_time_loss

            iter_end.record()
            ''' 
            #visualize masks
            if iteration % 1000 == 0:
                if not apply_apperance_decouple:
                    #lookup = os.path.join(dataset.model_path, 'vis')
                    lookup = '/home/baihy/cyn/CityGS/output/vis'
                    os.makedirs(lookup, exist_ok=True)
                    torchvision.utils.save_image(torch.cat([image, gt_image, mask], -1), os.path.join(lookup, f'{iteration:05d}.png'))
                else:
                    decouple_image, transformation_map = decouple_appearance(image, gaussians, viewpoint_cam.uid)
                    #lookup = os.path.join(dataset.model_path, 'vis')
                    lookup = '/home/baihy/cyn/CityGS/output/vis'
                    os.makedirs(lookup, exist_ok=True)
                    torchvision.utils.save_image(torch.cat([image, decouple_image, gt_image], -1), os.path.join(lookup, f'{iteration:05d}.png'))
            '''
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                
                grads = gaussians.xyz_gradient_accum / gaussians.denom
                grads[grads.isnan()] = 0.0
                ema_time = {
                    "render": ema_time_render,
                    "loss": ema_time_loss,
                    "densify": ema_time_densify,
                    "num_points": radii.shape[0],
                    "mean_grad": grads.mean().item(),
                }

                lr = {}
                for param_group in gaussians.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                # Log and save
                training_report(dataset, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr,
                                iter_start.elapsed_time(iter_end), testing_iterations, scene, render_large, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, dataset)
                if (iteration in refilter_iterations):
                    print("\n[ITER {}] Refiltering Training Data".format(iteration))
                    gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipe)

                # Densification
                if iteration < opt.densify_until_iter and args.type == '3dgs':
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        start = time.time()
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        end = time.time()
                        ema_time_densify = 0.4 * (end - start) + 0.6 * ema_time_densify

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            iteration += 1
            if iteration >= opt.iterations:
                break

def prepare_output_and_logger(args):    
    if not args.model_path:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        # time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        args.model_path = os.path.join("./output/", config_name)
        if args.block_id >= 0:
            if args.block_id < args.block_dim[0] * args.block_dim[1] * args.block_dim[2]:
                args.model_path = f"{args.model_path}/cells/cell{args.block_id}"
                if args.logger_config is not None:
                    args.logger_config['name'] = f"{args.logger_config['name']}_cell{args.block_id}"
            else:
                raise ValueError("Invalid block_id: {}".format(args.block_id))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # build logger
    log_writer = None
    image_logger = None
    logger_args = {
        "save_dir": args.model_path
    }
    if args.logger_config is None or args.logger_config['logger'] == "tensorboard":
        log_writer = TensorBoardLogger(**logger_args)
        image_logger = tensorboard_log_image
    elif args.logger_config['logger'] == "wandb":
        logger_args.update(name=args.logger_config['name'])
        logger_args.update(project=args.logger_config['project'])
        log_writer = WandbLogger(**logger_args)
        image_logger = wandb_log_image
    else:
        raise ValueError("Unknown logger: {}".format(args.logger_config['logger']))
    
    return log_writer, image_logger

def training_report(dataset, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr, elapsed, testing_iterations, scene : LargeScene, renderFunc, renderArgs):
    if log_writer:
        metrics_to_log = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "train_time/render": ema_time["render"],
            "train_time/loss": ema_time["loss"],
            "train_time/densify": ema_time["densify"],
            "train_time/num_points": ema_time["num_points"],
            "train_time/mean_grad": ema_time["mean_grad"],
            "iter_time": elapsed,
        }
        for key, value in lr.items():
            metrics_to_log["trainer/" + key] = value
        log_writer.log_metrics(metrics_to_log, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, camera in enumerate(config['cameras']):
                    viewpoint_cam = loadCam(dataset, id, camera, 1)
                    viewpoint = {
                        "FoVx": viewpoint_cam.FoVx,
                        "FoVy": viewpoint_cam.FoVy,
                        "image_name": viewpoint_cam.image_name,
                        "image_height": viewpoint_cam.image_height,
                        "image_width": viewpoint_cam.image_width,
                        "camera_center": viewpoint_cam.camera_center,
                        "world_view_transform": viewpoint_cam.world_view_transform,
                        "full_proj_transform": viewpoint_cam.full_proj_transform,
                    }
                    org_img = viewpoint_cam.original_image
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(org_img.to("cuda"), 0.0, 1.0)
                    if log_writer and (idx < 5):
                        grid = torchvision.utils.make_grid(torch.concat([image, gt_image], dim=-1))
                        image_logger(
                            log_writer=log_writer,
                            tag=config['name'] + "_view_{}".format(viewpoint["image_name"]),
                            image_tensor=grid,
                            step=iteration,
                        )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if log_writer:
                    metrics_to_log = {
                        config['name'] + '/loss_viewpoint/l1_loss': l1_test,
                        config['name'] + '/loss_viewpoint/psnr': psnr_test,
                    }
                    log_writer.log_metrics(metrics_to_log, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--type', type=str, default='3dgs', choices=['seg', '3dgs'])
    parser.add_argument('--block_id', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[16_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[16_000, 30_000])
    parser.add_argument("--refilter_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--use_depth_loss", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)
        args.save_iterations.append(op.iterations)
    
    print("Optimizing " + lp.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, args.test_iterations, args.save_iterations, args.refilter_iterations, args.checkpoint_iterations, args.checkpoint, args.debug_from, args.use_depth_loss)

    # All done
    print("\nTraining complete.")
