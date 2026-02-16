import os
import gc
import sys
import yaml
import time
from functools import partial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)
sys.path.append(os.path.join(SUBMODULES_DIR, 'Depth-Anything-V2'))

import uuid
import torch
import numpy as np
import open3d as o3d
from random import randint

from torch import nn
from tqdm import tqdm
from typing import Tuple
from copy import deepcopy
from argparse import ArgumentParser

from fused_ssim import fused_ssim

from base_gs_trainer.Loss.l1 import l1_loss
from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer

from utils.general_utils import inverse_sigmoid
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from gaussian_renderer import render_imp, render_simp, render_depth, render_full

from utils.loss_utils import l1_loss, L1_loss_appearance
from utils.geometry_utils import depth_to_normal
from utils.log_utils import log_training_progress
from regularization.regularizer.depth_order import (
    initialize_depth_order_supervision,
    compute_depth_order_regularization,
)
from regularization.regularizer.mesh import (
    initialize_mesh_regularization,
    compute_mesh_regularization,
    reset_mesh_state_at_next_iteration,
)

from milo.Config.config import ModelParams, PipelineParams, OptimizationParams
from milo.Method.render_kernel import render
from milo.Model.gs import GaussianModel


class Trainer(BaseGSTrainer):
    def __init__(
        self,
        colmap_data_folder_path: str='',
        device: str='cuda:0',
        save_result_folder_path: str='./output/',
        save_log_folder_path: str='./logs/',
        test_freq: int=10000,
        save_freq: int=10000,
    ) -> None:
        # Set up command line argument parser
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)

        # ----- Rasterization technique -----
        parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "gof"])

        # ----- Mesh-In-the-Loop Regularization -----
        parser.add_argument("--no_mesh_regularization", action="store_true")
        parser.add_argument("--mesh_config", type=str, default="default")
        # Gaussians management
        parser.add_argument("--dense_gaussians", action="store_true")

        # ----- Densification and Simplification -----
        # > Inspired by Mini-Splatting2.
        # > Used for pruning, densification and Gaussian pivots selection.
        parser.add_argument("--imp_metric", required=True, type=str, choices=["outdoor", "indoor"])
        parser.add_argument("--config_path", type=str, default="./configs/fast")
        # Aggressive Cloning
        parser.add_argument("--aggressive_clone_from_iter", type=int, default = 500)
        parser.add_argument("--aggressive_clone_interval", type=int, default = 250)
        # Depth Reinitialization
        parser.add_argument("--warn_until_iter", type=int, default = 3_000)
        parser.add_argument("--depth_reinit_iter", type=int, default=2_000)
        parser.add_argument("--num_depth_factor", type=float, default=1)
        # Simplification
        parser.add_argument("--simp_iteration1", type=int, default = 3_000)
        parser.add_argument("--simp_iteration2", type=int, default = 8_000)
        parser.add_argument("--sampling_factor", type=float, default = 0.6)

        # ----- Depth-Normal consistency Regularization -----
        # > Inspired by 2DGS, GOF, RaDe-GS...
        parser.add_argument("--regularization_from_iter", type=int, default = 3_000)
        parser.add_argument("--lambda_depth_normal", type=float, default = 0.05)

        # ----- Depth Order Regularization (Learned Prior) -----
        # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
        # > In the paper, MILo does not rely on any learned prior.
        parser.add_argument("--depth_order", action="store_true")
        parser.add_argument("--depth_order_config", type=str, default="default")

        # ----- 3D Mip Filter -----
        # > Inspired by Mip-Splatting.
        parser.add_argument("--disable_mip_filter", action="store_true", default=False)
        parser.add_argument("--update_mip_filter_every", type=int, default=100)

        # ----- Appearance Network for Exposure-aware loss -----
        # > Inspired by GOF.
        parser.add_argument("--decoupled_appearance", action="store_true")

        # ----- Logging -----
        parser.add_argument("--log_interval", type=int, default=None)

        args = parser.parse_args(sys.argv[1:])

        args.source_path = colmap_data_folder_path
        args.model_path = save_result_folder_path

        args.mesh_regularization = not args.no_mesh_regularization

        # Get depth order config file
        depth_order_config_file = os.path.join(BASE_DIR, "configs", "depth_order", f"{args.depth_order_config}.yaml")
        with open(depth_order_config_file, "r") as f:
            self.depth_order_config = yaml.safe_load(f)

        # ---Prepare Depth-Order Regularization---    
        print("[INFO] Using depth order regularization.")
        print(f"        > Using expected depth with depth_ratio {depth_order_config['depth_ratio']} for depth order regularization.")
        self.depth_priors = initialize_depth_order_supervision(
            scene=self.scene,
            config=self.depth_order_config,
            device='cuda',
        )

        # Get mesh regularization config file
        mesh_config_file = os.path.join(BASE_DIR, "configs", "mesh", f"{args.mesh_config}.yaml")
        with open(mesh_config_file, "r") as f:
            self.mesh_config = yaml.safe_load(f)
        print(f"[INFO] Using mesh regularization with config: {args.mesh_config}")

        # Message for imp_metric
        print(f"[INFO] Using importance metric: {args.imp_metric}.")

        # Import rendering function
        print(f"[INFO] Using {args.rasterizer} as rasterizer.")
        if args.rasterizer == "radegs":
            from gaussian_renderer.radegs import render_radegs as render
            from gaussian_renderer.radegs import integrate_radegs as integrate
        elif args.rasterizer == "gof":
            from gaussian_renderer.gof import render_gof as render
            from gaussian_renderer.gof import integrate_gof as integrate

        print("Optimizing " + args.model_path)

        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

        use_mip_filter = not args.disable_mip_filter

        self.gaussians = GaussianModel(
            sh_degree=0,
            use_mip_filter=use_mip_filter, 
            learn_occupancy=args.mesh_regularization,
            use_appearance_network=args.decoupled_appearance,
        )

        BaseGSTrainer.__init__(
            self,
            colmap_data_folder_path=colmap_data_folder_path,
            device=device,
            save_result_folder_path=save_result_folder_path,
            save_log_folder_path=save_log_folder_path,
            test_freq=test_freq,
            save_freq=save_freq,
        )

        print(f"[INFO] Using 3D Mip Filter: {self.gaussians.use_mip_filter}")
        print(f"[INFO] Using learnable SDF: {self.gaussians.learn_occupancy}")

        if args.dense_gaussians:
            print("[INFO] Using dense Gaussians.")

        # Initialize culling stats
        mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device='cuda')
        self.gaussians.init_culling(len(self.scene.train_cameras))

        # Initialize 3D Mip filter
        if use_mip_filter:
            self.gaussians.compute_3D_filter(cameras=self.scene.train_cameras)

        self.args = args
        return

    def render(self, viewpoint_cam) -> dict:
        return render(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.background,
            require_coord=False,
            require_depth=True,
        )

    def render_full(self, viewpoint_cam) -> dict:
        return render_full(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.background,
            culling=self.gaussians._culling[:,viewpoint_cam.uid],
            compute_expected_normals=False,
            compute_expected_depth=True,
            compute_accurate_median_depth_gradient=True,
        )

    def render_imp(self, viewpoint_cam) -> dict:
        return render_imp(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.background,
            culling=self.gaussians._culling[:,viewpoint_cam.uid],
        )

    def trainStep(
        self,
        iteration: int,
        viewpoint_cam,
        lambda_dssim: float = 0.2,
        lambda_normal: float = 0.01,
        lambda_dist: float = 0.01,
        lambda_opacity: float = 0.01,
        lambda_scaling: float = 1.0,
    ) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        reg_kick_on = iteration >= self.args.regularization_from_iter
        mesh_kick_on = iteration >= self.mesh_config["start_iter"]
        depth_order_kick_on = True

        # If depth-normal regularization or mesh-in-the-loop regularization are active,
        # we use the rasterizer compatible with depth and normal rendering.
        if reg_kick_on or mesh_kick_on:
            render_pkg = self.render()

        # Else, if depth-order regularization is active, we use Mini-Splatting2 rasterizer 
        # but we render depth maps. This rasterizer is necessary for densification and simplification.
        else:
            render_pkg = self.render_full()

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        gt_image = viewpoint_cam.original_image.cuda()

        # Rendering loss
        if self.args.decoupled_appearance:
            reg_loss = L1_loss_appearance(image, gt_image, self.gaussians, viewpoint_cam.uid)
        else:
            reg_loss = l1_loss(image, gt_image)

        reg_loss = l1_loss(image, gt_image)
        ssim_loss = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        rgb_loss = (1.0 - lambda_dssim) * reg_loss + lambda_dssim * ssim_loss

        # Depth-Normal Consistency Regularization
        depth_normal_loss = torch.zeros(1, device=self.device)
        if reg_kick_on:
            rendered_depth_to_normals: torch.Tensor = depth_to_normal(
                viewpoint_cam,
                render_pkg["median_depth"],  # 1, H, W
                render_pkg["expected_depth"],  # 1, H, W
            )  # 3, H, W or 2, 3, H, W
            rendered_normals: torch.Tensor = render_pkg["normal"]  # 3, H, W

            if rendered_depth_to_normals.ndim == 4:
                # If shape is 2, 3, H, W
                reg_depth_ratio = 0.6
                normal_error_map = 1. - (rendered_normals[None] * rendered_depth_to_normals).sum(dim=1)  # 2, H, W
                depth_normal_loss = self.args.lambda_depth_normal * (
                    (1. - reg_depth_ratio) * normal_error_map[0].mean() 
                    + reg_depth_ratio * normal_error_map[1].mean()
                )
            else:
                # If shape is 3, H, W
                depth_normal_loss = self.args.lambda_depth_normal * (1 - (rendered_normals * rendered_depth_to_normals).sum(dim=0)).mean()

        loss = \
            rgb_loss + \
            depth_normal_loss

        # Depth Order Regularization
        # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
        # > In the paper, MILo does not rely on any learned prior. 
        if depth_order_kick_on:
            if self.depth_order_config["depth_ratio"] < 1.:
                depth_for_depth_order = (
                    (1. - self.depth_order_config["depth_ratio"]) * render_pkg["expected_depth"]
                    + self.depth_order_config["depth_ratio"] * render_pkg["median_depth"]
                )
            else:
                depth_for_depth_order = render_pkg["median_depth"]

            depth_prior_loss, _, do_supervision_depth, lambda_depth_order = compute_depth_order_regularization(
                iteration=iteration,
                rendered_depth=depth_for_depth_order,
                depth_priors=self.depth_priors,
                viewpoint_idx=viewpoint_idx,
                gaussians=self.gaussians,
                config=self.depth_order_config,
            )

            loss = loss + depth_prior_loss
            depth_order_kick_on = lambda_depth_order > 0

        # Mesh-In-the-Loop Regularization
        if mesh_kick_on:
            mesh_regularization_pkg = compute_mesh_regularization(
                iteration=iteration,
                render_pkg=render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=self.gaussians,
                scene=self.scene,
                pipe=self.pipe,
                background=self.background,
                kernel_size=0.0,
                config=self.mesh_config,
                mesh_renderer=mesh_renderer,
                mesh_state=mesh_state,
                render_func=partial(render, require_coord=False, require_depth=True),
                weight_adjustment=100. / opt.iterations,
                args=args,
                integrate_func=integrate,
            )
            mesh_loss = mesh_regularization_pkg["mesh_loss"]
            mesh_depth_loss = mesh_regularization_pkg["mesh_depth_loss"]
            mesh_normal_loss = mesh_regularization_pkg["mesh_normal_loss"]
            occupied_centers_loss = mesh_regularization_pkg["occupied_centers_loss"]
            occupancy_labels_loss = mesh_regularization_pkg["occupancy_labels_loss"]
            mesh_state = mesh_regularization_pkg["updated_state"]
            mesh_render_pkg = mesh_regularization_pkg["mesh_render_pkg"]
            
            loss = loss + mesh_loss
        
        # ---Backward pass---
        loss.backward()

        reg_loss = l1_loss(image, gt_image)
        ssim_loss = 1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        rgb_loss = (1.0 - lambda_dssim) * reg_loss + lambda_dssim * ssim_loss

        lambda_normal = lambda_normal if iteration > 1000 else 0.0
        normal_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_normal > 0:
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_dot = torch.sum(rend_normal * surf_normal, dim=0)

            valid_dot_idxs = torch.where(normal_dot != 0)
            valid_normal_dot = normal_dot[valid_dot_idxs]

            normal_error = (1 - valid_normal_dot)
            normal_loss = lambda_normal * normal_error.mean()

        lambda_dist = lambda_dist if iteration > 1000 else 0.0
        dist_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_dist > 0:
            rend_dist = render_pkg["rend_dist"]
            valid_dist_idxs = torch.where(rend_dist != 0)
            valid_rend_dist = rend_dist[valid_dist_idxs]
            dist_loss = lambda_dist * valid_rend_dist.mean()

        # Phase A: Surface selection — push non-surface to 0 (opacity, scale)；Phase B 开始后彻底关闭
        opacity_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_opacity > 0:
            opacity_loss = lambda_opacity * nn.MSELoss()(self.gaussians.get_opacity, torch.zeros_like(self.gaussians._opacity))

        scaling_loss = torch.zeros([1], dtype=rgb_loss.dtype).to(rgb_loss.device)
        if lambda_scaling > 0:
            scaling_loss = lambda_scaling * nn.MSELoss()(self.gaussians.get_scaling, torch.zeros_like(self.gaussians._scaling))

        # loss
        total_loss = rgb_loss + dist_loss + normal_loss + opacity_loss + scaling_loss

        total_loss.backward()

        loss_dict = {
            'reg': reg_loss.item(),
            'ssim': ssim_loss.item(),
            'rgb': rgb_loss.item(),
            'dist': dist_loss.item(),
            'normal': normal_loss.item(),
            'opacity': opacity_loss.item(),
            'scaling': scaling_loss.item(),
            'total': total_loss.item(),
        }

        return render_pkg, loss_dict

    @torch.no_grad()
    def recordGrads(self, render_pkg: dict) -> bool:
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        return True

    @torch.no_grad()
    def densifyStep(self, render_pkg: dict) -> bool:
        size_threshold = 20
        my_viewpoint_stack = self.scene.train_cameras
        camlist = sampling_cameras(my_viewpoint_stack)

        # The multiview consistent densification of fastgs
        importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, self.gaussians, self.pipe, self.background, self.opt, DENSIFY=True)
        self.gaussians.densify_and_prune_fastgs(
            max_screen_size = size_threshold,
            min_opacity = 0.005,
            extent = self.scene.cameras_extent,
            radii=render_pkg['radii'],
            args = self.opt,
            importance_score = importance_score,
            pruning_score = pruning_score,
        )
        return True

    @torch.no_grad()
    def resetOpacity(self) -> bool:
        self.gaussians.reset_opacity()
        return True

    @torch.no_grad()
    def resetScaling(self) -> bool:
        self.gaussians.reset_scaling()
        return True

    @torch.no_grad()
    def updateGSParams(self, iteration: int) -> bool:
        self.gaussians.optimizer_step(iteration)
        return True

    @torch.no_grad()
    def saveScene(self, iteration: int) -> bool:
        point_cloud_path = os.path.join(self.dataset.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        return True

    def train(self, iteration_num: int = 30000):
        # Additional variables
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        viewpoint_stack = None
        postfix_dict = {}
        ema_loss_for_log = 0.0
        ema_depth_normal_loss_for_log = 0.0

        # ---Prepare Mesh-In-the-Loop Regularization---
        if self.args.mesh_regularization:
            print("[INFO] Using mesh regularization.")
            mesh_renderer, mesh_state = initialize_mesh_regularization(
                scene=self.scene,
                config=self.mesh_config,
            )
        ema_mesh_depth_loss_for_log = 0.0
        ema_mesh_normal_loss_for_log = 0.0
        ema_occupied_centers_loss_for_log = 0.0
        ema_occupancy_labels_loss_for_log = 0.0

        ema_depth_order_loss_for_log = 0.0

        # ---Log optimizable param groups---
        print(f"[INFO] Found {len(self.gaussians.optimizer.param_groups)} optimizable param groups:")
        n_total_params = 0
        for param in self.gaussians.optimizer.param_groups:
            name = param['name']
            n_params = len(param['params'])
            print(f"\n========== {name} ==========")
            print(f"Total number of param groups: {n_params}")
            for param_i in param['params']:
                print(f"   > Shape {param_i.shape}")
                n_total_params = n_total_params + param_i.numel()
        if self.gaussians.learn_occupancy:
            print(f"\n========== base_occupancy ==========")
            print(f"   > Not learnable")
            print(f"   > Shape {self.gaussians._base_occupancy.shape}")
        print(f"\nTotal number of optimizable parameters: {n_total_params}\n")

        # ---Start optimization loop---    
        progress_bar = tqdm(desc="Training progress", total=iteration_num)
        iteration = 0
        for _ in range(iteration_num):
            iteration += 1

            viewpoint_cam = self.scene[iteration]

            render_pkg, loss_dict = self.trainStep(iteration, viewpoint_cam)

            if iteration % 10 == 0:
                bar_loss_dict = {
                    "rgb": f"{loss_dict['rgb']:.{5}f}",
                    "distort": f"{loss_dict['dist']:.{5}f}",
                    "normal": f"{loss_dict['normal']:.{5}f}",
                    "Points": f"{len(self.gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(bar_loss_dict)
                progress_bar.update(10)

            self.logStep(iteration, loss_dict, is_fast=True)

            if iteration % self.save_freq == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                self.saveScene(iteration)

            # Densification
            if iteration < self.opt.densify_until_iter:
                self.recordGrads(render_pkg)
                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                    self.densifyStep(render_pkg)

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.resetOpacity()

                if iteration % self.opt.scaling_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.resetScaling()

            # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
            # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
            # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                self.finalPrune()

            # 每个 step 删除不在任一 mask 内的 gaussian
            # self.pruneGaussiansOutsideMasks()

            self.updateGSParams(iteration)

            self.iteration = iteration
        return True

    def exportMesh(
        self,
        mesh_res: int = 1024,
        voxel_size: float = -1.0,
        depth_trunc: float = -1.0,
        sdf_trunc: float = -1.0,
        num_cluster: int = 50,
    ) -> bool:
        export_scene = deepcopy(self.scene)
        export_gaussians = export_scene.gaussians

        train_dir = os.path.join(self.dataset.model_path, 'mesh', 'iter_' + str(self.iteration))
        gaussExtractor = GaussianExtractor(export_gaussians, render, self.pipe, bg_color=self.background.cpu().numpy())

        print('[INFO][Trainer::exportMesh]')
        print("\t start export mesh...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(export_scene.train_cameras)

        # extract the mesh and save
        name = 'fuse.ply'
        depth_trunc = (gaussExtractor.radius * 2.0) if depth_trunc < 0  else depth_trunc
        voxel_size = (depth_trunc / mesh_res) if voxel_size < 0 else voxel_size
        sdf_trunc = 5.0 * voxel_size if sdf_trunc < 0 else sdf_trunc
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
        return True
