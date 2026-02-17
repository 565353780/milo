import os
import gc
import sys
import yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../..'
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)
sys.path.append(os.path.join(SUBMODULES_DIR, 'Depth-Anything-V2'))

import torch

from typing import Tuple
from functools import partial
from tqdm import tqdm, trange
from argparse import ArgumentParser

from fused_ssim import fused_ssim

from base_gs_trainer.Loss.l1 import l1_loss
from base_gs_trainer.Module.base_gs_trainer import BaseGSTrainer

from gaussian_renderer import render_imp, render_simp, render_depth, render_full

from utils.log_utils import fix_normal_map
from utils.geometry_utils import depth_to_normal
from utils.loss_utils import L1_loss_appearance
from regularization.regularizer.depth_order import (
    initialize_depth_order_supervision,
    compute_depth_order_regularization,
)
from regularization.regularizer.mesh import (
    initialize_mesh_regularization,
    compute_mesh_regularization,
    reset_mesh_state_at_next_iteration,
)

# Import rendering function
rasterizer = 'radegs'
print(f"[INFO] Using {rasterizer} as rasterizer.")
if rasterizer == "radegs":
    from gaussian_renderer.radegs import render_radegs as render
    from gaussian_renderer.radegs import integrate_radegs as integrate
elif rasterizer == "gof":
    from gaussian_renderer.gof import render_gof as render
    from gaussian_renderer.gof import integrate_gof as integrate
else:
    print('rasterizer must in [radegs, gof]!')
    exit()

from milo.Config.config import ModelParams, PipelineParams, OptimizationParams
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
        args.imp_metric = 'indoor'

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

        # Get depth order config file
        depth_order_config_file = os.path.join(BASE_DIR, "configs", "depth_order", f"{args.depth_order_config}.yaml")
        with open(depth_order_config_file, "r") as f:
            self.depth_order_config = yaml.safe_load(f)

        # ---Prepare Depth-Order Regularization---    
        print("[INFO] Using depth order regularization.")
        print(f"        > Using expected depth with depth_ratio {self.depth_order_config['depth_ratio']} for depth order regularization.")
        self.depth_priors = []
        for camera in self.scene.train_cameras:
            depth = camera._cam.depth.detach().clone()
            self.depth_priors.append(depth)

        # ---Prepare Mesh-In-the-Loop Regularization---
        if self.args.mesh_regularization:
            print("[INFO] Using mesh regularization.")
            self.mesh_renderer, self.mesh_state = initialize_mesh_regularization(
                scene=self.scene,
                config=self.mesh_config,
            )

        # Get mesh regularization config file
        mesh_config_file = os.path.join(BASE_DIR, "configs", "mesh", f"{args.mesh_config}.yaml")
        with open(mesh_config_file, "r") as f:
            self.mesh_config = yaml.safe_load(f)
        print(f"[INFO] Using mesh regularization with config: {args.mesh_config}")

        # Message for imp_metric
        print(f"[INFO] Using importance metric: {args.imp_metric}.")
        print(f"[INFO] Using 3D Mip Filter: {self.gaussians.use_mip_filter}")
        print(f"[INFO] Using learnable SDF: {self.gaussians.learn_occupancy}")

        if args.dense_gaussians:
            print("[INFO] Using dense Gaussians.")

        # Initialize culling stats
        self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device='cuda')
        self.gaussians.init_culling(len(self.scene.train_cameras))

        # Initialize 3D Mip filter
        self.compute_3D_filter()

        self.args = args
        self.depth_order_kick_on = True
        return

    def renderImage(self, viewpoint_cam) -> dict:
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
    ) -> Tuple[dict, dict]:
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        viewpoint_idx = iteration % len(self.scene)

        viewpoint_cam = self.scene(viewpoint_idx)
        depth_prior = self.depth_priors[viewpoint_idx]

        reg_kick_on = iteration >= self.args.regularization_from_iter
        mesh_kick_on = iteration >= self.mesh_config["start_iter"]

        # If depth-normal regularization or mesh-in-the-loop regularization are active,
        # we use the rasterizer compatible with depth and normal rendering.
        if reg_kick_on or mesh_kick_on:
            render_pkg = self.renderImage(viewpoint_cam)

        # Else, if depth-order regularization is active, we use Mini-Splatting2 rasterizer 
        # but we render depth maps. This rasterizer is necessary for densification and simplification.
        else:
            render_pkg = self.render_full(viewpoint_cam)

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
        rgb_loss = (1.0 - self.opt.lambda_dssim) * reg_loss + self.opt.lambda_dssim * ssim_loss

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

        # Depth Order Regularization
        # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
        # > In the paper, MILo does not rely on any learned prior. 
        depth_prior_loss = torch.zeros(1, device=self.device)
        if self.depth_order_kick_on:
            if self.depth_order_config["depth_ratio"] < 1.:
                depth_for_depth_order = (
                    (1. - self.depth_order_config["depth_ratio"]) * render_pkg["expected_depth"]
                    + self.depth_order_config["depth_ratio"] * render_pkg["median_depth"]
                )
            else:
                depth_for_depth_order = render_pkg["median_depth"]

            depth_prior_loss, _, _, lambda_depth_order = compute_depth_order_regularization(
                iteration=iteration,
                rendered_depth=depth_for_depth_order,
                depth_prior=depth_prior,
                gaussians=self.gaussians,
                config=self.depth_order_config,
            )

            self.depth_order_kick_on = lambda_depth_order > 0

        # Mesh-In-the-Loop Regularization
        mesh_loss = torch.zeros(1, device=self.device)
        mesh_depth_loss = torch.zeros(1, device=self.device)
        mesh_normal_loss = torch.zeros(1, device=self.device)
        occupied_centers_loss = torch.zeros(1, device=self.device)
        occupancy_labels_loss = torch.zeros(1, device=self.device)
        mesh_render_pkg = {}
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
                mesh_renderer=self.mesh_renderer,
                mesh_state=self.mesh_state,
                render_func=partial(render, require_coord=False, require_depth=True),
                weight_adjustment=100. / self.opt.iterations,
                args=self.args,
                integrate_func=integrate,
            )
            mesh_loss = mesh_regularization_pkg["mesh_loss"]
            mesh_depth_loss = mesh_regularization_pkg["mesh_depth_loss"]
            mesh_normal_loss = mesh_regularization_pkg["mesh_normal_loss"]
            occupied_centers_loss = mesh_regularization_pkg["occupied_centers_loss"]
            occupancy_labels_loss = mesh_regularization_pkg["occupancy_labels_loss"]
            self.mesh_state = mesh_regularization_pkg["updated_state"]
            mesh_render_pkg = mesh_regularization_pkg["mesh_render_pkg"]

        total_loss = \
            rgb_loss + \
            depth_normal_loss + \
            depth_prior_loss + \
            mesh_loss

        # ---Backward pass---
        total_loss.backward()

        loss_dict = {
            'reg': reg_loss.item(),
            'ssim': ssim_loss.item(),
            'rgb': rgb_loss.item(),
            'depth_normal': depth_normal_loss.item(),
            'depth_prior': depth_prior_loss.item(),
            'mesh': mesh_loss.item(),
            'mesh_depth': mesh_depth_loss.item(),
            'mesh_normal': mesh_normal_loss.item(),
            'occupied_centers': occupied_centers_loss.item(),
            'occupancy_labels': occupancy_labels_loss.item(),
            'total': total_loss.item(),
        }

        return loss_dict, render_pkg

    @torch.no_grad()
    def recordGrads(self, iteration: int, render_pkg: dict) -> bool:
        viewpoint_cam = self.scene[iteration]

        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Keep track of max radii in image-space for pruning
        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

        if self.gaussians._culling[:,viewpoint_cam.uid].sum()==0:
            self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        else:
            # normalize xy gradient after culling
            self.gaussians.add_densification_stats_culling(viewspace_point_tensor, visibility_filter, self.gaussians.factor_culling)
        return True

    def compute_3D_filter(self) -> bool:
        if self.gaussians.use_mip_filter:
            self.gaussians.compute_3D_filter(cameras=self.scene.train_cameras)
        return True

    def culling(self, iteration: int) -> bool:
        if self.args.dense_gaussians:
            self.gaussians.culling_with_importance_pruning(self.scene, render_simp, iteration, self.args, self.pipe, self.background)
        else:
            self.gaussians.culling_with_interesction_sampling(self.scene, render_simp, iteration, self.args, self.pipe, self.background)
        return True

    def update_mask_blur(self, render_pkg: dict) -> bool:
        image = render_pkg['render']
        area_max = render_pkg["area_max"]
        self.mask_blur = torch.logical_or(self.mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))
        return True

    @torch.no_grad()
    def densifyStep(self, iteration: int, render_pkg: dict) -> bool:
        image, area_max = render_pkg['render'], render_pkg["area_max"]

        self.mask_blur = torch.logical_or(self.mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))

        if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0 and iteration != self.args.depth_reinit_iter:
            size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
            self.gaussians.densify_and_prune_mask(
                self.opt.densify_grad_threshold, 
                0.005, self.scene.cameras_extent, 
                size_threshold, self.mask_blur)

            self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device='cuda')

            self.compute_3D_filter()
        return True

    @torch.no_grad()
    def resetOpacity(self) -> bool:
        self.gaussians.reset_opacity()
        return True

    @torch.no_grad()
    def updateGSParams(self, render_pkg: dict) -> bool:
        if self.gaussians.use_appearance_network:
            self.gaussians.optimizer.step()
        else:
            radii = render_pkg['radii']
            visible = radii>0
            self.gaussians.optimizer.step(visible, radii.shape[0])
        self.gaussians.optimizer.zero_grad(set_to_none = True)
        return True

    @torch.no_grad()
    def logImageStep(
        self,
        iteration: int,
        render_image_num: int=1,
        is_fast: bool=True,
    ) -> bool:
        reg_kick_on = iteration >= self.args.regularization_from_iter
        mesh_kick_on = iteration >= self.mesh_config["start_iter"]

        print('[INFO][Trainer::logImageStep]')
        print('\t start log extra images...')
        for idx in trange(render_image_num):
            viewpoint = self.scene[idx]

            render_pkg = self.render(viewpoint)

            mesh_regularization_pkg = compute_mesh_regularization(
                iteration=iteration,
                render_pkg=render_pkg,
                viewpoint_cam=viewpoint,
                viewpoint_idx=idx,
                gaussians=self.gaussians,
                scene=self.scene,
                pipe=self.pipe,
                background=self.background,
                kernel_size=0.0,
                config=self.mesh_config,
                mesh_renderer=self.mesh_renderer,
                mesh_state=self.mesh_state,
                render_func=partial(render, require_coord=False, require_depth=True),
                weight_adjustment=100. / self.opt.iterations,
                args=self.args,
                integrate_func=integrate,
            )
            mesh_render_pkg = mesh_regularization_pkg['mesh_render_pkg']

            if not self.is_gt_logged:
                depth_prior = self.depth_priors[idx]
                self.logger.summary_writer.add_images("view_{}/depth_prior".format(viewpoint.image_name), depth_prior[None], global_step=iteration)

                depth_prior_normal = (1. - depth_to_normal(viewpoint, depth_prior)) / 2.
                self.logger.summary_writer.add_images("view_{}/depth_prior_normal".format(viewpoint.image_name), depth_prior_normal[None], global_step=iteration)

            if reg_kick_on or mesh_kick_on or self.depth_order_kick_on:
                render_depth = render_pkg['median_depth']
                self.logger.summary_writer.add_images("view_{}/render_depth".format(viewpoint.image_name), render_depth[None], global_step=iteration)

            if reg_kick_on or mesh_kick_on:
                render_normal = (1. - render_pkg["normal"]) / 2.
                self.logger.summary_writer.add_images("view_{}/render_normal".format(viewpoint.image_name), render_normal[None], global_step=iteration)

            if mesh_kick_on:
                mesh_depth = torch.where(
                    mesh_render_pkg["depth"].detach() > 0,
                    mesh_render_pkg["depth"].detach(),
                    mesh_render_pkg["depth"].detach().max().item(),
                )
                self.logger.summary_writer.add_images("view_{}/mesh_depth".format(viewpoint.image_name), mesh_depth[None], global_step=iteration)

                mesh_normal = (1. - fix_normal_map(viewpoint, mesh_render_pkg["normals"].detach())) / 2.
                self.logger.summary_writer.add_images("view_{}/mesh_normal".format(viewpoint.image_name), mesh_normal[None], global_step=iteration)
 
        BaseGSTrainer.logImageStep(
            self,
            iteration=iteration,
            render_image_num=render_image_num,
            is_fast=is_fast,
        )
        return True

    @torch.no_grad()
    def saveScene(self, iteration: int) -> bool:
        torch.save((self.gaussians.capture(), iteration), self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")  
        return True

    def train(self, iteration_num: int = 30000):
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

            loss_dict, render_pkg = self.trainStep(iteration)

            if iteration % 10 == 0:
                bar_loss_dict = {
                    "rgb": f"{loss_dict['rgb']:.{5}f}",
                    "distort": f"{loss_dict['dist']:.{5}f}",
                    "normal": f"{loss_dict['normal']:.{5}f}",
                    "Points": f"{len(self.gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(bar_loss_dict)
                progress_bar.update(10)

            self.logStep(iteration, loss_dict)

            if iteration % self.test_freq == 0:
                self.logImageStep(
                    iteration,
                    render_image_num=1,
                    is_fast=True,
                )

            # ---Densification---
            gaussians_have_changed = False
            if iteration < self.opt.densify_until_iter:
                self.recordGrads(iteration, render_pkg)

                self.update_mask_blur(render_pkg)

                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0 and iteration != self.args.depth_reinit_iter:
                    self.densifyStep(iteration, render_pkg)
                    gaussians_have_changed = True

                if iteration == self.args.depth_reinit_iter:
                    num_depth = self.gaussians._xyz.shape[0]*self.args.num_depth_factor

                    # interesction_preserving for better point cloud reconstruction result at the early stage, not affect rendering quality
                    self.gaussians.interesction_preserving(self.scene, render_simp, iteration, self.args, self.pipe, self.background)
                    self.compute_3D_filter()

                    pts, rgb = self.gaussians.depth_reinit(self.scene, render_depth, iteration, num_depth, self.args, self.pipe, self.background)

                    self.gaussians.reinitial_pts(pts, rgb)

                    self.gaussians.training_setup(self.opt)
                    self.gaussians.init_culling(len(self.scene))

                    self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device='cuda')
                    torch.cuda.empty_cache()
                    gaussians_have_changed = True
                    self.compute_3D_filter()

                if iteration >= self.args.aggressive_clone_from_iter and iteration % self.args.aggressive_clone_interval == 0 and iteration!=self.args.depth_reinit_iter:
                    self.gaussians.culling_with_clone(self.scene, render_simp, iteration, self.args, self.pipe, self.background)
                    torch.cuda.empty_cache()

                    self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device='cuda')
                    gaussians_have_changed = True
                    self.compute_3D_filter()

            # ---Pruning and simplification---
            if iteration == self.args.simp_iteration1:
                self.culling(iteration)

                self.gaussians.max_sh_degree=self.dataset.sh_degree
                self.gaussians.extend_features_rest()

                self.gaussians.training_setup(self.opt)
                torch.cuda.empty_cache()
                gaussians_have_changed = True
                self.compute_3D_filter()

            if iteration == self.args.simp_iteration2:
                self.culling(iteration)
                torch.cuda.empty_cache()
                gaussians_have_changed = True
                self.compute_3D_filter()

            if iteration == (self.args.simp_iteration2+self.opt.iterations)//2:
                self.gaussians.init_culling(len(self.scene))

            # ---Reset mesh state if Gaussians have changed---
            mesh_kick_on = iteration >= self.mesh_config["start_iter"]
            if mesh_kick_on and gaussians_have_changed:
                self.mesh_state = reset_mesh_state_at_next_iteration(self.mesh_state)

            # ---Update 3D Mip Filter---
            if self.gaussians.use_mip_filter and (
                (iteration == self.args.warn_until_iter)
                or (iteration % self.args.update_mip_filter_every == 0)
            ):
                if iteration < self.opt.iterations - self.args.update_mip_filter_every:
                    self.compute_3D_filter()
                else:
                    print(f"[INFO] Skipping 3D Mip Filter update at iteration {iteration}")

            # ---Optimizer step---
            self.updateGSParams(render_pkg)

            # ---Save checkpoint---
            if iteration % self.save_freq == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                self.saveScene(iteration)

        if iteration % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return True
