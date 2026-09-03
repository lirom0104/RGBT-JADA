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

import os
import torch
from random import randint
from torchvision.utils import save_image
from utils.loss_utils import build_reconstruction_criterion, edge_aware_tv_loss, l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from lpipsPyTorch.modules.lpips import LPIPS


lpips_model = LPIPS(net_type='vgg').cuda()
for param in lpips_model.parameters():
    param.requires_grad = False

import time
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def _prefix_metrics(prefix, metrics):
    return {f"{prefix}/{metric_name}": metric_value for metric_name, metric_value in metrics.items()}

def _build_anchor_contribution_proxy(radii, opacity):
    return radii.detach().clamp_min(0.0) * opacity.detach().reshape(-1)

def _build_pairing_runtime_metrics(use_paired_views, pairing_summary, attempts, hits, sampling_fallbacks):
    hit_rate = (hits / attempts) if attempts > 0 else 0.0
    return {
        "paired_view_enabled": float(bool(use_paired_views)),
        "total_cameras_rgb": float(pairing_summary.get("total_cameras_rgb", 0)),
        "total_cameras_thermal": float(pairing_summary.get("total_cameras_thermal", 0)),
        "paired_camera_count": float(pairing_summary.get("paired_camera_count", 0)),
        "paired_sampling_hit_rate": hit_rate if use_paired_views else 0.0,
        "paired_sampling_fallback_count": float(sampling_fallbacks),
        "paired_camera_fallback_count": float(pairing_summary.get("fallback_count", 0)),
    }

def _build_adaptive_weight_state():
    return {
        "ema_rgb": None,
        "ema_th": None,
        "ref_rgb": None,
        "ref_th": None,
    }

def _restore_adaptive_weight_state(state):
    restored = _build_adaptive_weight_state()
    if not isinstance(state, dict):
        return restored
    for key in restored.keys():
        value = state.get(key)
        restored[key] = None if value is None else float(value)
    return restored

def _compute_adaptive_branch_weights(iteration, recon_rgb, recon_th, opt, state):
    adaptive_enabled = bool(getattr(opt, "adaptive_branch_weight", False))
    ema_decay = max(0.0, min(float(getattr(opt, "adaptive_branch_weight_ema", 0.98)), 0.9999))
    warmup_iters = max(0, int(getattr(opt, "adaptive_branch_weight_warmup_iters", 0)))
    ramp_iters = max(1, int(getattr(opt, "adaptive_branch_weight_ramp_iters", 1)))
    beta = max(0.0, float(getattr(opt, "adaptive_branch_weight_beta", 1.0)))
    min_weight = max(0.0, min(float(getattr(opt, "adaptive_branch_weight_min", 0.0)), 0.49))
    eps = 1e-6

    recon_rgb_value = float(recon_rgb.detach().item())
    recon_th_value = float(recon_th.detach().item())

    if state["ema_rgb"] is None or state["ema_th"] is None:
        state["ema_rgb"] = recon_rgb_value
        state["ema_th"] = recon_th_value
    else:
        state["ema_rgb"] = ema_decay * state["ema_rgb"] + (1.0 - ema_decay) * recon_rgb_value
        state["ema_th"] = ema_decay * state["ema_th"] + (1.0 - ema_decay) * recon_th_value

    if iteration >= warmup_iters and (state["ref_rgb"] is None or state["ref_th"] is None):
        state["ref_rgb"] = max(state["ema_rgb"], eps)
        state["ref_th"] = max(state["ema_th"], eps)

    w_rgb = 0.5
    w_th = 0.5
    raw_w_rgb = 0.5
    raw_w_th = 0.5
    norm_rgb = 1.0
    norm_th = 1.0
    alpha = 0.0

    if adaptive_enabled and state["ref_rgb"] is not None and state["ref_th"] is not None:
        norm_rgb = state["ema_rgb"] / max(state["ref_rgb"], eps)
        norm_th = state["ema_th"] / max(state["ref_th"], eps)
        logits = torch.tensor(
            (-beta * norm_rgb, -beta * norm_th),
            dtype=torch.float32,
        )
        raw_weights = torch.softmax(logits, dim=0)
        raw_w_rgb = float(raw_weights[0].item())
        raw_w_th = float(raw_weights[1].item())
        if iteration > warmup_iters:
            alpha = min(1.0, (iteration - warmup_iters) / float(ramp_iters))
        w_rgb = (1.0 - alpha) * 0.5 + alpha * raw_w_rgb
        w_rgb = max(min_weight, min(1.0 - min_weight, w_rgb))
        w_th = 1.0 - w_rgb

    metrics = {
        "enabled": float(adaptive_enabled),
        "weight_rgb": w_rgb,
        "weight_th": w_th,
        "raw_weight_rgb": raw_w_rgb,
        "raw_weight_th": raw_w_th,
        "ema_rgb": state["ema_rgb"],
        "ema_th": state["ema_th"],
        "ref_rgb": 0.0 if state["ref_rgb"] is None else state["ref_rgb"],
        "ref_th": 0.0 if state["ref_th"] is None else state["ref_th"],
        "norm_rgb": norm_rgb,
        "norm_th": norm_th,
        "blend_alpha": alpha,
    }
    return w_rgb, w_th, metrics

def _linear_ramp(iteration, start_iter, end_iter):
    start_iter = int(start_iter)
    end_iter = int(end_iter)
    if iteration <= start_iter:
        return 0.0
    if iteration >= end_iter:
        return 1.0
    return (iteration - start_iter) / float(max(1, end_iter - start_iter))

def _decayed_weight(iteration, start_weight, final_weight, start_iter, end_iter):
    alpha = _linear_ramp(iteration, start_iter, end_iter)
    return (1.0 - alpha) * float(start_weight) + alpha * float(final_weight)

def _apply_late_color_priority(iteration, w_rgb, opt):
    if not getattr(opt, "late_color_priority", False):
        return w_rgb, 1.0 - w_rgb, 0.0

    start_iter = int(getattr(opt, "late_color_priority_start_iter", 0))
    ramp_iters = int(getattr(opt, "late_color_priority_ramp_iters", 1))
    alpha = _linear_ramp(iteration, start_iter, start_iter + max(1, ramp_iters))
    target_rgb = float(getattr(opt, "late_color_priority_rgb_weight", w_rgb))
    min_weight = max(0.0, min(float(getattr(opt, "adaptive_branch_weight_min", 0.0)), 0.49))
    target_rgb = max(min_weight, min(1.0 - min_weight, target_rgb))
    w_rgb = (1.0 - alpha) * w_rgb + alpha * target_rgb
    return w_rgb, 1.0 - w_rgb, alpha

def _named_ema_sources(gaussians):
    attr_names = (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_thermal_dc",
        "_thermal_rest",
        "_scaling",
        "_rotation",
        "_opacity_base",
        "_at_gom_opacity_bias_th",
        "_at_gom_center_residual",
        "_at_gom_log_scale_residual",
        "_render_calibration_color_scale",
        "_render_calibration_color_bias",
        "_render_calibration_thermal_scale",
        "_render_calibration_thermal_bias",
    )
    sources = {}
    for attr_name in attr_names:
        value = getattr(gaussians, attr_name, None)
        if torch.is_tensor(value) and value.numel() > 0:
            sources[f"attr:{attr_name}"] = value

    for module_name in ("bgfc", "color_refinement"):
        module = getattr(gaussians, module_name, None)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            if torch.is_tensor(value) and value.is_floating_point():
                sources[f"module:{module_name}:{key}"] = value
    return sources

def _update_ema_export_state(ema_state, gaussians, decay):
    sources = _named_ema_sources(gaussians)
    if not sources:
        return None

    should_reset = ema_state is None or set(ema_state.keys()) != set(sources.keys())
    if not should_reset:
        for key, value in sources.items():
            if ema_state[key].shape != value.shape:
                should_reset = True
                break

    if should_reset:
        return {key: value.detach().clone() for key, value in sources.items()}

    decay = max(0.0, min(float(decay), 0.9999))
    for key, value in sources.items():
        ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
    return ema_state

def _apply_ema_export_state(ema_state, gaussians):
    if not ema_state:
        return None

    backup = {}
    module_updates = {}
    module_backups = {}
    for key, ema_value in ema_state.items():
        if key.startswith("attr:"):
            attr_name = key.split(":", 1)[1]
            value = getattr(gaussians, attr_name, None)
            if torch.is_tensor(value) and value.shape == ema_value.shape:
                backup[key] = value.detach().clone()
                value.data.copy_(ema_value.to(device=value.device, dtype=value.dtype))
        elif key.startswith("module:"):
            _, module_name, state_key = key.split(":", 2)
            module_updates.setdefault(module_name, {})[state_key] = ema_value

    for module_name, updates in module_updates.items():
        module = getattr(gaussians, module_name, None)
        if module is None:
            continue
        current_state = module.state_dict()
        module_backups[module_name] = {
            state_key: value.detach().clone()
            for state_key, value in current_state.items()
            if torch.is_tensor(value)
        }
        new_state = {}
        for state_key, value in current_state.items():
            ema_value = updates.get(state_key)
            if ema_value is not None and value.shape == ema_value.shape:
                new_state[state_key] = ema_value.to(device=value.device, dtype=value.dtype)
            else:
                new_state[state_key] = value
        module.load_state_dict(new_state, strict=False)

    return {"attrs": backup, "modules": module_backups}

def _restore_ema_export_backup(backup, gaussians):
    if not backup:
        return
    for key, saved_value in backup.get("attrs", {}).items():
        attr_name = key.split(":", 1)[1]
        value = getattr(gaussians, attr_name, None)
        if torch.is_tensor(value) and value.shape == saved_value.shape:
            value.data.copy_(saved_value.to(device=value.device, dtype=value.dtype))
    for module_name, saved_state in backup.get("modules", {}).items():
        module = getattr(gaussians, module_name, None)
        if module is not None:
            module.load_state_dict(saved_state, strict=False)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    reconstruction_criterion = build_reconstruction_criterion(
        name=getattr(opt, "reconstruction_loss", "charbonnier"),
        charbonnier_eps=getattr(opt, "charbonnier_eps", 1e-3),
    )
    gaussians = GaussianModel(
        dataset.sh_degree,
        use_bgfc=getattr(dataset, "use_bgfc", False),
        use_at_gom=getattr(dataset, "use_at_gom", False),
        bgfc_hidden_dim=getattr(dataset, "bgfc_hidden_dim", 32),
        bgfc_gate_init_bias=getattr(dataset, "bgfc_gate_init_bias", -2.2),
        bgfc_thermal_grayscale_context=getattr(dataset, "bgfc_thermal_grayscale_context", True),
        bgfc_rgb_luma_transfer_only=getattr(dataset, "bgfc_rgb_luma_transfer_only", True),
        use_render_calibration=getattr(dataset, "use_render_calibration", False),
        use_color_refinement=getattr(dataset, "use_color_refinement", False),
        color_refinement_hidden_dim=getattr(dataset, "color_refinement_hidden_dim", 16),
        color_refinement_max_residual=getattr(dataset, "color_refinement_max_residual", 0.06),
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    use_paired_views = getattr(dataset, "use_paired_views", False)
    pairing_summary = scene.getPairingSummary("train")
    if use_paired_views and pairing_summary.get("paired_camera_count", 0) == 0:
        print("[Warning] --use_paired_views enabled but no paired cameras were discovered. Falling back to the legacy train-camera pool.")
    print(
        "[Pairing:train] enabled={} rgb={} thermal={} paired={}".format(
            use_paired_views,
            pairing_summary.get("total_cameras_rgb", 0),
            pairing_summary.get("total_cameras_thermal", 0),
            pairing_summary.get("paired_camera_count", 0),
        )
    )
    adaptive_weight_state = _build_adaptive_weight_state()

    if checkpoint:
        checkpoint_state = torch.load(checkpoint)
        adaptive_weight_state_data = None
        if isinstance(checkpoint_state, tuple):
            if len(checkpoint_state) == 2:
                (model_params, first_iter) = checkpoint_state
            elif len(checkpoint_state) >= 3:
                (model_params, first_iter, adaptive_weight_state_data) = checkpoint_state[:3]
            else:
                raise ValueError("Unexpected checkpoint format.")
        elif isinstance(checkpoint_state, dict):
            model_params = checkpoint_state["model_params"]
            first_iter = checkpoint_state["iteration"]
            adaptive_weight_state_data = checkpoint_state.get("adaptive_weight_state")
        else:
            raise ValueError("Unexpected checkpoint format.")
        gaussians.restore(model_params, opt)
        adaptive_weight_state = _restore_adaptive_weight_state(adaptive_weight_state_data)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    paired_sampling_attempts = 0
    paired_sampling_hits = 0
    paired_sampling_fallbacks = 0

    ema_loss_for_log = 0.0
    ema_export_state = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    


    for iteration in range(first_iter, opt.iterations + 1):        
        color_refinement_active = (
            gaussians.use_color_refinement
            and iteration > int(getattr(opt, "color_refinement_start_iter", opt.densify_until_iter))
        )
        gaussians.set_color_refinement_runtime_enabled(color_refinement_active)

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_thermal = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render_thermal"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    net_thermal_bytes = memoryview((torch.clamp(net_thermal, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send([net_image_bytes,net_thermal_bytes], dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        late_prune_phase_active = (
            getattr(opt, "late_prune_only", False)
            and iteration >= max(opt.densify_until_iter, getattr(opt, "late_prune_only_from_iter", opt.densify_until_iter))
            and iteration <= getattr(opt, "late_prune_only_until_iter", opt.iterations)
        )

        # Increase the active SH degree every 1,000 iterations up to the configured maximum.
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainSamplingCameras(paired_only=use_paired_views).copy()
            
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        paired_sampling_attempts += 1
        if use_paired_views and getattr(viewpoint_cam, "has_paired_view", False):
            paired_sampling_hits += 1
        elif use_paired_views:
            paired_sampling_fallbacks += 1
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, thermal = render_pkg["render_color"], render_pkg["render_thermal"]
        rgb_viewspace_point_tensor = render_pkg["rgb_viewspace_points"]
        rgb_visibility_filter = render_pkg["rgb_visibility_filter"]
        rgb_radii = render_pkg["rgb_radii"]
        thermal_viewspace_point_tensor = render_pkg["thermal_viewspace_points"]
        thermal_visibility_filter = render_pkg["thermal_visibility_filter"]
        thermal_radii = render_pkg["thermal_radii"]



        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_thermal = viewpoint_cam.original_thermal.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        rgb_data_term = reconstruction_criterion(image, gt_image)
        ssim_rgb_loss = 1.0 - ssim(image, gt_image)
        Ll1_thermal = l1_loss(thermal, gt_thermal)
        thermal_data_term = reconstruction_criterion(thermal, gt_thermal)
        ssim_thermal_loss = 1.0 - ssim(thermal, gt_thermal)
        smoothloss_thermal = edge_aware_tv_loss(
            thermal,
            gt_thermal,
            beta=float(getattr(opt, "thermal_edge_aware_tv_beta", 8.0)),
        )

        # The LPIPS loss uses the standard input range [-1, 1].
        if iteration > 15000:
            lpips_rgb = lpips_model(image * 2.0 - 1.0, gt_image * 2.0 - 1.0).mean()
        else:
            lpips_rgb = Ll1.new_zeros(())

        if iteration > 15000:
            lpips_thermal = lpips_model(thermal * 2.0 - 1.0, gt_thermal * 2.0 - 1.0).mean()
        else:
            lpips_thermal = Ll1.new_zeros(())

        recon_rgb = (1.0 - opt.lambda_dssim) * rgb_data_term + opt.lambda_dssim * ssim_rgb_loss + 0.01 * lpips_rgb
        recon_thermal = (1.0 - opt.lambda_dssim) * thermal_data_term + opt.lambda_dssim * ssim_thermal_loss + 0.01 * lpips_thermal

        w_rgb, w_thermal, adaptive_weight_metrics = _compute_adaptive_branch_weights(
            iteration=iteration,
            recon_rgb=recon_rgb,
            recon_th=recon_thermal,
            opt=opt,
            state=adaptive_weight_state,
        )
        w_rgb, w_thermal, late_color_priority_alpha = _apply_late_color_priority(iteration, w_rgb, opt)
        adaptive_weight_metrics["weight_rgb_after_priority"] = w_rgb
        adaptive_weight_metrics["weight_th_after_priority"] = w_thermal
        adaptive_weight_metrics["late_color_priority_alpha"] = late_color_priority_alpha

        at_gom_regularization = Ll1.new_zeros(())
        bgfc_stability_reg = Ll1.new_zeros(())
        gate_sparsity_reg = Ll1.new_zeros(())
        gate_collapse_reg = Ll1.new_zeros(())
        gate_overlap_reg = Ll1.new_zeros(())
        render_calibration_reg = Ll1.new_zeros(())
        color_refinement_reg = Ll1.new_zeros(())
        bgfc_log_metrics = {}

        if gaussians.use_at_gom:
            at_gom_regularization = opt.at_gom_regularization_weight * gaussians.get_at_gom_regularization()
        if gaussians.use_render_calibration:
            render_calibration_reg_weight = _decayed_weight(
                iteration=iteration,
                start_weight=getattr(opt, "render_calibration_reg_weight", 1e-4),
                final_weight=getattr(opt, "render_calibration_reg_final_weight", getattr(opt, "render_calibration_reg_weight", 1e-4)),
                start_iter=getattr(opt, "render_calibration_reg_decay_start_iter", opt.densify_until_iter),
                end_iter=getattr(opt, "render_calibration_reg_decay_end_iter", opt.iterations),
            )
            render_calibration_reg = render_calibration_reg_weight * gaussians.get_render_calibration_reg()
        else:
            render_calibration_reg_weight = 0.0
        if gaussians.use_color_refinement:
            color_refinement_reg_weight = _decayed_weight(
                iteration=iteration,
                start_weight=getattr(opt, "color_refinement_reg_weight", 0.01),
                final_weight=getattr(opt, "color_refinement_reg_final_weight", getattr(opt, "color_refinement_reg_weight", 0.01)),
                start_iter=getattr(opt, "color_refinement_reg_decay_start_iter", opt.densify_until_iter),
                end_iter=getattr(opt, "color_refinement_reg_decay_end_iter", opt.iterations),
            )
            color_refinement_reg = color_refinement_reg_weight * gaussians.get_color_refinement_reg()
        else:
            color_refinement_reg_weight = 0.0

        if gaussians.use_bgfc:
            bgfc_stability_reg = render_pkg["bgfc_stability_reg"]
            gate_sparsity_reg = render_pkg["bgfc_gate_sparsity_reg"]
            gate_collapse_reg = render_pkg["bgfc_gate_collapse_reg"]
            gate_overlap_reg = render_pkg["bgfc_gate_overlap_reg"]
            bgfc_log_metrics = {
                "gate_th2rgb_mean": render_pkg["bgfc_gate_th2rgb_mean"].detach().item(),
                "gate_th2rgb_std": render_pkg["bgfc_gate_th2rgb_std"].detach().item(),
                "gate_th2rgb_max": render_pkg["bgfc_gate_th2rgb_max"].detach().item(),
                "gate_rgb2th_mean": render_pkg["bgfc_gate_rgb2th_mean"].detach().item(),
                "gate_rgb2th_std": render_pkg["bgfc_gate_rgb2th_std"].detach().item(),
                "gate_rgb2th_max": render_pkg["bgfc_gate_rgb2th_max"].detach().item(),
                "delta_th2rgb_mag_mean": render_pkg["bgfc_delta_th2rgb_mag_mean"].detach().item(),
                "delta_rgb2th_mag_mean": render_pkg["bgfc_delta_rgb2th_mag_mean"].detach().item(),
                "bgfc_stability_reg": bgfc_stability_reg.detach().item(),
                "gate_sparsity_reg": gate_sparsity_reg.detach().item(),
                "gate_collapse_reg": gate_collapse_reg.detach().item(),
                "gate_overlap_reg": gate_overlap_reg.detach().item(),
            }
            
        loss = (
            w_rgb * recon_rgb
            + w_thermal * recon_thermal
            + opt.thermal_smooth_weight * smoothloss_thermal
            + at_gom_regularization
            + opt.bgfc_stability_weight * bgfc_stability_reg
            + opt.bgfc_gate_sparsity_weight * gate_sparsity_reg
            + opt.bgfc_gate_collapse_weight * gate_collapse_reg
            + opt.bgfc_gate_overlap_weight * gate_overlap_reg
            + render_calibration_reg
            + color_refinement_reg
        )

        torch.cuda.synchronize()
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            local_residual_pool_kernel = getattr(opt, "cmo_local_residual_pool_kernel", 5)
            rgb_local_residual_map = (image.detach() - gt_image).abs().mean(dim=0, keepdim=True)
            thermal_local_residual_map = (thermal.detach() - gt_thermal).abs().mean(dim=0, keepdim=True)
            rgb_local_residual_proxy = gaussians.sample_anchor_local_residuals(
                camera=viewpoint_cam,
                residual_map=rgb_local_residual_map,
                visibility_mask=render_pkg["rgb_visibility_filter"],
                use_at_gom_geometry=False,
                pool_kernel=local_residual_pool_kernel,
            )
            thermal_local_residual_proxy = gaussians.sample_anchor_local_residuals(
                camera=viewpoint_cam,
                residual_map=thermal_local_residual_map,
                visibility_mask=render_pkg["thermal_visibility_filter"],
                use_at_gom_geometry=True,
                pool_kernel=local_residual_pool_kernel,
            )

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            pairing_runtime_metrics = _build_pairing_runtime_metrics(
                use_paired_views=use_paired_views,
                pairing_summary=pairing_summary,
                attempts=paired_sampling_attempts,
                hits=paired_sampling_hits,
                sampling_fallbacks=paired_sampling_fallbacks,
            )
            if iteration >= getattr(opt, "cmo_state_warmup_iters", 0):
                cmo_states_log_metrics = gaussians.update_cmo_states(
                    rgb_visibility_filter=render_pkg["rgb_visibility_filter"],
                    thermal_visibility_filter=render_pkg["thermal_visibility_filter"],
                    rgb_contribution_proxy=_build_anchor_contribution_proxy(render_pkg["rgb_radii"], gaussians.get_rgb_opacity),
                    thermal_contribution_proxy=_build_anchor_contribution_proxy(render_pkg["thermal_radii"], gaussians.get_thermal_opacity),
                    rgb_residual_proxy=rgb_local_residual_proxy,
                    thermal_residual_proxy=thermal_local_residual_proxy,
                    bgfc_usage_th2rgb=render_pkg["bgfc_gate_th2rgb_anchor"],
                    bgfc_usage_rgb2th=render_pkg["bgfc_gate_rgb2th_anchor"],
                    at_gom_usage=gaussians.get_at_gom_usage(),
                    ema=opt.cmo_state_ema,
                )
            else:
                cmo_states_log_metrics = gaussians.get_cmo_states_summary()

            cmo_scores = gaussians.get_cmo_scores(iteration=iteration)
            cmo_log_metrics = cmo_scores["diagnostics"]

            train_metric_scalars = {}
            train_metric_scalars.update(_prefix_metrics("pairing", pairing_runtime_metrics))
            train_metric_scalars.update(_prefix_metrics("cmo_states", cmo_states_log_metrics))
            train_metric_scalars.update(_prefix_metrics("cmo", cmo_log_metrics))
            train_metric_scalars.update(_prefix_metrics("adaptive_weight", adaptive_weight_metrics))
            train_metric_scalars.update(_prefix_metrics("render_calibration", gaussians.get_render_calibration_metrics()))
            train_metric_scalars.update(_prefix_metrics("color_refinement", gaussians.get_color_refinement_metrics()))
            train_metric_scalars["render_calibration/reg_weight"] = render_calibration_reg_weight
            train_metric_scalars["color_refinement/reg_weight"] = color_refinement_reg_weight
            if gaussians.last_cmo_gradient_diagnostics:
                train_metric_scalars.update(
                    _prefix_metrics("cmo/gradient", gaussians.last_cmo_gradient_diagnostics)
                )
            if bgfc_log_metrics:
                train_metric_scalars.update(_prefix_metrics("bgfc", bgfc_log_metrics))

            # Log and save
            if gaussians.use_bgfc and (iteration == first_iter or iteration % 100 == 0):
                print(
                    "\n[ITER {}] BGFC gate_th2rgb mean/std/max {:.6f} {:.6f} {:.6f} | "
                    "gate_rgb2th mean/std/max {:.6f} {:.6f} {:.6f}".format(
                        iteration,
                        bgfc_log_metrics["gate_th2rgb_mean"],
                        bgfc_log_metrics["gate_th2rgb_std"],
                        bgfc_log_metrics["gate_th2rgb_max"],
                        bgfc_log_metrics["gate_rgb2th_mean"],
                        bgfc_log_metrics["gate_rgb2th_std"],
                        bgfc_log_metrics["gate_rgb2th_max"],
                    )
                )
                print(
                    "[ITER {}] BGFC delta_th2rgb_mag_mean {:.6f} delta_rgb2th_mag_mean {:.6f} | "
                    "stability {:.6f} sparsity {:.6f} collapse {:.6f} overlap {:.6f}".format(
                        iteration,
                        bgfc_log_metrics["delta_th2rgb_mag_mean"],
                        bgfc_log_metrics["delta_rgb2th_mag_mean"],
                        bgfc_log_metrics["bgfc_stability_reg"],
                        bgfc_log_metrics["gate_sparsity_reg"],
                        bgfc_log_metrics["gate_collapse_reg"],
                        bgfc_log_metrics["gate_overlap_reg"],
                    )
                )
            if iteration == first_iter or iteration % 100 == 0:
                print(
                    "[ITER {}] Adaptive weight enabled={} rgb/th {:.6f} {:.6f} | "
                    "raw {:.6f}/{:.6f} norm {:.6f}/{:.6f} alpha {:.6f} priority {:.6f}".format(
                        iteration,
                        int(adaptive_weight_metrics["enabled"]),
                        adaptive_weight_metrics["weight_rgb_after_priority"],
                        adaptive_weight_metrics["weight_th_after_priority"],
                        adaptive_weight_metrics["raw_weight_rgb"],
                        adaptive_weight_metrics["raw_weight_th"],
                        adaptive_weight_metrics["norm_rgb"],
                        adaptive_weight_metrics["norm_th"],
                        adaptive_weight_metrics["blend_alpha"],
                        adaptive_weight_metrics["late_color_priority_alpha"],
                    )
                )
                print(
                    "[ITER {}] Pairing enabled={} paired={} hit_rate {:.6f} sample_fallbacks {} dataset_fallbacks {}".format(
                        iteration,
                        use_paired_views,
                        int(pairing_runtime_metrics["paired_camera_count"]),
                        pairing_runtime_metrics["paired_sampling_hit_rate"],
                        int(pairing_runtime_metrics["paired_sampling_fallback_count"]),
                        int(pairing_runtime_metrics["paired_camera_fallback_count"]),
                    )
                )
                print(
                    "[ITER {}] CMO states vis_rgb {:.6f}/{:.6f} vis_th {:.6f}/{:.6f} | "
                    "contrib_rgb {:.6f}/{:.6f} contrib_th {:.6f}/{:.6f}".format(
                        iteration,
                        cmo_states_log_metrics["visibility_rgb_mean"],
                        cmo_states_log_metrics["visibility_rgb_std"],
                        cmo_states_log_metrics["visibility_th_mean"],
                        cmo_states_log_metrics["visibility_th_std"],
                        cmo_states_log_metrics["contribution_rgb_mean"],
                        cmo_states_log_metrics["contribution_rgb_std"],
                        cmo_states_log_metrics["contribution_th_mean"],
                        cmo_states_log_metrics["contribution_th_std"],
                    )
                )
                print(
                    "[ITER {}] CMO states residual_rgb {:.6f}/{:.6f} residual_th {:.6f}/{:.6f} | "
                    "bgfc_usage_th2rgb {:.6f}/{:.6f} bgfc_usage_rgb2th {:.6f}/{:.6f} | "
                    "at_gom_usage {:.6f}/{:.6f}".format(
                        iteration,
                        cmo_states_log_metrics["residual_rgb_mean"],
                        cmo_states_log_metrics["residual_rgb_std"],
                        cmo_states_log_metrics["residual_th_mean"],
                        cmo_states_log_metrics["residual_th_std"],
                        cmo_states_log_metrics["bgfc_usage_th2rgb_mean"],
                        cmo_states_log_metrics["bgfc_usage_th2rgb_std"],
                        cmo_states_log_metrics["bgfc_usage_rgb2th_mean"],
                        cmo_states_log_metrics["bgfc_usage_rgb2th_std"],
                        cmo_states_log_metrics["at_gom_usage_mean"],
                        cmo_states_log_metrics["at_gom_usage_std"],
                    )
                )
                print(
                    "[ITER {}] CMO enabled={} points {} growth {:.6f} | "
                    "split {:.6f}/{:.6f}/{:.6f}".format(
                        iteration,
                        int(cmo_log_metrics["cmo_enabled"]),
                        int(cmo_log_metrics["current_point_count"]),
                        cmo_log_metrics["point_growth_ratio_vs_baseline_or_start"],
                        cmo_log_metrics["cmo_split_score_mean"],
                        cmo_log_metrics["cmo_split_score_std"],
                        cmo_log_metrics["cmo_split_score_max"],
                    )
                )
                print(
                    "[ITER {}] CMO extra_split {} | suppressed_by_budget {} | "
                    "prune_candidate_ratio {:.6f} veto_ratio {:.6f}".format(
                        iteration,
                        int(cmo_log_metrics["cmo_split_trigger_count"]),
                        int(cmo_log_metrics["split_suppressed_by_budget_count"]),
                        cmo_log_metrics["cmo_prune_candidate_ratio"],
                        cmo_log_metrics["cmo_prune_veto_ratio"],
                    )
                )
                print(
                    "[ITER {}] CMO prune_both {} | split_rgb {} split_th {} | "
                    "boost_bgfc {} boost_at_gom {}".format(
                        iteration,
                        int(cmo_log_metrics["prune_by_both_modalities_count"]),
                        int(cmo_log_metrics["split_triggered_by_rgb_count"]),
                        int(cmo_log_metrics["split_triggered_by_th_count"]),
                        int(cmo_log_metrics["split_boosted_by_bgfc_count"]),
                        int(cmo_log_metrics["split_boosted_by_at_gom_count"]),
                    )
                )

            training_report(
                tb_writer,
                iteration,
                Ll1,
                Ll1_thermal,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                train_metrics=train_metric_scalars,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                if (
                    getattr(opt, "use_ema_export", False)
                    and ema_export_state is not None
                    and iteration >= int(getattr(opt, "ema_export_start_iter", opt.iterations + 1))
                ):
                    print("[ITER {}] Exporting EMA-smoothed Gaussians".format(iteration))
                    ema_backup = _apply_ema_export_state(ema_export_state, gaussians)
                    scene.save(iteration)
                    _restore_ema_export_backup(ema_backup, gaussians)
                else:
                    scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[rgb_visibility_filter] = torch.max(
                    gaussians.max_radii2D[rgb_visibility_filter],
                    rgb_radii[rgb_visibility_filter],
                )
                gaussians.add_cmo_densification_stats(
                    rgb_viewspace_point_tensor=rgb_viewspace_point_tensor,
                    rgb_update_filter=rgb_visibility_filter,
                    thermal_viewspace_point_tensor=thermal_viewspace_point_tensor,
                    thermal_update_filter=thermal_visibility_filter,
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.01,
                        scene.cameras_extent,
                        size_threshold,
                        iteration=iteration,
                    )
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif late_prune_phase_active:
                gaussians.max_radii2D[rgb_visibility_filter] = torch.max(
                    gaussians.max_radii2D[rgb_visibility_filter],
                    rgb_radii[rgb_visibility_filter],
                )
                if getattr(opt, "late_prune_interval", 0) > 0 and iteration % opt.late_prune_interval == 0:
                    removed_count = gaussians.late_prune_only(
                        min_opacity=0.005,
                        extent=scene.cameras_extent,
                        max_screen_size=None,
                        iteration=iteration,
                    )
                    print("[ITER {}] Late prune-only removed {}".format(iteration, removed_count))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.constrain_thermal_scaling(scene.cameras_extent)
                gaussians.optimizer.zero_grad(set_to_none = True)
                if (
                    getattr(opt, "use_ema_export", False)
                    and iteration >= int(getattr(opt, "ema_export_start_iter", opt.iterations + 1))
                ):
                    ema_export_state = _update_ema_export_state(
                        ema_state=ema_export_state,
                        gaussians=gaussians,
                        decay=getattr(opt, "ema_export_decay", 0.995),
                    )
                pass

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration, dict(adaptive_weight_state)),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

def prepare_output_and_logger(args, optimization_args=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    if optimization_args is not None:
        with open(os.path.join(args.model_path, "optimization_args"), 'w') as opt_log_f:
            opt_log_f.write(str(Namespace(**vars(optimization_args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Ll1_thermal, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_metrics=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_thermal_loss', Ll1_thermal.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if train_metrics:
            for metric_name, metric_value in train_metrics.items():
                tb_writer.add_scalar(metric_name, metric_value, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                l1_thermal_test = 0.0
                psnr_thermal_test = 0.0
                ssim_thermal_test = 0.0
                lpips_thermal_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render_color"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    thermal = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render_thermal"], 0.0, 1.0)
                    gt_thermal = torch.clamp(viewpoint.original_thermal.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/thermal_render".format(viewpoint.image_name), thermal[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/thermal_ground_truth".format(viewpoint.image_name), gt_thermal[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

                    l1_thermal_test += l1_loss(thermal, gt_thermal).mean().double()
                    psnr_thermal_test += psnr(thermal, gt_thermal).mean().double()
                    ssim_thermal_test += ssim(thermal, gt_thermal).mean().double()
                    lpips_thermal_test += lpips(thermal, gt_thermal, net_type='vgg').mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])   
                
                psnr_thermal_test /= len(config['cameras'])
                l1_thermal_test /= len(config['cameras'])
                ssim_thermal_test /= len(config['cameras'])
                lpips_thermal_test /= len(config['cameras'])


                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                print("\n[ITER {}] Thermal Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(iteration, config['name'], l1_thermal_test, psnr_thermal_test, ssim_thermal_test, lpips_thermal_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_thermal', l1_thermal_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_thermal', psnr_thermal_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim_thermal', ssim_thermal_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips_thermal', lpips_thermal_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
