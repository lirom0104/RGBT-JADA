#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr



import torch
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, fov2focal
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.bgfc import BGFC

class DualModalRefinementHead(nn.Module):
    def __init__(self, hidden_dim=16, max_residual=0.06):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_residual = float(max_residual)
        input_channels = 10
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.color_residual_head = nn.Conv2d(self.hidden_dim, 3, kernel_size=3, padding=1)
        self.color_gate_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=3, padding=1)
        self.thermal_residual_head = nn.Conv2d(self.hidden_dim, 3, kernel_size=3, padding=1)
        self.thermal_gate_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=3, padding=1)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.encoder:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        for head in (
            self.color_residual_head,
            self.color_gate_head,
            self.thermal_residual_head,
            self.thermal_gate_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _luma(image):
        if image.shape[0] == 1:
            return image
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        return (image[:3] * weights).sum(dim=0, keepdim=True)

    @staticmethod
    def _edge_magnitude(luma):
        dx = F.pad(luma[:, :, 1:] - luma[:, :, :-1], (0, 1, 0, 0))
        dy = F.pad(luma[:, 1:, :] - luma[:, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-8)

    def _branch_input(self, target, guide):
        guide = guide.detach()
        target_luma = self._luma(target)
        guide_luma = self._luma(guide)
        target_edge = self._edge_magnitude(target_luma)
        guide_edge = self._edge_magnitude(guide_luma)
        return torch.cat(
            (
                target,
                guide,
                target_luma,
                guide_luma,
                target_edge,
                guide_edge,
            ),
            dim=0,
        ).unsqueeze(0)

    def _predict_residual(self, branch_input, residual_head, gate_head):
        features = self.encoder(branch_input)
        gate = torch.sigmoid(gate_head(features))
        residual = torch.tanh(residual_head(features)) * gate * self.max_residual
        return residual.squeeze(0), gate.squeeze(0)

    def forward(self, color, thermal):
        color_input = self._branch_input(color, thermal)
        thermal_input = self._branch_input(thermal, color)
        color_residual, color_gate = self._predict_residual(
            color_input,
            self.color_residual_head,
            self.color_gate_head,
        )
        thermal_residual, thermal_gate = self._predict_residual(
            thermal_input,
            self.thermal_residual_head,
            self.thermal_gate_head,
        )
        return {
            "color": color + color_residual,
            "thermal": thermal + thermal_residual,
            "color_residual": color_residual,
            "thermal_residual": thermal_residual,
            "color_gate": color_gate,
            "thermal_gate": thermal_gate,
        }

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(
        self,
        sh_degree : int,
        use_bgfc : bool = False,
        use_at_gom : bool = False,
        bgfc_hidden_dim : int = 32,
        bgfc_gate_init_bias : float = -2.2,
        bgfc_thermal_grayscale_context : bool = True,
        bgfc_rgb_luma_transfer_only : bool = True,
        use_render_calibration : bool = False,
        use_color_refinement : bool = False,
        color_refinement_hidden_dim : int = 16,
        color_refinement_max_residual : float = 0.06,
    ):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._thermal_dc = torch.empty(0)
        self._thermal_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity_base = torch.empty(0)
        self._at_gom_opacity_bias_rgb = torch.empty(0)
        self._at_gom_opacity_bias_th = torch.empty(0)
        self._at_gom_center_residual = torch.empty(0)
        self._at_gom_log_scale_residual = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.cmo_lifecycle_state = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.at_gom_center_scheduler_args = None
        self.at_gom_log_scale_lr = None
        self.at_gom_warmup_iters = 0
        self.cmo_states = {}
        self.cmo_state_ema = 0.95
        self.save_cmo_states_enabled = False
        self.cmo_enabled = False
        self.cmo_warmup_iters = 0
        self.cmo_split_rgb_weight = 1.0
        self.cmo_split_th_weight = 1.1
        self.cmo_split_bgfc_boost = 0.04
        self.cmo_split_at_gom_boost = 0.04
        self.cmo_prune_visibility_thresh = 0.05
        self.cmo_prune_contribution_thresh = 0.25
        self.cmo_prune_residual_thresh = 0.25
        self.cmo_prune_bgfc_veto_thresh = 1.25
        self.cmo_prune_at_gom_veto_thresh = 1.25
        self.cmo_split_score_threshold = 1.1
        self.cmo_split_partner_balance_min = 0.3
        self.cmo_split_max_extra_ratio = 0.15
        self.cmo_max_point_ratio = 1.8
        self.cmo_gradient_consensus_floor = 0.7
        self.cmo_gradient_single_view_discount = 0.8
        self.last_cmo_gradient_diagnostics = {}
        self.cmo_start_point_count = 0
        self.cmo_budget_reference_count = 0
        self.last_cmo_selection_diagnostics = {
            "cmo_split_trigger_count": 0.0,
            "split_suppressed_by_partner_balance_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
        }
        self.last_cmo_diagnostics = {}
        self.use_bgfc = use_bgfc
        self.use_at_gom = use_at_gom
        self.bgfc_hidden_dim = bgfc_hidden_dim
        self.bgfc_gate_init_bias = bgfc_gate_init_bias
        self.bgfc_thermal_grayscale_context = bgfc_thermal_grayscale_context
        self.bgfc_rgb_luma_transfer_only = bgfc_rgb_luma_transfer_only
        self.use_render_calibration = use_render_calibration
        self.use_color_refinement = use_color_refinement
        self.color_refinement_hidden_dim = color_refinement_hidden_dim
        self.color_refinement_max_residual = color_refinement_max_residual
        self.color_refinement_runtime_enabled = True
        self.color_refinement = None
        self.last_color_refinement_residual = None
        self.last_thermal_refinement_residual = None
        self.bgfc_gate_target_std = 0.1
        self.bgfc = None
        self._render_calibration_color_scale = self._make_render_calibration_parameter(1.0)
        self._render_calibration_color_bias = self._make_render_calibration_parameter(0.0)
        self._render_calibration_thermal_scale = self._make_render_calibration_parameter(1.0)
        self._render_calibration_thermal_bias = self._make_render_calibration_parameter(0.0)
        self.setup_functions()
        self._configure_bgfc_module()
        self._configure_color_refinement_module()

    def capture(self):
        return {
            "version": 5,
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "thermal_dc": self._thermal_dc,
            "thermal_rest": self._thermal_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity_base": self._opacity_base,
            "at_gom_opacity_bias_rgb": self._at_gom_opacity_bias_rgb,
            "at_gom_opacity_bias_th": self._at_gom_opacity_bias_th,
            "at_gom_center_residual": self._at_gom_center_residual,
            "at_gom_log_scale_residual": self._at_gom_log_scale_residual,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "spatial_lr_scale": self.spatial_lr_scale,
            "use_bgfc": self.use_bgfc,
            "use_at_gom": self.use_at_gom,
            "bgfc_hidden_dim": self.bgfc_hidden_dim,
            "bgfc_gate_init_bias": self.bgfc_gate_init_bias,
            "bgfc_thermal_grayscale_context": self.bgfc_thermal_grayscale_context,
            "bgfc_rgb_luma_transfer_only": self.bgfc_rgb_luma_transfer_only,
            "use_render_calibration": self.use_render_calibration,
            "render_calibration_color_scale": self._render_calibration_color_scale,
            "render_calibration_color_bias": self._render_calibration_color_bias,
            "render_calibration_thermal_scale": self._render_calibration_thermal_scale,
            "render_calibration_thermal_bias": self._render_calibration_thermal_bias,
            "use_color_refinement": self.use_color_refinement,
            "color_refinement_hidden_dim": self.color_refinement_hidden_dim,
            "color_refinement_max_residual": self.color_refinement_max_residual,
            "color_refinement_state": self.color_refinement.state_dict() if self.color_refinement is not None else None,
            "bgfc_gate_target_std": self.bgfc_gate_target_std,
            "bgfc_state": self.bgfc.state_dict() if self.bgfc is not None else None,
            "cmo_states": self.get_cmo_states(),
            "cmo_state_ema": self.cmo_state_ema,
            "save_cmo_states_enabled": self.save_cmo_states_enabled,
            "cmo_gradient_consensus_floor": self.cmo_gradient_consensus_floor,
            "cmo_gradient_single_view_discount": self.cmo_gradient_single_view_discount,
            "cmo_split_partner_balance_min": self.cmo_split_partner_balance_min,
        }
    
    def restore(self, model_args, training_args):
        legacy_checkpoint = not isinstance(model_args, dict)
        cmo_states = None
        if legacy_checkpoint:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._thermal_dc,
                self._thermal_rest,
                self._scaling,
                self._rotation,
                opacity_legacy,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            self._opacity_base = opacity_legacy
            self._at_gom_opacity_bias_rgb = self._make_zero_parameter_like(self._opacity_base)
            self._at_gom_opacity_bias_th = self._make_zero_parameter_like(self._opacity_base)
            self._at_gom_center_residual = self._make_zero_parameter_like(self._xyz)
            self._at_gom_log_scale_residual = self._make_zero_parameter_like(self._scaling)
            cmo_states = None
        else:
            self.active_sh_degree = model_args["active_sh_degree"]
            self._xyz = model_args["xyz"]
            self._features_dc = model_args["features_dc"]
            self._features_rest = model_args["features_rest"]
            self._thermal_dc = model_args["thermal_dc"]
            self._thermal_rest = model_args["thermal_rest"]
            self._scaling = model_args["scaling"]
            self._rotation = model_args["rotation"]
            self._opacity_base = model_args["opacity_base"] if "opacity_base" in model_args else model_args["opacity"]
            self._at_gom_opacity_bias_rgb = (
                model_args["at_gom_opacity_bias_rgb"] if "at_gom_opacity_bias_rgb" in model_args else self._make_zero_parameter_like(self._opacity_base)
            )
            self._at_gom_opacity_bias_th = (
                model_args["at_gom_opacity_bias_th"] if "at_gom_opacity_bias_th" in model_args else self._make_zero_parameter_like(self._opacity_base)
            )
            self._at_gom_center_residual = (
                model_args["at_gom_center_residual"] if "at_gom_center_residual" in model_args else self._make_zero_parameter_like(self._xyz)
            )
            self._at_gom_log_scale_residual = (
                model_args["at_gom_log_scale_residual"] if "at_gom_log_scale_residual" in model_args else self._make_zero_parameter_like(self._scaling)
            )
            self.max_radii2D = model_args["max_radii2D"]
            xyz_gradient_accum = model_args["xyz_gradient_accum"]
            denom = model_args["denom"]
            opt_dict = model_args.get("optimizer_state")
            self.spatial_lr_scale = model_args["spatial_lr_scale"]
            self.use_bgfc = model_args.get("use_bgfc", self.use_bgfc)
            self.use_at_gom = model_args.get(
                "use_at_gom", self.use_at_gom
            )
            self.bgfc_hidden_dim = model_args.get("bgfc_hidden_dim", self.bgfc_hidden_dim)
            self.bgfc_gate_init_bias = model_args.get("bgfc_gate_init_bias", self.bgfc_gate_init_bias)
            self.bgfc_thermal_grayscale_context = model_args.get(
                "bgfc_thermal_grayscale_context", self.bgfc_thermal_grayscale_context
            )
            self.bgfc_rgb_luma_transfer_only = model_args.get(
                "bgfc_rgb_luma_transfer_only", self.bgfc_rgb_luma_transfer_only
            )
            self.use_render_calibration = model_args.get("use_render_calibration", self.use_render_calibration)
            self._render_calibration_color_scale = self._load_render_calibration_parameter(
                model_args.get("render_calibration_color_scale"),
                default_value=1.0,
            )
            self._render_calibration_color_bias = self._load_render_calibration_parameter(
                model_args.get("render_calibration_color_bias"),
                default_value=0.0,
            )
            self._render_calibration_thermal_scale = self._load_render_calibration_parameter(
                model_args.get("render_calibration_thermal_scale"),
                default_value=1.0,
            )
            self._render_calibration_thermal_bias = self._load_render_calibration_parameter(
                model_args.get("render_calibration_thermal_bias"),
                default_value=0.0,
            )
            self.use_color_refinement = model_args.get("use_color_refinement", self.use_color_refinement)
            color_refinement_state = model_args.get("color_refinement_state")
            self.color_refinement_hidden_dim = model_args.get(
                "color_refinement_hidden_dim",
                self.color_refinement_hidden_dim,
            )
            self.color_refinement_max_residual = model_args.get(
                "color_refinement_max_residual",
                self.color_refinement_max_residual,
            )
            self.bgfc_gate_target_std = model_args.get("bgfc_gate_target_std", self.bgfc_gate_target_std)
            self.cmo_state_ema = model_args.get("cmo_state_ema", self.cmo_state_ema)
            self.save_cmo_states_enabled = model_args.get("save_cmo_states_enabled", self.save_cmo_states_enabled)
            self.cmo_gradient_consensus_floor = model_args.get(
                "cmo_gradient_consensus_floor", self.cmo_gradient_consensus_floor
            )
            self.cmo_gradient_single_view_discount = model_args.get(
                "cmo_gradient_single_view_discount", self.cmo_gradient_single_view_discount
            )
            self.cmo_split_partner_balance_min = model_args.get(
                "cmo_split_partner_balance_min", self.cmo_split_partner_balance_min
            )
            cmo_states = model_args.get("cmo_states")

        # Version 4 and older learned an RGB-specific opacity offset.  Fold that
        # offset into the canonical opacity logit and compensate the thermal
        # offset, preserving both rendered opacities while enforcing alpha_r=alpha.
        self._canonicalize_rgb_opacity()

        self._configure_bgfc_module()
        self._configure_color_refinement_module()
        self._refresh_optional_parameter_grad_flags()
        self.training_setup(training_args)
        bgfc_state = model_args.get("bgfc_state") if isinstance(model_args, dict) else None
        self._load_bgfc_state_compat(bgfc_state)
        color_refinement_state = model_args.get("color_refinement_state") if isinstance(model_args, dict) else None
        self._load_color_refinement_state_compat(color_refinement_state)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.cmo_lifecycle_state = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._restore_cmo_states(cmo_states, self.get_xyz.shape[0])
        if opt_dict is None:
            return
        if legacy_checkpoint:
            print("[Warning] Loaded a legacy checkpoint; optimizer state is reinitialized for the refactored GaussianModel.")
            return
        try:
            self.optimizer.load_state_dict(opt_dict)
        except ValueError:
            legacy_groups = opt_dict.get("param_groups", []) if isinstance(opt_dict, dict) else []
            filtered_groups = [
                group for group in legacy_groups
                if group.get("name") != "at_gom_opacity_bias_rgb"
            ]
            if len(filtered_groups) == len(self.optimizer.param_groups) and len(filtered_groups) != len(legacy_groups):
                retained_ids = {
                    parameter_id
                    for group in filtered_groups
                    for parameter_id in group.get("params", [])
                }
                migrated_state = {
                    "state": {
                        parameter_id: state
                        for parameter_id, state in opt_dict.get("state", {}).items()
                        if parameter_id in retained_ids
                    },
                    "param_groups": filtered_groups,
                }
                try:
                    self.optimizer.load_state_dict(migrated_state)
                    print("[Warning] Migrated optimizer state from the legacy RGB-opacity-bias layout.")
                    return
                except ValueError:
                    pass
            print("[Warning] Optimizer state is incompatible with the refactored GaussianModel and was reinitialized.")

    def _make_zero_parameter_like(self, reference):
        return nn.Parameter(torch.zeros_like(reference).requires_grad_(True))

    def _canonicalize_rgb_opacity(self):
        if self._opacity_base.numel() == 0:
            return

        rgb_bias = self._at_gom_opacity_bias_rgb.detach()
        base_requires_grad = bool(getattr(self._opacity_base, "requires_grad", True))
        thermal_requires_grad = bool(getattr(self._at_gom_opacity_bias_th, "requires_grad", True))
        self._opacity_base = nn.Parameter(
            (self._opacity_base.detach() + rgb_bias).clone(),
            requires_grad=base_requires_grad,
        )
        self._at_gom_opacity_bias_th = nn.Parameter(
            (self._at_gom_opacity_bias_th.detach() - rgb_bias).clone(),
            requires_grad=thermal_requires_grad,
        )
        # Kept as a zero-valued compatibility field for old checkpoints/PLYs.
        self._at_gom_opacity_bias_rgb = nn.Parameter(
            torch.zeros_like(self._opacity_base),
            requires_grad=False,
        )

    def _make_render_calibration_parameter(self, value):
        return nn.Parameter(
            torch.full((3, 1, 1), float(value), dtype=torch.float, device="cuda"),
            requires_grad=bool(self.use_render_calibration),
        )

    def _load_render_calibration_parameter(self, value, default_value):
        if value is None:
            return self._make_render_calibration_parameter(default_value)
        tensor = torch.as_tensor(value, dtype=torch.float, device="cuda")
        if tensor.shape != (3, 1, 1):
            tensor = tensor.reshape(3, 1, 1)
        return nn.Parameter(tensor.detach().clone(), requires_grad=bool(self.use_render_calibration))

    def _bgfc_feature_dim(self):
        return 3 * ((self.max_sh_degree + 1) ** 2)

    def _bgfc_base_anchor_context_dim(self):
        return 4

    def _bgfc_anchor_context_dim(self):
        return self._bgfc_base_anchor_context_dim()

    def _configure_bgfc_module(self):
        if not self.use_bgfc:
            self.bgfc = None
            return

        expected_feature_dim = self._bgfc_feature_dim()
        expected_base_anchor_context_dim = self._bgfc_base_anchor_context_dim()
        expected_anchor_context_dim = self._bgfc_anchor_context_dim()
        if (
            self.bgfc is None
            or self.bgfc.feature_dim != expected_feature_dim
            or self.bgfc.hidden_dim != self.bgfc_hidden_dim
            or getattr(self.bgfc, "base_anchor_context_dim", 0) != expected_base_anchor_context_dim
            or getattr(self.bgfc, "anchor_context_dim", 0) != expected_anchor_context_dim
            or self.bgfc.gate_init_bias != self.bgfc_gate_init_bias
        ):
            self.bgfc = BGFC(
                feature_dim=expected_feature_dim,
                hidden_dim=self.bgfc_hidden_dim,
                base_anchor_context_dim=expected_base_anchor_context_dim,
                gate_init_bias=self.bgfc_gate_init_bias,
            ).cuda()

    def _configure_color_refinement_module(self):
        if not self.use_color_refinement:
            self.color_refinement = None
            return

        if (
            self.color_refinement is None
            or not isinstance(self.color_refinement, DualModalRefinementHead)
            or self.color_refinement.hidden_dim != int(self.color_refinement_hidden_dim)
            or self.color_refinement.max_residual != float(self.color_refinement_max_residual)
        ):
            self.color_refinement = DualModalRefinementHead(
                hidden_dim=self.color_refinement_hidden_dim,
                max_residual=self.color_refinement_max_residual,
            ).cuda()

    def _load_bgfc_state_compat(self, bgfc_state):
        if bgfc_state is None or self.bgfc is None:
            return

        current_state = self.bgfc.state_dict()
        adapted_state = {}
        incompatible_keys = []

        for key, current_value in current_state.items():
            loaded_value = bgfc_state.get(key)
            if loaded_value is None:
                adapted_state[key] = current_value
                incompatible_keys.append(key)
                continue

            if loaded_value.shape == current_value.shape:
                adapted_state[key] = loaded_value
                continue

            if key == "backbone.0.weight" and loaded_value.shape[0] == current_value.shape[0]:
                adapted_weight = current_value.new_zeros(current_value.shape)
                overlap = min(loaded_value.shape[1], current_value.shape[1])
                adapted_weight[:, :overlap] = loaded_value[:, :overlap]
                adapted_state[key] = adapted_weight
                continue

            adapted_state[key] = current_value
            incompatible_keys.append(key)

        self.bgfc.load_state_dict(adapted_state, strict=False)
        if incompatible_keys:
            print(
                "[Warning] BGFC checkpoint partially reused; incompatible keys were reinitialized: {}".format(
                    ", ".join(sorted(incompatible_keys))
                )
            )

    def _load_color_refinement_state_compat(self, color_refinement_state):
        if color_refinement_state is None or self.color_refinement is None:
            return
        try:
            self.color_refinement.load_state_dict(color_refinement_state, strict=False)
        except RuntimeError:
            print("[Warning] Color refinement state is incompatible and was reinitialized.")

    def _legacy_shared_mode(self):
        return (not self.use_bgfc) and (not self.use_at_gom)

    def _cmo_state_names(self):
        return (
            "visibility_rgb",
            "visibility_th",
            "contribution_rgb",
            "contribution_th",
            "residual_rgb",
            "residual_th",
            "bgfc_usage_th2rgb",
            "bgfc_usage_rgb2th",
            "at_gom_usage",
        )

    def _cmo_state_device(self):
        if self._xyz.numel() > 0:
            return self._xyz.device
        return torch.device("cuda")

    def _empty_cmo_states(self, num_anchors=0, device=None):
        device = self._cmo_state_device() if device is None else device
        return {
            stat_name: torch.zeros((num_anchors, 1), device=device)
            for stat_name in self._cmo_state_names()
        }

    def _initialize_cmo_states(self, num_anchors, device=None):
        self.cmo_states = self._empty_cmo_states(num_anchors, device=device)

    def _cmo_states_size(self):
        if not self.cmo_states:
            return 0
        first_stat = next(iter(self.cmo_states.values()), None)
        if first_stat is None:
            return 0
        return first_stat.shape[0]

    def _resize_cmo_states(self, num_anchors, device=None):
        device = self._cmo_state_device() if device is None else device
        resized_stats = self._empty_cmo_states(num_anchors, device=device)
        current_size = self._cmo_states_size()
        if current_size == 0:
            self.cmo_states = resized_stats
            return

        overlap = min(current_size, num_anchors)
        for stat_name in self._cmo_state_names():
            stat_tensor = self.cmo_states.get(stat_name)
            if stat_tensor is None:
                continue
            stat_tensor = torch.as_tensor(stat_tensor, device=device)
            if stat_tensor.ndim == 1:
                stat_tensor = stat_tensor.unsqueeze(1)
            resized_stats[stat_name][:overlap] = stat_tensor[:overlap].detach().clone()
        self.cmo_states = resized_stats

    def _ensure_cmo_states(self, num_anchors=None, device=None):
        if num_anchors is None:
            num_anchors = self.get_xyz.shape[0]
        device = self._cmo_state_device() if device is None else device
        if not self.cmo_states:
            self._initialize_cmo_states(num_anchors, device=device)
            return

        expected_names = set(self._cmo_state_names())
        current_names = set(self.cmo_states.keys())
        shape_matches = all(
            self.cmo_states[stat_name].shape[0] == num_anchors
            for stat_name in current_names
        )
        if current_names != expected_names or not shape_matches:
            self._resize_cmo_states(num_anchors, device=device)
            current_names = set(self.cmo_states.keys())

        for stat_name in expected_names - current_names:
            self.cmo_states[stat_name] = torch.zeros((num_anchors, 1), device=device)
        for stat_name in expected_names:
            self.cmo_states[stat_name] = self.cmo_states[stat_name].to(device)

    def _restore_cmo_states(self, stats_dict, num_anchors):
        device = self._cmo_state_device()
        self._initialize_cmo_states(num_anchors, device=device)
        if not isinstance(stats_dict, dict):
            return

        for stat_name in self._cmo_state_names():
            stat_value = stats_dict.get(stat_name)
            if stat_value is None:
                continue
            stat_tensor = torch.as_tensor(stat_value, device=device)
            if stat_tensor.ndim == 1:
                stat_tensor = stat_tensor.unsqueeze(1)
            if stat_tensor.shape[0] != num_anchors:
                continue
            self.cmo_states[stat_name] = stat_tensor.detach().clone()

    def _prune_cmo_states(self, valid_points_mask):
        self._ensure_cmo_states(valid_points_mask.shape[0])
        for stat_name in self._cmo_state_names():
            self.cmo_states[stat_name] = self.cmo_states[stat_name][valid_points_mask].detach().clone()

    def _extend_cmo_states(self, extension_stats, num_new_anchors, previous_num_anchors=None):
        device = self._cmo_state_device()
        if previous_num_anchors is None:
            previous_num_anchors = max(self.get_xyz.shape[0] - num_new_anchors, 0)
        self._ensure_cmo_states(previous_num_anchors, device=device)
        if extension_stats is None:
            extension_stats = self._empty_cmo_states(num_new_anchors, device=device)

        for stat_name in self._cmo_state_names():
            extension_value = extension_stats.get(stat_name)
            if extension_value is None:
                extension_tensor = torch.zeros((num_new_anchors, 1), device=device)
            else:
                extension_tensor = torch.as_tensor(extension_value, device=device)
                if extension_tensor.ndim == 1:
                    extension_tensor = extension_tensor.unsqueeze(1)
            self.cmo_states[stat_name] = torch.cat(
                (self.cmo_states[stat_name], extension_tensor.detach().clone()),
                dim=0,
            )

    def _prepare_cmo_state_observation(self, values, num_anchors, visibility_mask=None, default_value=0.0):
        device = self._cmo_state_device()
        if values is None:
            observed = torch.full((num_anchors,), float(default_value), device=device)
        else:
            observed_values = torch.as_tensor(values, device=device).detach().reshape(-1)
            observed_values = torch.nan_to_num(
                observed_values,
                nan=float(default_value),
                posinf=float(default_value),
                neginf=float(default_value),
            )
            if observed_values.numel() == 1 and num_anchors != 1:
                observed_values = observed_values.expand(num_anchors)
            observed = torch.full(
                (num_anchors,),
                float(default_value),
                device=device,
                dtype=observed_values.dtype,
            )
            overlap = min(num_anchors, observed_values.shape[0])
            if overlap > 0:
                observed[:overlap] = observed_values[:overlap]

        if visibility_mask is not None:
            visibility_mask = visibility_mask.detach().reshape(-1).bool()
            if visibility_mask.numel() != num_anchors:
                raise ValueError("CMO state observation expects visibility masks with one value per anchor.")
            observed = observed.clone()
            observed[~visibility_mask] = float(default_value)

        return observed.reshape(-1, 1)

    def _ema_update_cmo_state(self, stat_name, observed_values, ema):
        self._ensure_cmo_states()
        target = self.cmo_states[stat_name]
        if observed_values.shape != target.shape:
            raise ValueError("CMO state EMA update expects full per-anchor observations.")
        target.mul_(ema).add_(observed_values * (1.0 - ema))

    def _refresh_optional_parameter_grad_flags(self):
        delta_requires_grad = self.use_at_gom
        if isinstance(self._at_gom_opacity_bias_rgb, nn.Parameter):
            # RGB always uses the canonical opacity alpha_i^r = alpha_i.
            self._at_gom_opacity_bias_rgb.requires_grad_(False)
        if isinstance(self._at_gom_opacity_bias_th, nn.Parameter):
            self._at_gom_opacity_bias_th.requires_grad_(self.use_at_gom)
        if isinstance(self._at_gom_center_residual, nn.Parameter):
            self._at_gom_center_residual.requires_grad_(delta_requires_grad)
        if isinstance(self._at_gom_log_scale_residual, nn.Parameter):
            self._at_gom_log_scale_residual.requires_grad_(delta_requires_grad)
        for parameter in (
            self._render_calibration_color_scale,
            self._render_calibration_color_bias,
            self._render_calibration_thermal_scale,
            self._render_calibration_thermal_bias,
        ):
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad_(bool(self.use_render_calibration))

    def _parameter_from_tensor(self, tensor, requires_grad):
        return nn.Parameter(tensor.detach().clone(), requires_grad=requires_grad)

    def _masked_parameter(self, parameter, mask):
        return self._parameter_from_tensor(parameter[mask], parameter.requires_grad)

    def _concatenated_parameter(self, parameter, extension_tensor):
        return self._parameter_from_tensor(torch.cat((parameter, extension_tensor), dim=0), parameter.requires_grad)

    def _bgfc_regularization_zero(self):
        if self._xyz.numel() > 0:
            return self._xyz.new_zeros(())
        return torch.tensor(0.0, device="cuda")

    def _bgfc_summary_stats(self, values):
        if values.numel() == 0:
            zero = self._bgfc_regularization_zero()
            return zero, zero, zero
        return values.mean(), values.std(unbiased=False), values.max()

    def _bgfc_anchorwise_mean(self, tensor):
        return tensor.flatten(start_dim=1).mean(dim=1)

    def _bgfc_anchorwise_norm(self, tensor):
        return tensor.flatten(start_dim=1).norm(dim=1)

    def _bgfc_luma_replicated_features(self, tensor):
        if tensor is None or tensor.numel() == 0 or tensor.shape[-1] != 3:
            return tensor
        luma_weights = tensor.new_tensor((0.299, 0.587, 0.114)).view(1, 1, 3)
        luma = (tensor * luma_weights).sum(dim=-1, keepdim=True)
        return luma.expand_as(tensor)

    def _bgfc_chunked_gated_l1(self, gate_tensor, delta_tensor, chunk_size=16384):
        if gate_tensor.numel() == 0:
            return self._bgfc_regularization_zero()

        total = self._bgfc_regularization_zero()
        num_anchors = gate_tensor.shape[0]
        for chunk_start in range(0, num_anchors, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_anchors)
            gate_chunk = gate_tensor[chunk_start:chunk_end]
            delta_chunk = delta_tensor[chunk_start:chunk_end]
            total = total + torch.sum(gate_chunk * delta_chunk.abs())
        return total / float(gate_tensor.numel())

    def _get_anchor_summary(self):
        return torch.cat((self.get_xyz, self.get_scaling, self.get_rotation), dim=1)

    def _build_render_params(self, means3D, scaling, features, opacity, scaling_modifier=1.0):
        return {
            "means3D": means3D,
            "xyz": means3D,
            "features": features,
            "shs": features,
            "opacity": opacity,
            "scales": scaling,
            "scaling": scaling,
            "rotations": self.get_rotation,
            "rotation": self.get_rotation,
            "cov3D_precomp": self.get_covariance(scaling_modifier=scaling_modifier, scaling=scaling, rotation=self._rotation),
        }

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_rgb_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features(self):
        return self.get_rgb_features
    
    @property
    def get_thermal_features(self):  
        thermal_dc = self._thermal_dc
        thermal_rest = self._thermal_rest
        return torch.cat((thermal_dc, thermal_rest), dim=1)
    
    @property
    def get_opacity_base(self):
        return self.opacity_activation(self._opacity_base)
    
    @property
    def get_rgb_opacity(self):
        return self.get_opacity_base

    @property
    def get_thermal_opacity(self):
        return self.opacity_activation(self._opacity_base + self._at_gom_opacity_bias_th)

    @property
    def get_opacity(self):
        return torch.maximum(self.get_rgb_opacity, self.get_thermal_opacity)

    @property
    def get_thermal_xyz(self):
        if not self.use_at_gom:
            return self.get_xyz
        return self._xyz + self._at_gom_center_residual

    @property
    def get_thermal_scaling(self):
        if not self.use_at_gom:
            return self.get_scaling
        return self.scaling_activation(self._scaling + self._at_gom_log_scale_residual)

    def constrain_thermal_scaling(self, scene_extent, max_extent_ratio=0.6):
        """Bound AT-GOM thermal scales without deleting any Gaussian anchors."""
        if not self.use_at_gom or self._at_gom_log_scale_residual.numel() == 0:
            return

        max_scale = max(float(scene_extent) * float(max_extent_ratio), 1e-6)
        max_log_scale = self._scaling.new_tensor(max_scale).log()
        with torch.no_grad():
            max_residual = max_log_scale - self._scaling
            self._at_gom_log_scale_residual.copy_(
                torch.minimum(self._at_gom_log_scale_residual, max_residual)
            )

    def get_covariance(self, scaling_modifier = 1, scaling = None, rotation = None):
        scaling = self.get_scaling if scaling is None else scaling
        rotation = self._rotation if rotation is None else rotation
        return self.covariance_activation(scaling, scaling_modifier, rotation)

    def get_bgfc_outputs(self):
        rgb_features = self.get_rgb_features
        thermal_features = self.get_thermal_features
        if not self.use_bgfc or self.bgfc is None:
            zero = self._bgfc_regularization_zero()
            return {
                "gate_th2rgb": None,
                "gate_rgb2th": None,
                "delta_th2rgb": None,
                "delta_rgb2th": None,
                "updated_rgb_features": rgb_features,
                "updated_thermal_features": thermal_features,
                "gate_th2rgb_anchor": rgb_features.new_zeros((rgb_features.shape[0],)),
                "gate_rgb2th_anchor": thermal_features.new_zeros((thermal_features.shape[0],)),
                "stability_reg": zero,
                "gate_sparsity_reg": zero,
                "gate_collapse_reg": zero,
                "gate_overlap_reg": zero,
                "gate_th2rgb_mean": zero,
                "gate_th2rgb_std": zero,
                "gate_th2rgb_max": zero,
                "gate_rgb2th_mean": zero,
                "gate_rgb2th_std": zero,
                "gate_rgb2th_max": zero,
                "delta_th2rgb_mag_mean": zero,
                "delta_rgb2th_mag_mean": zero,
            }

        thermal_context_features = thermal_features
        if self.bgfc_thermal_grayscale_context:
            thermal_context_features = self._bgfc_luma_replicated_features(thermal_features)
        anchor_state = self._get_anchor_summary()
        bgfc_anchor_context = self._get_base_bgfc_anchor_context()

        bgfc_outputs = self.bgfc(
            anchor_state=anchor_state,
            rgb_features=rgb_features,
            thermal_features=thermal_features,
            thermal_context_features=thermal_context_features,
            anchor_context=bgfc_anchor_context,
        )

        if self.bgfc_rgb_luma_transfer_only:
            bgfc_outputs["gate_th2rgb"] = self._bgfc_luma_replicated_features(bgfc_outputs["gate_th2rgb"])
            bgfc_outputs["delta_th2rgb"] = self._bgfc_luma_replicated_features(bgfc_outputs["delta_th2rgb"])
            bgfc_outputs["updated_rgb_features"] = (
                rgb_features + bgfc_outputs["gate_th2rgb"] * bgfc_outputs["delta_th2rgb"]
            )

        gate_th2rgb_anchor = self._bgfc_anchorwise_mean(bgfc_outputs["gate_th2rgb"])
        gate_rgb2th_anchor = self._bgfc_anchorwise_mean(bgfc_outputs["gate_rgb2th"])
        delta_th2rgb_mag = self._bgfc_anchorwise_norm(bgfc_outputs["delta_th2rgb"])
        delta_rgb2th_mag = self._bgfc_anchorwise_norm(bgfc_outputs["delta_rgb2th"])
        gate_th2rgb_mean, gate_th2rgb_std, gate_th2rgb_max = self._bgfc_summary_stats(gate_th2rgb_anchor)
        gate_rgb2th_mean, gate_rgb2th_std, gate_rgb2th_max = self._bgfc_summary_stats(gate_rgb2th_anchor)

        bgfc_outputs["stability_reg"] = (
            self._bgfc_chunked_gated_l1(bgfc_outputs["gate_th2rgb"], bgfc_outputs["delta_th2rgb"])
            + self._bgfc_chunked_gated_l1(bgfc_outputs["gate_rgb2th"], bgfc_outputs["delta_rgb2th"])
        )
        bgfc_outputs["gate_sparsity_reg"] = bgfc_outputs["gate_th2rgb"].mean() + bgfc_outputs["gate_rgb2th"].mean()
        bgfc_outputs["gate_collapse_reg"] = (
            torch.relu(gate_th2rgb_std.new_tensor(self.bgfc_gate_target_std) - gate_th2rgb_std)
            + torch.relu(gate_rgb2th_std.new_tensor(self.bgfc_gate_target_std) - gate_rgb2th_std)
        )
        bgfc_outputs["gate_overlap_reg"] = (gate_th2rgb_anchor * gate_rgb2th_anchor).mean()
        bgfc_outputs["gate_th2rgb_anchor"] = gate_th2rgb_anchor
        bgfc_outputs["gate_rgb2th_anchor"] = gate_rgb2th_anchor
        bgfc_outputs["gate_th2rgb_mean"] = gate_th2rgb_mean
        bgfc_outputs["gate_th2rgb_std"] = gate_th2rgb_std
        bgfc_outputs["gate_th2rgb_max"] = gate_th2rgb_max
        bgfc_outputs["gate_rgb2th_mean"] = gate_rgb2th_mean
        bgfc_outputs["gate_rgb2th_std"] = gate_rgb2th_std
        bgfc_outputs["gate_rgb2th_max"] = gate_rgb2th_max
        bgfc_outputs["delta_th2rgb_mag_mean"] = delta_th2rgb_mag.mean()
        bgfc_outputs["delta_rgb2th_mag_mean"] = delta_rgb2th_mag.mean()
        return bgfc_outputs

    def get_rgb_render_params(self, scaling_modifier=1.0, bgfc_outputs=None):
        if bgfc_outputs is None:
            bgfc_outputs = self.get_bgfc_outputs()
        return self._build_render_params(
            means3D=self.get_xyz,
            scaling=self.get_scaling,
            features=bgfc_outputs["updated_rgb_features"],
            opacity=self.get_rgb_opacity,
            scaling_modifier=scaling_modifier,
        )

    def get_thermal_render_params(self, scaling_modifier=1.0, bgfc_outputs=None):
        if bgfc_outputs is None:
            bgfc_outputs = self.get_bgfc_outputs()
        return self._build_render_params(
            means3D=self.get_thermal_xyz,
            scaling=self.get_thermal_scaling,
            features=bgfc_outputs["updated_thermal_features"],
            opacity=self.get_thermal_opacity,
            scaling_modifier=scaling_modifier,
        )

    def get_at_gom_regularization(self):
        if not self.use_at_gom:
            return self.get_xyz.new_zeros(())
        return self._at_gom_center_residual.abs().mean() + self._at_gom_log_scale_residual.abs().mean()

    def apply_render_calibration(self, image, modality):
        if not self.use_render_calibration:
            return image
        if modality == "thermal":
            return image * self._render_calibration_thermal_scale + self._render_calibration_thermal_bias
        return image * self._render_calibration_color_scale + self._render_calibration_color_bias

    def get_render_calibration_reg(self):
        if not self.use_render_calibration:
            return self.get_xyz.new_zeros(())
        return (
            (self._render_calibration_color_scale - 1.0).square().mean()
            + self._render_calibration_color_bias.square().mean()
            + (self._render_calibration_thermal_scale - 1.0).square().mean()
            + self._render_calibration_thermal_bias.square().mean()
        )

    def get_render_calibration_metrics(self):
        if not self.use_render_calibration:
            return {
                "enabled": 0.0,
                "color_scale_mean": 1.0,
                "color_bias_mean": 0.0,
                "thermal_scale_mean": 1.0,
                "thermal_bias_mean": 0.0,
            }
        return {
            "enabled": 1.0,
            "color_scale_mean": float(self._render_calibration_color_scale.detach().mean().item()),
            "color_bias_mean": float(self._render_calibration_color_bias.detach().mean().item()),
            "thermal_scale_mean": float(self._render_calibration_thermal_scale.detach().mean().item()),
            "thermal_bias_mean": float(self._render_calibration_thermal_bias.detach().mean().item()),
        }

    def set_color_refinement_runtime_enabled(self, enabled):
        self.color_refinement_runtime_enabled = bool(enabled)

    def apply_multimodal_refinement(self, color, thermal):
        if (
            not self.use_color_refinement
            or self.color_refinement is None
            or not self.color_refinement_runtime_enabled
        ):
            self.last_color_refinement_residual = None
            self.last_thermal_refinement_residual = None
            return color, thermal

        refined = self.color_refinement(color, thermal)
        self.last_color_refinement_residual = refined["color_residual"]
        self.last_thermal_refinement_residual = refined["thermal_residual"]
        return refined["color"], refined["thermal"]

    def get_color_refinement_reg(self):
        residuals = [
            residual
            for residual in (
                self.last_color_refinement_residual,
                self.last_thermal_refinement_residual,
            )
            if residual is not None
        ]
        if not residuals:
            return self.get_xyz.new_zeros(())
        return sum(residual.square().mean() for residual in residuals) / len(residuals)

    def get_color_refinement_metrics(self):
        if (
            not self.use_color_refinement
            or self.color_refinement is None
            or self.last_color_refinement_residual is None
        ):
            return {
                "enabled": float(bool(self.use_color_refinement)),
                "active": 0.0,
                "residual_abs_mean": 0.0,
                "residual_abs_max": 0.0,
                "thermal_residual_abs_mean": 0.0,
                "thermal_residual_abs_max": 0.0,
            }
        residual_abs = self.last_color_refinement_residual.detach().abs()
        thermal_residual_abs = (
            self.last_thermal_refinement_residual.detach().abs()
            if self.last_thermal_refinement_residual is not None
            else residual_abs.new_zeros(())
        )
        return {
            "enabled": 1.0,
            "active": float(bool(self.color_refinement_runtime_enabled)),
            "residual_abs_mean": float(residual_abs.mean().item()),
            "residual_abs_max": float(residual_abs.max().item()),
            "thermal_residual_abs_mean": float(thermal_residual_abs.mean().item()),
            "thermal_residual_abs_max": float(thermal_residual_abs.max().item()),
        }

    def get_at_gom_usage(self):
        if not self.use_at_gom or self.get_xyz.shape[0] == 0:
            return self.get_xyz.new_zeros((self.get_xyz.shape[0],))
        return self._at_gom_center_residual.norm(dim=1) + self._at_gom_log_scale_residual.norm(dim=1)

    def _get_base_bgfc_anchor_context(self):
        num_anchors = self.get_xyz.shape[0]
        device = self._cmo_state_device()
        context_dim = self._bgfc_base_anchor_context_dim()
        if num_anchors == 0:
            return torch.zeros((0, context_dim), device=device)

        self._ensure_cmo_states(num_anchors)
        visibility_rgb = torch.clamp(self._cmo_state_vector("visibility_rgb"), 0.0, 1.0)
        visibility_th = torch.clamp(self._cmo_state_vector("visibility_th"), 0.0, 1.0)
        contribution_rgb = self._normalize_cmo_state(self._cmo_state_vector("contribution_rgb"))
        contribution_th = self._normalize_cmo_state(self._cmo_state_vector("contribution_th"))
        return torch.stack(
            (visibility_rgb, visibility_th, contribution_rgb, contribution_th),
            dim=1,
        )

    def get_cmo_states(self):
        self._ensure_cmo_states()
        return {
            stat_name: stat_tensor.detach().clone()
            for stat_name, stat_tensor in self.cmo_states.items()
        }

    def get_cmo_states_summary(self):
        self._ensure_cmo_states()
        summary = {}
        for stat_name in self._cmo_state_names():
            stat_tensor = self.cmo_states[stat_name].detach().flatten()
            if stat_tensor.numel() == 0:
                summary[f"{stat_name}_mean"] = 0.0
                summary[f"{stat_name}_std"] = 0.0
            else:
                summary[f"{stat_name}_mean"] = stat_tensor.mean().item()
                summary[f"{stat_name}_std"] = stat_tensor.std(unbiased=False).item()
        return summary

    def update_cmo_states(
        self,
        rgb_visibility_filter,
        thermal_visibility_filter,
        rgb_contribution_proxy,
        thermal_contribution_proxy,
        rgb_residual_proxy,
        thermal_residual_proxy,
        bgfc_usage_th2rgb=None,
        bgfc_usage_rgb2th=None,
        at_gom_usage=None,
        ema=None,
    ):
        self._ensure_cmo_states()
        ema = self.cmo_state_ema if ema is None else ema
        ema = max(0.0, min(float(ema), 0.9999))
        num_anchors = self.get_xyz.shape[0]

        rgb_visibility_filter = rgb_visibility_filter.detach().reshape(-1).bool()
        thermal_visibility_filter = thermal_visibility_filter.detach().reshape(-1).bool()
        if rgb_visibility_filter.numel() != num_anchors or thermal_visibility_filter.numel() != num_anchors:
            raise ValueError("CMO states update expects per-anchor rgb/thermal visibility masks.")

        self._ema_update_cmo_state(
            "visibility_rgb",
            self._prepare_cmo_state_observation(rgb_visibility_filter.float(), num_anchors),
            ema,
        )
        self._ema_update_cmo_state(
            "visibility_th",
            self._prepare_cmo_state_observation(thermal_visibility_filter.float(), num_anchors),
            ema,
        )
        self._ema_update_cmo_state(
            "contribution_rgb",
            self._prepare_cmo_state_observation(
                rgb_contribution_proxy,
                num_anchors,
                visibility_mask=rgb_visibility_filter,
            ),
            ema,
        )
        self._ema_update_cmo_state(
            "contribution_th",
            self._prepare_cmo_state_observation(
                thermal_contribution_proxy,
                num_anchors,
                visibility_mask=thermal_visibility_filter,
            ),
            ema,
        )
        self._ema_update_cmo_state(
            "residual_rgb",
            self._prepare_cmo_state_observation(
                rgb_residual_proxy,
                num_anchors,
                visibility_mask=rgb_visibility_filter,
            ),
            ema,
        )
        self._ema_update_cmo_state(
            "residual_th",
            self._prepare_cmo_state_observation(
                thermal_residual_proxy,
                num_anchors,
                visibility_mask=thermal_visibility_filter,
            ),
            ema,
        )
        self._ema_update_cmo_state(
            "bgfc_usage_th2rgb",
            self._prepare_cmo_state_observation(
                bgfc_usage_th2rgb,
                num_anchors,
                visibility_mask=rgb_visibility_filter,
            ),
            ema,
        )
        self._ema_update_cmo_state(
            "bgfc_usage_rgb2th",
            self._prepare_cmo_state_observation(
                bgfc_usage_rgb2th,
                num_anchors,
                visibility_mask=thermal_visibility_filter,
            ),
            ema,
        )
        self._ema_update_cmo_state(
            "at_gom_usage",
            self._prepare_cmo_state_observation(
                at_gom_usage,
                num_anchors,
                visibility_mask=thermal_visibility_filter,
            ),
            ema,
        )
        return self.get_cmo_states_summary()

    def _cmo_ready(self, iteration=None):
        if not self.cmo_enabled:
            return False
        return iteration is None or iteration >= self.cmo_warmup_iters

    def _cmo_budget_state(self, iteration=None):
        current_point_count = int(self.get_xyz.shape[0])
        if current_point_count > 0 and self.cmo_start_point_count <= 0:
            self.cmo_start_point_count = current_point_count

        if self._cmo_ready(iteration) and self.cmo_budget_reference_count <= 0 and current_point_count > 0:
            self.cmo_budget_reference_count = current_point_count

        reference_point_count = self.cmo_budget_reference_count
        if reference_point_count <= 0:
            reference_point_count = self.cmo_start_point_count
        if reference_point_count <= 0:
            reference_point_count = max(1, current_point_count)

        reference_point_count = int(reference_point_count)
        growth_ratio = (
            current_point_count / float(reference_point_count)
            if current_point_count > 0
            else 0.0
        )
        max_points = max(reference_point_count, int(np.ceil(reference_point_count * self.cmo_max_point_ratio)))
        remaining_point_budget = max(0, max_points - current_point_count)
        return {
            "current_point_count": current_point_count,
            "reference_point_count": reference_point_count,
            "point_growth_ratio": growth_ratio,
            "max_points": max_points,
            "remaining_point_budget": remaining_point_budget,
            "budget_exceeded": current_point_count >= max_points,
        }

    def _cmo_zero_scores(self, iteration=None):
        num_anchors = self.get_xyz.shape[0]
        device = self._cmo_state_device()
        zero_scores = torch.zeros((num_anchors,), device=device)
        false_mask = torch.zeros((num_anchors,), dtype=torch.bool, device=device)
        true_mask = torch.ones((num_anchors,), dtype=torch.bool, device=device)
        budget_state = self._cmo_budget_state(iteration)
        diagnostics = {
            "cmo_enabled": 0.0,
            "cmo_split_score_mean": 0.0,
            "cmo_split_score_std": 0.0,
            "cmo_split_score_max": 0.0,
            "current_point_count": float(budget_state["current_point_count"]),
            "point_growth_ratio_vs_baseline_or_start": budget_state["point_growth_ratio"],
            "cmo_split_trigger_count": 0.0,
            "split_suppressed_by_partner_balance_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
            "cmo_prune_candidate_ratio": 0.0,
            "cmo_prune_veto_ratio": 0.0,
            "prune_by_both_modalities_count": 0.0,
            "split_triggered_by_rgb_count": 0.0,
            "split_triggered_by_th_count": 0.0,
            "split_boosted_by_bgfc_count": 0.0,
            "split_boosted_by_at_gom_count": 0.0,
            "cmo_split_balance_mean": 0.0,
        }
        self.last_cmo_diagnostics = diagnostics
        return {
            "cmo_split_score": zero_scores,
            "cmo_prune_mask": true_mask,
            "cmo_prune_candidate_mask": false_mask,
            "cmo_prune_veto_mask": false_mask,
            "split_rgb_score": zero_scores,
            "split_th_score": zero_scores,
            "split_partner_score": zero_scores,
            "split_balance_score": zero_scores,
            "bgfc_boost": zero_scores,
            "at_gom_boost": zero_scores,
            "diagnostics": diagnostics,
        }

    def _cmo_state_vector(self, stat_name):
        num_anchors = self.get_xyz.shape[0]
        device = self._cmo_state_device()
        stat_tensor = self.cmo_states.get(stat_name)
        if stat_tensor is None:
            return torch.zeros((num_anchors,), device=device)
        return torch.nan_to_num(
            torch.as_tensor(stat_tensor, device=device).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _normalize_cmo_state(self, values, clamp_max=2.0):
        values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if values.numel() == 0:
            return values
        positive_values = values[values > 0]
        if positive_values.numel() == 0:
            return torch.zeros_like(values)
        scale = positive_values.mean()
        if positive_values.numel() > 1:
            scale = scale + positive_values.std(unbiased=False)
        scale = scale.clamp_min(1e-6)
        return torch.clamp(values / scale, 0.0, clamp_max)

    def _pad_per_anchor_signal(self, values, fill_value=0.0):
        num_anchors = self.get_xyz.shape[0]
        device = self._cmo_state_device()
        if values is None:
            return torch.full((num_anchors,), fill_value, device=device)
        values = torch.nan_to_num(values.detach().reshape(-1), nan=fill_value, posinf=fill_value, neginf=fill_value)
        if values.shape[0] == num_anchors:
            return values.to(device)
        padded = torch.full((num_anchors,), fill_value, device=device, dtype=values.dtype)
        overlap = min(num_anchors, values.shape[0])
        if overlap > 0:
            padded[:overlap] = values[:overlap].to(device)
        return padded

    def get_cmo_scores(self, iteration=None):
        num_anchors = self.get_xyz.shape[0]
        if num_anchors == 0 or (not self._cmo_ready(iteration)):
            return self._cmo_zero_scores(iteration=iteration)

        self._ensure_cmo_states(num_anchors)
        budget_state = self._cmo_budget_state(iteration)

        visibility_rgb = torch.clamp(self._cmo_state_vector("visibility_rgb"), 0.0, 1.0)
        visibility_th = torch.clamp(self._cmo_state_vector("visibility_th"), 0.0, 1.0)
        contribution_rgb = self._normalize_cmo_state(self._cmo_state_vector("contribution_rgb"))
        contribution_th = self._normalize_cmo_state(self._cmo_state_vector("contribution_th"))
        residual_rgb = self._normalize_cmo_state(self._cmo_state_vector("residual_rgb"))
        residual_th = self._normalize_cmo_state(self._cmo_state_vector("residual_th"))
        bgfc_usage_th2rgb = self._normalize_cmo_state(self._cmo_state_vector("bgfc_usage_th2rgb"))
        bgfc_usage_rgb2th = self._normalize_cmo_state(self._cmo_state_vector("bgfc_usage_rgb2th"))
        at_gom_usage = self._normalize_cmo_state(self._cmo_state_vector("at_gom_usage"))

        split_rgb = self.cmo_split_rgb_weight * visibility_rgb * contribution_rgb * (1.0 + 0.5 * residual_rgb)
        split_th = self.cmo_split_th_weight * visibility_th * contribution_th * (1.0 + 0.5 * residual_th)
        split_partner = torch.minimum(split_rgb, split_th)
        base_split = torch.maximum(split_rgb, split_th)
        split_balance = torch.where(
            base_split > 0,
            split_partner / base_split.clamp_min(1e-6),
            torch.zeros_like(base_split),
        )

        bgfc_usage_proxy = torch.maximum(bgfc_usage_th2rgb, bgfc_usage_rgb2th)
        bgfc_boost = self.cmo_split_bgfc_boost * bgfc_usage_proxy
        at_gom_boost = self.cmo_split_at_gom_boost * at_gom_usage
        cmo_split_score = base_split + bgfc_boost + at_gom_boost

        low_rgb = (
            (visibility_rgb < self.cmo_prune_visibility_thresh)
            & (contribution_rgb < self.cmo_prune_contribution_thresh)
            & (residual_rgb < self.cmo_prune_residual_thresh)
        )
        low_th = (
            (visibility_th < self.cmo_prune_visibility_thresh)
            & (contribution_th < self.cmo_prune_contribution_thresh)
            & (residual_th < self.cmo_prune_residual_thresh)
        )
        cmo_prune_candidate_mask = low_rgb & low_th
        cmo_prune_veto_mask = (
            (bgfc_usage_proxy > self.cmo_prune_bgfc_veto_thresh)
            | (at_gom_usage > self.cmo_prune_at_gom_veto_thresh)
        )
        cmo_prune_mask = cmo_prune_candidate_mask & (~cmo_prune_veto_mask)

        split_threshold = self.cmo_split_score_threshold
        base_trigger = base_split >= split_threshold
        bgfc_trigger = (base_split + bgfc_boost) >= split_threshold
        final_trigger = cmo_split_score >= split_threshold

        split_triggered_by_rgb = (split_rgb >= split_threshold) & (split_rgb >= split_th)
        split_triggered_by_th = (split_th >= split_threshold) & (split_th > split_rgb)
        split_boosted_by_bgfc = (~base_trigger) & bgfc_trigger & (bgfc_boost > 0)
        split_boosted_by_at_gom = (~bgfc_trigger) & final_trigger & (at_gom_boost > 0)

        candidate_count = int(cmo_prune_candidate_mask.sum().item())
        veto_ratio = (
            cmo_prune_veto_mask[cmo_prune_candidate_mask].float().mean().item()
            if candidate_count > 0
            else 0.0
        )
        diagnostics = {
            "cmo_enabled": 1.0,
            "cmo_split_score_mean": cmo_split_score.mean().item(),
            "cmo_split_score_std": cmo_split_score.std(unbiased=False).item(),
            "cmo_split_score_max": cmo_split_score.max().item(),
            "current_point_count": float(budget_state["current_point_count"]),
            "point_growth_ratio_vs_baseline_or_start": budget_state["point_growth_ratio"],
            "cmo_split_trigger_count": self.last_cmo_selection_diagnostics["cmo_split_trigger_count"],
            "split_suppressed_by_partner_balance_count": self.last_cmo_selection_diagnostics[
                "split_suppressed_by_partner_balance_count"
            ],
            "split_suppressed_by_budget_count": self.last_cmo_selection_diagnostics[
                "split_suppressed_by_budget_count"
            ],
            "cmo_prune_candidate_ratio": cmo_prune_candidate_mask.float().mean().item(),
            "cmo_prune_veto_ratio": veto_ratio,
            "prune_by_both_modalities_count": float(candidate_count),
            "split_triggered_by_rgb_count": float(split_triggered_by_rgb.sum().item()),
            "split_triggered_by_th_count": float(split_triggered_by_th.sum().item()),
            "split_boosted_by_bgfc_count": float(split_boosted_by_bgfc.sum().item()),
            "split_boosted_by_at_gom_count": float(split_boosted_by_at_gom.sum().item()),
            "cmo_split_balance_mean": split_balance.mean().item(),
        }
        self.last_cmo_diagnostics = diagnostics
        return {
            "cmo_split_score": cmo_split_score,
            "cmo_prune_mask": cmo_prune_mask,
            "cmo_prune_candidate_mask": cmo_prune_candidate_mask,
            "cmo_prune_veto_mask": cmo_prune_veto_mask,
            "split_rgb_score": split_rgb,
            "split_th_score": split_th,
            "split_partner_score": split_partner,
            "split_balance_score": split_balance,
            "bgfc_boost": bgfc_boost,
            "at_gom_boost": at_gom_boost,
            "diagnostics": diagnostics,
        }

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def _thermal_to_rgb_channels(self, thermal_tensor):
        if thermal_tensor.ndim != 3 or thermal_tensor.shape[0] == 0:
            raise ValueError("Thermal tensor must have shape [C, H, W] with at least one channel.")

        if thermal_tensor.shape[0] == 1:
            return thermal_tensor.repeat(3, 1, 1)

        if thermal_tensor.shape[0] >= 3:
            return thermal_tensor[:3]

        return thermal_tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)

    def _build_thermal_features_from_cameras(self, xyz, init_cameras):
        coeff_count = (self.max_sh_degree + 1) ** 2
        thermal_features = torch.zeros((xyz.shape[0], 3, coeff_count), dtype=xyz.dtype, device=xyz.device)

        if not init_cameras:
            print("[ThermalInit] No training cameras available; using neutral thermal RGB 0.5.")
            thermal_rgb = torch.full((xyz.shape[0], 3), 0.5, dtype=xyz.dtype, device=xyz.device)
        else:
            thermal_sum = torch.zeros((xyz.shape[0], 3), dtype=xyz.dtype, device=xyz.device)
            thermal_count = torch.zeros((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            cameras_used = 0

            with torch.no_grad():
                for camera in init_cameras:
                    thermal_tensor = getattr(camera, "original_thermal", None)
                    if thermal_tensor is None or thermal_tensor.numel() == 0:
                        continue

                    thermal_image = self._thermal_to_rgb_channels(thermal_tensor.to(device=xyz.device, dtype=xyz.dtype))
                    height = int(thermal_image.shape[1])
                    width = int(thermal_image.shape[2])
                    if height <= 0 or width <= 0:
                        continue

                    R = torch.as_tensor(camera.R, dtype=xyz.dtype, device=xyz.device)
                    T = torch.as_tensor(camera.T, dtype=xyz.dtype, device=xyz.device)
                    camera_points = xyz @ R + T.unsqueeze(0)

                    depth = camera_points[:, 2]
                    positive_depth = depth > 1e-6
                    if not positive_depth.any():
                        continue

                    focal_x = float(fov2focal(float(camera.FoVx), width))
                    focal_y = float(fov2focal(float(camera.FoVy), height))
                    pixel_x = focal_x * (camera_points[:, 0] / depth) + (width * 0.5)
                    pixel_y = focal_y * (camera_points[:, 1] / depth) + (height * 0.5)

                    valid = (
                        positive_depth
                        & (pixel_x >= 0.0)
                        & (pixel_x <= max(width - 1, 0))
                        & (pixel_y >= 0.0)
                        & (pixel_y <= max(height - 1, 0))
                    )
                    if not valid.any():
                        continue

                    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
                    denom_x = max(width - 1, 1)
                    denom_y = max(height - 1, 1)
                    grid_x = (pixel_x[valid_idx] / denom_x) * 2.0 - 1.0
                    grid_y = (pixel_y[valid_idx] / denom_y) * 2.0 - 1.0
                    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)

                    sampled_rgb = F.grid_sample(
                        thermal_image.unsqueeze(0),
                        grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    ).squeeze(0).squeeze(-1).transpose(0, 1)

                    thermal_sum[valid_idx] += sampled_rgb
                    thermal_count[valid_idx] += 1.0
                    cameras_used += 1

            observed_mask = thermal_count.squeeze(1) > 0
            thermal_rgb = torch.full((xyz.shape[0], 3), 0.5, dtype=xyz.dtype, device=xyz.device)
            if observed_mask.any():
                thermal_rgb[observed_mask] = thermal_sum[observed_mask] / thermal_count[observed_mask]
                fallback_rgb = thermal_rgb[observed_mask].mean(dim=0, keepdim=True)
                thermal_rgb[~observed_mask] = fallback_rgb
                print(
                    "[ThermalInit] cameras={} covered_points={}/{} fallback_points={} mean_rgb=({:.4f}, {:.4f}, {:.4f})".format(
                        cameras_used,
                        int(observed_mask.sum().item()),
                        xyz.shape[0],
                        int((~observed_mask).sum().item()),
                        float(fallback_rgb[0, 0].item()),
                        float(fallback_rgb[0, 1].item()),
                        float(fallback_rgb[0, 2].item()),
                    )
                )
            else:
                print("[ThermalInit] No valid thermal projections found; using neutral thermal RGB 0.5.")

        thermal_features[:, :, 0] = RGB2SH(thermal_rgb.clamp(0.0, 1.0))
        return thermal_features

    def _project_points_to_image_grid(self, xyz, camera, height, width):
        if xyz.numel() == 0:
            empty = torch.zeros((0,), device=xyz.device, dtype=xyz.dtype)
            return empty, empty, torch.zeros((0,), device=xyz.device, dtype=torch.bool)

        R = torch.as_tensor(camera.R, dtype=xyz.dtype, device=xyz.device)
        T = torch.as_tensor(camera.T, dtype=xyz.dtype, device=xyz.device)
        camera_points = xyz @ R + T.unsqueeze(0)

        depth = camera_points[:, 2]
        positive_depth = depth > 1e-6
        if height <= 0 or width <= 0:
            return (
                torch.zeros_like(depth),
                torch.zeros_like(depth),
                torch.zeros_like(positive_depth),
            )

        focal_x = float(fov2focal(float(camera.FoVx), width))
        focal_y = float(fov2focal(float(camera.FoVy), height))
        pixel_x = focal_x * (camera_points[:, 0] / depth.clamp_min(1e-6)) + (width * 0.5)
        pixel_y = focal_y * (camera_points[:, 1] / depth.clamp_min(1e-6)) + (height * 0.5)

        valid = (
            positive_depth
            & (pixel_x >= 0.0)
            & (pixel_x <= max(width - 1, 0))
            & (pixel_y >= 0.0)
            & (pixel_y <= max(height - 1, 0))
        )
        denom_x = max(width - 1, 1)
        denom_y = max(height - 1, 1)
        grid_x = (pixel_x / denom_x) * 2.0 - 1.0
        grid_y = (pixel_y / denom_y) * 2.0 - 1.0
        return grid_x, grid_y, valid

    def sample_anchor_local_residuals(
        self,
        camera,
        residual_map,
        visibility_mask=None,
        use_at_gom_geometry=False,
        pool_kernel=5,
    ):
        xyz = self.get_thermal_xyz if use_at_gom_geometry else self.get_xyz
        device = xyz.device
        dtype = xyz.dtype
        num_anchors = xyz.shape[0]
        sampled_residuals = torch.zeros((num_anchors,), device=device, dtype=dtype)
        if num_anchors == 0 or residual_map is None:
            return sampled_residuals

        residual_tensor = torch.as_tensor(residual_map, device=device, dtype=dtype).detach()
        if residual_tensor.ndim == 2:
            residual_tensor = residual_tensor.unsqueeze(0)
        elif residual_tensor.ndim != 3:
            raise ValueError("Residual map must have shape [H, W] or [C, H, W].")
        if residual_tensor.shape[0] > 1:
            residual_tensor = residual_tensor.mean(dim=0, keepdim=True)

        residual_tensor = residual_tensor.unsqueeze(0)
        kernel_size = max(1, int(pool_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > 1:
            residual_tensor = F.avg_pool2d(
                residual_tensor,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )

        _, _, height, width = residual_tensor.shape
        with torch.no_grad():
            grid_x, grid_y, valid = self._project_points_to_image_grid(
                xyz.detach(),
                camera=camera,
                height=height,
                width=width,
            )
            if visibility_mask is not None:
                visibility_mask = visibility_mask.detach().reshape(-1).bool().to(device)
                if visibility_mask.numel() != num_anchors:
                    raise ValueError("Local residual sampling expects one visibility value per anchor.")
                valid = valid & visibility_mask

            if not valid.any():
                return sampled_residuals

            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            grid = torch.stack((grid_x[valid_idx], grid_y[valid_idx]), dim=-1).view(1, -1, 1, 2)
            sampled_values = F.grid_sample(
                residual_tensor,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).squeeze(0).squeeze(0).squeeze(-1)
            sampled_residuals[valid_idx] = sampled_values

        return sampled_residuals

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_cameras = None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        thermal_features = self._build_thermal_features_from_cameras(fused_point_cloud, init_cameras)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_dc = nn.Parameter(thermal_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_rest = nn.Parameter(thermal_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity_base = nn.Parameter(opacities.requires_grad_(True))
        self._at_gom_opacity_bias_rgb = nn.Parameter(torch.zeros_like(opacities), requires_grad=False)
        self._at_gom_opacity_bias_th = nn.Parameter(torch.zeros_like(opacities).requires_grad_(True))
        self._at_gom_center_residual = nn.Parameter(torch.zeros_like(fused_point_cloud).requires_grad_(True))
        self._at_gom_log_scale_residual = nn.Parameter(torch.zeros_like(scales).requires_grad_(True))
        self._refresh_optional_parameter_grad_flags()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.cmo_lifecycle_state = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._initialize_cmo_states(self.get_xyz.shape[0])

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.cmo_lifecycle_state = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.bgfc_gate_target_std = getattr(training_args, "bgfc_gate_target_std", self.bgfc_gate_target_std)
        self.cmo_state_ema = getattr(training_args, "cmo_state_ema", self.cmo_state_ema)
        self.save_cmo_states_enabled = getattr(training_args, "save_cmo_states", self.save_cmo_states_enabled)
        self.cmo_gradient_consensus_floor = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        training_args,
                        "cmo_gradient_consensus_floor",
                        self.cmo_gradient_consensus_floor,
                    )
                ),
            ),
        )
        self.cmo_gradient_single_view_discount = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        training_args,
                        "cmo_gradient_single_view_discount",
                        self.cmo_gradient_single_view_discount,
                    )
                ),
            ),
        )
        self.last_cmo_gradient_diagnostics = {}
        self.cmo_enabled = getattr(training_args, "use_cmo", self.cmo_enabled)
        self.cmo_warmup_iters = max(
            getattr(training_args, "cmo_state_warmup_iters", 0),
            getattr(training_args, "cmo_warmup_iters", self.cmo_warmup_iters),
        )
        self.cmo_split_rgb_weight = getattr(training_args, "cmo_split_rgb_weight", self.cmo_split_rgb_weight)
        self.cmo_split_th_weight = getattr(training_args, "cmo_split_th_weight", self.cmo_split_th_weight)
        self.cmo_split_bgfc_boost = getattr(training_args, "cmo_split_bgfc_boost", self.cmo_split_bgfc_boost)
        self.cmo_split_at_gom_boost = getattr(
            training_args, "cmo_split_at_gom_boost", self.cmo_split_at_gom_boost
        )
        self.cmo_split_score_threshold = getattr(
            training_args, "cmo_split_score_threshold", self.cmo_split_score_threshold
        )
        self.cmo_split_partner_balance_min = float(
            np.clip(
                getattr(
                    training_args,
                    "cmo_split_partner_balance_min",
                    self.cmo_split_partner_balance_min,
                ),
                0.0,
                1.0,
            )
        )
        self.cmo_split_max_extra_ratio = max(
            0.0,
            getattr(training_args, "cmo_split_max_extra_ratio", self.cmo_split_max_extra_ratio),
        )
        self.cmo_max_point_ratio = max(
            1.0,
            getattr(training_args, "cmo_max_point_ratio", self.cmo_max_point_ratio),
        )
        self.cmo_prune_visibility_thresh = getattr(
            training_args, "cmo_prune_visibility_thresh", self.cmo_prune_visibility_thresh
        )
        self.cmo_prune_contribution_thresh = getattr(
            training_args, "cmo_prune_contribution_thresh", self.cmo_prune_contribution_thresh
        )
        self.cmo_prune_residual_thresh = getattr(
            training_args, "cmo_prune_residual_thresh", self.cmo_prune_residual_thresh
        )
        self.cmo_prune_bgfc_veto_thresh = getattr(
            training_args, "cmo_prune_bgfc_veto_thresh", self.cmo_prune_bgfc_veto_thresh
        )
        self.cmo_prune_at_gom_veto_thresh = getattr(
            training_args, "cmo_prune_at_gom_veto_thresh", self.cmo_prune_at_gom_veto_thresh
        )
        self.cmo_start_point_count = int(self.get_xyz.shape[0]) if self.get_xyz.shape[0] > 0 else 0
        self.cmo_budget_reference_count = 0
        self.last_cmo_selection_diagnostics = {
            "cmo_split_trigger_count": 0.0,
            "split_suppressed_by_partner_balance_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
        }
        self._configure_bgfc_module()
        self._configure_color_refinement_module()
        self._refresh_optional_parameter_grad_flags()
        self.at_gom_center_scheduler_args = None
        self.at_gom_log_scale_lr = None
        self.at_gom_warmup_iters = max(
            0,
            int(getattr(training_args, "at_gom_warmup_iters", self.at_gom_warmup_iters)),
        )
        self._ensure_cmo_states(self.get_xyz.shape[0])

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz", "per_anchor": True},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc", "per_anchor": True},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest", "per_anchor": True},
            {'params': [self._thermal_dc], 'lr': training_args.thermal_feature_lr, "name": "thermal_dc", "per_anchor": True},
            {'params': [self._thermal_rest], 'lr': training_args.thermal_feature_lr / 20.0, "name": "t_rest", "per_anchor": True},
            {'params': [self._opacity_base], 'lr': training_args.opacity_lr, "name": "opacity_base", "per_anchor": True},
            {'params': [self._at_gom_opacity_bias_th], 'lr': training_args.opacity_lr, "name": "at_gom_opacity_bias_th", "per_anchor": True},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling", "per_anchor": True},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation", "per_anchor": True},
        ]
        if self.use_bgfc:
            bgfc_lr_scale = getattr(training_args, "bgfc_lr_scale", 0.1)
            l.append(
                {
                    'params': list(self.bgfc.parameters()),
                    'lr': training_args.feature_lr * bgfc_lr_scale,
                    "name": "bgfc",
                    "per_anchor": False,
                }
            )
        if self.use_color_refinement and self.color_refinement is not None:
            l.append(
                {
                    'params': list(self.color_refinement.parameters()),
                    'lr': getattr(training_args, "color_refinement_lr", 0.001),
                    "name": "color_refinement",
                    "per_anchor": False,
                }
            )
        if self.use_render_calibration:
            render_calibration_lr = getattr(training_args, "render_calibration_lr", 0.001)
            l.extend([
                {
                    'params': [self._render_calibration_color_scale],
                    'lr': render_calibration_lr,
                    "name": "render_calibration_color_scale",
                    "per_anchor": False,
                },
                {
                    'params': [self._render_calibration_color_bias],
                    'lr': render_calibration_lr,
                    "name": "render_calibration_color_bias",
                    "per_anchor": False,
                },
                {
                    'params': [self._render_calibration_thermal_scale],
                    'lr': render_calibration_lr,
                    "name": "render_calibration_thermal_scale",
                    "per_anchor": False,
                },
                {
                    'params': [self._render_calibration_thermal_bias],
                    'lr': render_calibration_lr,
                    "name": "render_calibration_thermal_bias",
                    "per_anchor": False,
                },
            ])
        if self.use_at_gom:
            at_gom_lr_scale = getattr(training_args, "at_gom_lr_scale", 0.1)
            delta_xyz_lr_init = training_args.position_lr_init * self.spatial_lr_scale * at_gom_lr_scale
            delta_xyz_lr_final = training_args.position_lr_final * self.spatial_lr_scale * at_gom_lr_scale
            self.at_gom_log_scale_lr = training_args.scaling_lr * at_gom_lr_scale
            l.extend([
                {'params': [self._at_gom_center_residual], 'lr': delta_xyz_lr_init, "name": "at_gom_center_residual", "per_anchor": True},
                {'params': [self._at_gom_log_scale_residual], 'lr': self.at_gom_log_scale_lr, "name": "at_gom_log_scale_residual", "per_anchor": True},
            ])
            self.at_gom_center_scheduler_args = get_expon_lr_func(
                lr_init=delta_xyz_lr_init,
                lr_final=delta_xyz_lr_final,
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=training_args.position_lr_max_steps,
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = self.xyz_scheduler_args(iteration)
        at_gom_warmup_active = iteration <= self.at_gom_warmup_iters
        at_gom_center_lr = None
        if self.at_gom_center_scheduler_args is not None:
            at_gom_center_lr = (
                0.0 if at_gom_warmup_active else self.at_gom_center_scheduler_args(iteration)
            )
        at_gom_log_scale_lr = None
        if self.at_gom_log_scale_lr is not None:
            at_gom_log_scale_lr = 0.0 if at_gom_warmup_active else self.at_gom_log_scale_lr
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = xyz_lr
            elif param_group["name"] == "at_gom_center_residual" and at_gom_center_lr is not None:
                param_group['lr'] = at_gom_center_lr
            elif param_group["name"] == "at_gom_log_scale_residual" and at_gom_log_scale_lr is not None:
                param_group['lr'] = at_gom_log_scale_lr
        return xyz_lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._thermal_dc.shape[1]*self._thermal_dc.shape[2]):
            l.append('t_dc_{}'.format(i))
        for i in range(self._thermal_rest.shape[1]*self._thermal_rest.shape[2]):
            l.append('t_rest_{}'.format(i))
        l.append('opacity')
        l.append('opacity_base')
        l.append('at_gom_opacity_bias_rgb')
        l.append('at_gom_opacity_bias_th')
        for i in range(self._at_gom_center_residual.shape[1]):
            l.append('at_gom_center_residual_{}'.format(i))
        for i in range(self._at_gom_log_scale_residual.shape[1]):
            l.append('at_gom_log_scale_residual_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        thermal_dc = self._thermal_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        t_rest = self._thermal_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        compatibility_opacity = self._opacity_base.detach().cpu().numpy()
        opacity_base = self._opacity_base.detach().cpu().numpy()
        at_gom_opacity_bias_rgb = self._at_gom_opacity_bias_rgb.detach().cpu().numpy()
        at_gom_opacity_bias_th = self._at_gom_opacity_bias_th.detach().cpu().numpy()
        at_gom_center_residual = self._at_gom_center_residual.detach().cpu().numpy()
        at_gom_log_scale_residual = self._at_gom_log_scale_residual.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                thermal_dc,
                t_rest,
                compatibility_opacity,
                opacity_base,
                at_gom_opacity_bias_rgb,
                at_gom_opacity_bias_th,
                at_gom_center_residual,
                at_gom_log_scale_residual,
                scale,
                rotation,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_opacity(self):
        base_opacity_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(base_opacity_new, "opacity_base")
        self._opacity_base = optimizable_tensors["opacity_base"]
        bias_rgb_new = torch.zeros_like(self._at_gom_opacity_bias_rgb)
        optimizable_tensors = self.replace_tensor_to_optimizer(bias_rgb_new, "at_gom_opacity_bias_rgb")
        if "at_gom_opacity_bias_rgb" in optimizable_tensors:
            self._at_gom_opacity_bias_rgb = optimizable_tensors["at_gom_opacity_bias_rgb"]
        bias_th_new = torch.zeros_like(self._at_gom_opacity_bias_th)
        optimizable_tensors = self.replace_tensor_to_optimizer(bias_th_new, "at_gom_opacity_bias_th")
        if "at_gom_opacity_bias_th" in optimizable_tensors:
            self._at_gom_opacity_bias_th = optimizable_tensors["at_gom_opacity_bias_th"]
        

    def load_ply(self, path):
        plydata = PlyData.read(path)
        property_names = {p.name for p in plydata.elements[0].properties}

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacity_base = (
            np.asarray(plydata.elements[0]["opacity_base"])[..., np.newaxis]
            if "opacity_base" in property_names
            else np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        )
        at_gom_opacity_bias_rgb = (
            np.asarray(plydata.elements[0]["at_gom_opacity_bias_rgb"])[..., np.newaxis]
            if "at_gom_opacity_bias_rgb" in property_names
            else np.zeros((xyz.shape[0], 1))
        )
        at_gom_opacity_bias_th = (
            np.asarray(plydata.elements[0]["at_gom_opacity_bias_th"])[..., np.newaxis]
            if "at_gom_opacity_bias_th" in property_names
            else np.zeros((xyz.shape[0], 1))
        )


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        thermal_dc = np.zeros((xyz.shape[0], 3, 1))
        thermal_dc[:, 0, 0] = np.asarray(plydata.elements[0]["t_dc_0"])
        thermal_dc[:, 1, 0] = np.asarray(plydata.elements[0]["t_dc_1"])
        thermal_dc[:, 2, 0] = np.asarray(plydata.elements[0]["t_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        extra_t_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("t_rest_")]
        extra_t_names = sorted(extra_t_names, key = lambda x: int(x.split('_')[-1]))

        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))

        assert len(extra_t_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        thermal_extra = np.zeros((xyz.shape[0], len(extra_t_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(extra_t_names):
            thermal_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        thermal_extra = thermal_extra.reshape((thermal_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        at_gom_center_residual = np.zeros((xyz.shape[0], 3))
        for idx in range(at_gom_center_residual.shape[1]):
            attr_name = "at_gom_center_residual_{}".format(idx)
            if attr_name in property_names:
                at_gom_center_residual[:, idx] = np.asarray(plydata.elements[0][attr_name])

        at_gom_log_scale_residual = np.zeros((xyz.shape[0], 3))
        for idx in range(at_gom_log_scale_residual.shape[1]):
            attr_name = "at_gom_log_scale_residual_{}".format(idx)
            if attr_name in property_names:
                at_gom_log_scale_residual[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_dc = nn.Parameter(torch.tensor(thermal_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_rest = nn.Parameter(torch.tensor(thermal_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity_base = nn.Parameter(torch.tensor(opacity_base, dtype=torch.float, device="cuda").requires_grad_(True))
        self._at_gom_opacity_bias_rgb = nn.Parameter(torch.tensor(at_gom_opacity_bias_rgb, dtype=torch.float, device="cuda").requires_grad_(True))
        self._at_gom_opacity_bias_th = nn.Parameter(torch.tensor(at_gom_opacity_bias_th, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._at_gom_center_residual = nn.Parameter(torch.tensor(at_gom_center_residual, dtype=torch.float, device="cuda").requires_grad_(True))
        self._at_gom_log_scale_residual = nn.Parameter(torch.tensor(at_gom_log_scale_residual, dtype=torch.float, device="cuda").requires_grad_(True))
        self._canonicalize_rgb_opacity()
        self._refresh_optional_parameter_grad_flags()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.cmo_lifecycle_state = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._initialize_cmo_states(self.get_xyz.shape[0])

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                requires_grad = group["params"][0].requires_grad
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.detach().clone(), requires_grad=requires_grad)
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.detach().clone(), requires_grad=requires_grad)

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group.get("per_anchor", True):
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            requires_grad = group["params"][0].requires_grad
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].detach().clone(), requires_grad=requires_grad)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].detach().clone(), requires_grad=requires_grad)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._thermal_dc = optimizable_tensors["thermal_dc"]
        self._thermal_rest = optimizable_tensors["t_rest"]
        self._opacity_base = optimizable_tensors["opacity_base"]
        self._at_gom_opacity_bias_rgb = optimizable_tensors.get("at_gom_opacity_bias_rgb", self._masked_parameter(self._at_gom_opacity_bias_rgb, valid_points_mask))
        self._at_gom_opacity_bias_th = optimizable_tensors.get("at_gom_opacity_bias_th", self._masked_parameter(self._at_gom_opacity_bias_th, valid_points_mask))
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._at_gom_center_residual = optimizable_tensors.get("at_gom_center_residual", self._masked_parameter(self._at_gom_center_residual, valid_points_mask))
        self._at_gom_log_scale_residual = optimizable_tensors.get("at_gom_log_scale_residual", self._masked_parameter(self._at_gom_log_scale_residual, valid_points_mask))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.cmo_lifecycle_state = self.cmo_lifecycle_state[valid_points_mask]
        self._prune_cmo_states(valid_points_mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group.get("per_anchor", True):
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            requires_grad = group["params"][0].requires_grad
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).detach().clone(),
                    requires_grad=requires_grad,
                )
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).detach().clone(),
                    requires_grad=requires_grad,
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_thermal_dc,
        new_thermal_rest,
        new_opacity_base,
        new_at_gom_opacity_bias_rgb,
        new_at_gom_opacity_bias_th,
        new_scaling,
        new_rotation,
        new_at_gom_center_residual,
        new_at_gom_log_scale_residual,
        new_cmo_states=None,
    ):
        previous_num_anchors = self.get_xyz.shape[0]
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "thermal_dc": new_thermal_dc,
        "t_rest": new_thermal_rest,
        "opacity_base": new_opacity_base,
        "at_gom_opacity_bias_rgb": new_at_gom_opacity_bias_rgb,
        "at_gom_opacity_bias_th": new_at_gom_opacity_bias_th,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "at_gom_center_residual": new_at_gom_center_residual,
        "at_gom_log_scale_residual": new_at_gom_log_scale_residual}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._thermal_dc = optimizable_tensors["thermal_dc"]
        self._thermal_rest = optimizable_tensors["t_rest"]
        self._opacity_base = optimizable_tensors["opacity_base"]
        self._at_gom_opacity_bias_rgb = optimizable_tensors.get(
            "at_gom_opacity_bias_rgb", self._concatenated_parameter(self._at_gom_opacity_bias_rgb, new_at_gom_opacity_bias_rgb)
        )
        self._at_gom_opacity_bias_th = optimizable_tensors.get(
            "at_gom_opacity_bias_th", self._concatenated_parameter(self._at_gom_opacity_bias_th, new_at_gom_opacity_bias_th)
        )
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._at_gom_center_residual = optimizable_tensors.get(
            "at_gom_center_residual", self._concatenated_parameter(self._at_gom_center_residual, new_at_gom_center_residual)
        )
        self._at_gom_log_scale_residual = optimizable_tensors.get(
            "at_gom_log_scale_residual", self._concatenated_parameter(self._at_gom_log_scale_residual, new_at_gom_log_scale_residual)
        )

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.cmo_lifecycle_state = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._extend_cmo_states(
            new_cmo_states,
            new_xyz.shape[0],
            previous_num_anchors=previous_num_anchors,
        )

    def _baseline_densify_selection_mask(self, grads, grad_threshold, scene_extent, prefer_large_scales):
        n_init_points = self.get_xyz.shape[0]
        grad_values = grads.detach().reshape(-1)
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grad_values.shape[0]] = grad_values
        selected_pts_mask = padded_grad >= grad_threshold

        scale_mask = torch.max(self.get_scaling, dim=1).values
        if prefer_large_scales:
            scale_mask = scale_mask > self.percent_dense * scene_extent
        else:
            scale_mask = scale_mask <= self.percent_dense * scene_extent
        return selected_pts_mask & scale_mask

    def _select_topk_mask(self, candidate_mask, scores, k):
        if k <= 0:
            return torch.zeros_like(candidate_mask)

        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            return torch.zeros_like(candidate_mask)
        if candidate_indices.numel() <= k:
            return candidate_mask

        candidate_scores = scores[candidate_indices]
        topk_indices = torch.topk(candidate_scores, k=k, largest=True, sorted=False).indices
        selected_mask = torch.zeros_like(candidate_mask)
        selected_mask[candidate_indices[topk_indices]] = True
        return selected_mask

    def _cmo_densify_selection_mask(
        self,
        grads,
        grad_threshold,
        scene_extent,
        prefer_large_scales,
        cmo_split_score=None,
        cmo_split_balance=None,
        iteration=None,
    ):
        baseline_mask = self._baseline_densify_selection_mask(
            grads=grads,
            grad_threshold=grad_threshold,
            scene_extent=scene_extent,
            prefer_large_scales=prefer_large_scales,
        )
        selection_diagnostics = {
            "cmo_split_trigger_count": 0.0,
            "split_suppressed_by_partner_balance_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
        }

        if (
            (not prefer_large_scales)
            or cmo_split_score is None
            or (not self._cmo_ready(iteration))
            or self.cmo_split_max_extra_ratio <= 0.0
        ):
            return baseline_mask, selection_diagnostics

        baseline_count = int(baseline_mask.sum().item())
        if baseline_count <= 0:
            return baseline_mask, selection_diagnostics

        padded_cmo_score = self._pad_per_anchor_signal(cmo_split_score, fill_value=0.0)
        if cmo_split_balance is None:
            padded_cmo_balance = torch.ones_like(padded_cmo_score)
        else:
            padded_cmo_balance = self._pad_per_anchor_signal(cmo_split_balance, fill_value=0.0)
        scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        raw_cmo_extra_candidates = (
            (~baseline_mask)
            & scale_mask
            & (padded_cmo_score >= self.cmo_split_score_threshold)
        )
        if self.cmo_split_partner_balance_min > 0.0:
            partner_balance_mask = padded_cmo_balance >= self.cmo_split_partner_balance_min
            selection_diagnostics["split_suppressed_by_partner_balance_count"] = float(
                (raw_cmo_extra_candidates & (~partner_balance_mask)).sum().item()
            )
        else:
            partner_balance_mask = torch.ones_like(raw_cmo_extra_candidates)
        cmo_extra_candidates = raw_cmo_extra_candidates & partner_balance_mask
        cmo_extra_candidate_count = int(cmo_extra_candidates.sum().item())
        if cmo_extra_candidate_count <= 0:
            return baseline_mask, selection_diagnostics

        max_extra_candidates = max(1, int(np.ceil(baseline_count * self.cmo_split_max_extra_ratio)))
        capped_extra_candidates = min(cmo_extra_candidate_count, max_extra_candidates)

        budget_state = self._cmo_budget_state(iteration)
        allowed_extra_candidates = min(capped_extra_candidates, budget_state["remaining_point_budget"])
        selection_diagnostics["split_suppressed_by_budget_count"] = float(
            max(0, capped_extra_candidates - allowed_extra_candidates)
        )
        if allowed_extra_candidates <= 0:
            return baseline_mask, selection_diagnostics

        selected_cmo_extras = self._select_topk_mask(
            cmo_extra_candidates,
            padded_cmo_score,
            allowed_extra_candidates,
        )
        selection_diagnostics["cmo_split_trigger_count"] = float(selected_cmo_extras.sum().item())
        return baseline_mask | selected_cmo_extras, selection_diagnostics

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        N=2,
        cmo_split_score=None,
        cmo_split_balance=None,
        iteration=None,
    ):
        selected_pts_mask, selection_diagnostics = self._cmo_densify_selection_mask(
            grads=grads,
            grad_threshold=grad_threshold,
            scene_extent=scene_extent,
            prefer_large_scales=True,
            cmo_split_score=cmo_split_score,
            cmo_split_balance=cmo_split_balance,
            iteration=iteration,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_thermal_dc = self._thermal_dc[selected_pts_mask].repeat(N,1,1)
        new_thermal_rest = self._thermal_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity_base = self._opacity_base[selected_pts_mask].repeat(N,1)
        new_at_gom_opacity_bias_rgb = self._at_gom_opacity_bias_rgb[selected_pts_mask].repeat(N,1)
        new_at_gom_opacity_bias_th = self._at_gom_opacity_bias_th[selected_pts_mask].repeat(N,1)
        new_at_gom_center_residual = self._at_gom_center_residual[selected_pts_mask].repeat(N,1)
        new_at_gom_log_scale_residual = self._at_gom_log_scale_residual[selected_pts_mask].repeat(N,1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_thermal_dc,
            new_thermal_rest,
            new_opacity_base,
            new_at_gom_opacity_bias_rgb,
            new_at_gom_opacity_bias_th,
            new_scaling,
            new_rotation,
            new_at_gom_center_residual,
            new_at_gom_log_scale_residual,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        return selection_diagnostics

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = self._baseline_densify_selection_mask(
            grads=grads,
            grad_threshold=grad_threshold,
            scene_extent=scene_extent,
            prefer_large_scales=False,
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_thermal_dc = self._thermal_dc[selected_pts_mask]
        new_thermal_rest = self._thermal_rest[selected_pts_mask]
        new_opacity_base = self._opacity_base[selected_pts_mask]
        new_at_gom_opacity_bias_rgb = self._at_gom_opacity_bias_rgb[selected_pts_mask]
        new_at_gom_opacity_bias_th = self._at_gom_opacity_bias_th[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_at_gom_center_residual = self._at_gom_center_residual[selected_pts_mask]
        new_at_gom_log_scale_residual = self._at_gom_log_scale_residual[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_thermal_dc,
            new_thermal_rest,
            new_opacity_base,
            new_at_gom_opacity_bias_rgb,
            new_at_gom_opacity_bias_th,
            new_scaling,
            new_rotation,
            new_at_gom_center_residual,
            new_at_gom_log_scale_residual,
        )

    def _build_prune_mask(self, min_opacity, extent, max_screen_size, iteration=None):
        cmo_post_scores = self.get_cmo_scores(iteration=iteration)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = prune_mask & cmo_post_scores["cmo_prune_mask"]
        # Keep max_screen_size in the public signature for call compatibility,
        # but size alone must never bypass low opacity and the full CMO gate.
        return prune_mask

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iteration=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        cmo_pre_scores = self.get_cmo_scores(iteration=iteration)
        self.last_cmo_selection_diagnostics = {
            "cmo_split_trigger_count": 0.0,
            "split_suppressed_by_partner_balance_count": 0.0,
            "split_suppressed_by_budget_count": 0.0,
        }
        self.densify_and_clone(grads, max_grad, extent)
        self.last_cmo_selection_diagnostics = self.densify_and_split(
            grads,
            max_grad,
            extent,
            cmo_split_score=cmo_pre_scores["cmo_split_score"],
            cmo_split_balance=cmo_pre_scores["split_balance_score"],
            iteration=iteration,
        )

        prune_mask = self._build_prune_mask(
            min_opacity=min_opacity,
            extent=extent,
            max_screen_size=max_screen_size,
            iteration=iteration,
        )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def late_prune_only(self, min_opacity, extent, max_screen_size, iteration=None):
        prune_mask = self._build_prune_mask(
            min_opacity=min_opacity,
            extent=extent,
            max_screen_size=max_screen_size,
            iteration=iteration,
        )
        removed_count = int(prune_mask.sum().item())
        self.prune_points(prune_mask)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        torch.cuda.empty_cache()
        return removed_count

    def add_densification_stats(self, viewspace_point_tensor, update_filter, weight=1.0):
        if viewspace_point_tensor is None or viewspace_point_tensor.grad is None:
            return

        update_filter = update_filter.detach().reshape(-1).bool()
        if update_filter.numel() == 0 or (not torch.any(update_filter)):
            return

        weight = float(weight)
        if weight <= 0.0:
            return

        self.xyz_gradient_accum[update_filter] += (
            torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True) * weight
        )
        self.denom[update_filter] += weight

    def add_cmo_densification_stats(
        self,
        rgb_viewspace_point_tensor,
        rgb_update_filter,
        thermal_viewspace_point_tensor=None,
        thermal_update_filter=None,
        rgb_weight=1.0,
        thermal_weight=1.0,
    ):
        def _extract_grad_tensor(viewspace_point_tensor, update_filter, weight):
            if viewspace_point_tensor is None or viewspace_point_tensor.grad is None:
                return None
            grad_tensor = torch.zeros((num_anchors, 2), device=device)
            grad_tensor[update_filter] = viewspace_point_tensor.grad[update_filter, :2] * float(weight)
            return grad_tensor

        num_anchors = self.get_xyz.shape[0]
        device = self.get_xyz.device

        rgb_update_filter = rgb_update_filter.detach().reshape(-1).bool()
        if rgb_update_filter.numel() != num_anchors:
            raise ValueError("RGB densification stats expect one visibility value per anchor.")

        if thermal_update_filter is None:
            thermal_update_filter = torch.zeros_like(rgb_update_filter)
        else:
            thermal_update_filter = thermal_update_filter.detach().reshape(-1).bool()
            if thermal_update_filter.numel() != num_anchors:
                raise ValueError("Thermal densification stats expect one visibility value per anchor.")

        union_filter = rgb_update_filter | thermal_update_filter
        if union_filter.numel() == 0 or (not torch.any(union_filter)):
            return

        rgb_grad = _extract_grad_tensor(rgb_viewspace_point_tensor, rgb_update_filter, rgb_weight)
        thermal_grad = _extract_grad_tensor(thermal_viewspace_point_tensor, thermal_update_filter, thermal_weight)

        rgb_grad_norm = torch.zeros((num_anchors, 1), device=device)
        if rgb_grad is not None:
            rgb_grad_norm = torch.norm(rgb_grad, dim=-1, keepdim=True)

        thermal_grad_norm = torch.zeros((num_anchors, 1), device=device)
        if thermal_grad is not None:
            thermal_grad_norm = torch.norm(thermal_grad, dim=-1, keepdim=True)

        active_modality_count = rgb_update_filter.float().unsqueeze(1) + thermal_update_filter.float().unsqueeze(1)
        active_modality_count = active_modality_count.clamp_min(1.0)

        combined_grad_norm = torch.sqrt((rgb_grad_norm.square() + thermal_grad_norm.square()) / active_modality_count)

        both_visible = rgb_update_filter & thermal_update_filter
        consensus_weight = torch.full(
            (num_anchors, 1),
            self.cmo_gradient_single_view_discount,
            device=device,
        )
        consensus_balance = torch.zeros((num_anchors, 1), device=device)
        if torch.any(both_visible):
            dominant_grad_norm = torch.maximum(rgb_grad_norm, thermal_grad_norm).clamp_min(1e-6)
            consensus_balance[both_visible] = (
                torch.minimum(rgb_grad_norm[both_visible], thermal_grad_norm[both_visible])
                / dominant_grad_norm[both_visible]
            )
            consensus_weight[both_visible] = (
                self.cmo_gradient_consensus_floor
                + (1.0 - self.cmo_gradient_consensus_floor) * consensus_balance[both_visible]
            )

        combined_grad_norm = combined_grad_norm * consensus_weight

        self.xyz_gradient_accum[union_filter] += combined_grad_norm[union_filter]
        # Match the legacy shared-means behavior: each training view contributes one
        # densification observation, even if both branches render the anchor.
        self.denom[union_filter] += 1
        self.last_cmo_gradient_diagnostics = {
            "union_visible_count": float(union_filter.sum().item()),
            "both_visible_count": float((rgb_update_filter & thermal_update_filter).sum().item()),
            "single_visible_count": float((union_filter & (~both_visible)).sum().item()),
            "consensus_floor": float(self.cmo_gradient_consensus_floor),
            "single_view_discount": float(self.cmo_gradient_single_view_discount),
            "consensus_balance_mean": float(
                consensus_balance[both_visible].mean().item() if torch.any(both_visible) else 0.0
            ),
        }

    def save_feature_modules(self, path):
        if (
            (not self.use_bgfc or self.bgfc is None)
            and (not self.use_color_refinement or self.color_refinement is None)
            and not self.use_render_calibration
        ):
            return

        bgfc_state = self.bgfc.state_dict() if self.bgfc is not None else None
        color_refinement_state = (
            self.color_refinement.state_dict() if self.color_refinement is not None else None
        )
        torch.save(
            {
                "use_bgfc": self.use_bgfc,
                "bgfc_hidden_dim": self.bgfc_hidden_dim,
                "bgfc_gate_init_bias": self.bgfc_gate_init_bias,
                "bgfc_thermal_grayscale_context": self.bgfc_thermal_grayscale_context,
                "bgfc_rgb_luma_transfer_only": self.bgfc_rgb_luma_transfer_only,
                "use_render_calibration": self.use_render_calibration,
                "render_calibration_color_scale": self._render_calibration_color_scale.detach().cpu(),
                "render_calibration_color_bias": self._render_calibration_color_bias.detach().cpu(),
                "render_calibration_thermal_scale": self._render_calibration_thermal_scale.detach().cpu(),
                "render_calibration_thermal_bias": self._render_calibration_thermal_bias.detach().cpu(),
                "use_color_refinement": self.use_color_refinement,
                "color_refinement_hidden_dim": self.color_refinement_hidden_dim,
                "color_refinement_max_residual": self.color_refinement_max_residual,
                "color_refinement_state": color_refinement_state,
                "bgfc_state": bgfc_state,
            },
            os.path.join(path, "feature_modules.pth"),
        )

    def load_feature_modules(self, path):
        feature_module_path = os.path.join(path, "feature_modules.pth")
        if not os.path.exists(feature_module_path):
            return

        module_state = torch.load(feature_module_path, map_location="cuda")
        self.bgfc_hidden_dim = module_state.get("bgfc_hidden_dim", self.bgfc_hidden_dim)
        self.bgfc_gate_init_bias = module_state.get("bgfc_gate_init_bias", self.bgfc_gate_init_bias)
        self.bgfc_thermal_grayscale_context = module_state.get(
            "bgfc_thermal_grayscale_context", self.bgfc_thermal_grayscale_context
        )
        self.bgfc_rgb_luma_transfer_only = module_state.get(
            "bgfc_rgb_luma_transfer_only", self.bgfc_rgb_luma_transfer_only
        )
        self.use_render_calibration = module_state.get("use_render_calibration", self.use_render_calibration)
        self._render_calibration_color_scale = self._load_render_calibration_parameter(
            module_state.get("render_calibration_color_scale"),
            default_value=1.0,
        )
        self._render_calibration_color_bias = self._load_render_calibration_parameter(
            module_state.get("render_calibration_color_bias"),
            default_value=0.0,
        )
        self._render_calibration_thermal_scale = self._load_render_calibration_parameter(
            module_state.get("render_calibration_thermal_scale"),
            default_value=1.0,
        )
        self._render_calibration_thermal_bias = self._load_render_calibration_parameter(
            module_state.get("render_calibration_thermal_bias"),
            default_value=0.0,
        )
        self._refresh_optional_parameter_grad_flags()
        self.use_color_refinement = module_state.get("use_color_refinement", self.use_color_refinement)
        color_refinement_state = module_state.get("color_refinement_state")
        self.color_refinement_hidden_dim = module_state.get(
            "color_refinement_hidden_dim",
            self.color_refinement_hidden_dim,
        )
        self.color_refinement_max_residual = module_state.get(
            "color_refinement_max_residual",
            self.color_refinement_max_residual,
        )
        self._configure_bgfc_module()
        bgfc_state = module_state.get("bgfc_state")
        self._load_bgfc_state_compat(bgfc_state)
        self._configure_color_refinement_module()
        self._load_color_refinement_state_compat(color_refinement_state)

    def save_cmo_states(self, path):
        if not self.save_cmo_states_enabled:
            return

        mkdir_p(path)
        torch.save(
            {
                "cmo_state_ema": self.cmo_state_ema,
                "cmo_states": {
                    stat_name: stat_tensor.detach().cpu()
                    for stat_name, stat_tensor in self.get_cmo_states().items()
                },
                "summary": self.get_cmo_states_summary(),
            },
            os.path.join(path, "cmo_states.pth"),
        )

    def load_cmo_states(self, path):
        cmo_states_path = os.path.join(path, "cmo_states.pth")
        if not os.path.exists(cmo_states_path):
            self._ensure_cmo_states(self.get_xyz.shape[0])
            return

        stats_state = torch.load(cmo_states_path, map_location="cuda")
        self.cmo_state_ema = stats_state.get("cmo_state_ema", self.cmo_state_ema)
        self._restore_cmo_states(
            stats_state.get("cmo_states"),
            self.get_xyz.shape[0],
        )
