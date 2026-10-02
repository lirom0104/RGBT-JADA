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

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._thermal = "thermal"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.use_bgfc = True
        self.use_at_gom = True
        self.use_paired_views = True
        self.use_camera_calibration = True
        self.bgfc_hidden_dim = 32
        self.bgfc_gate_init_bias = -2.2
        self.bgfc_thermal_grayscale_context = True
        self.bgfc_rgb_luma_transfer_only = True
        self.use_render_calibration = True
        self.use_color_refinement = True
        self.color_refinement_hidden_dim = 16
        self.color_refinement_max_residual = 0.06
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.thermal_feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.reconstruction_loss = "charbonnier"
        self.charbonnier_eps = 1e-3
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.00025
        self.cmo_gradient_consensus_floor = 0.7
        self.cmo_gradient_single_view_discount = 0.8
        self.late_prune_only = True
        self.late_prune_only_from_iter = 15_000
        self.late_prune_only_until_iter = 18_000
        self.late_prune_interval = 250
        self.random_background = False
        self.thermal_smooth_weight = 0.1
        self.thermal_edge_aware_tv_beta = 10.0
        self.at_gom_lr_scale = 0.1
        self.at_gom_warmup_iters = 4_000
        self.at_gom_regularization_weight = 1e-4
        self.render_calibration_lr = 0.001
        self.render_calibration_reg_weight = 5e-5
        self.render_calibration_reg_final_weight = 1e-5
        self.render_calibration_reg_decay_start_iter = 15_000
        self.render_calibration_reg_decay_end_iter = 27_000
        self.color_refinement_start_iter = 12_000
        self.color_refinement_lr = 0.001
        self.color_refinement_reg_weight = 0.006
        self.color_refinement_reg_final_weight = 0.002
        self.color_refinement_reg_decay_start_iter = 15_000
        self.color_refinement_reg_decay_end_iter = 27_000
        self.bgfc_lr_scale = 0.1
        self.adaptive_branch_weight = True
        self.adaptive_branch_weight_ema = 0.98
        self.adaptive_branch_weight_warmup_iters = 2_000
        self.adaptive_branch_weight_ramp_iters = 2_000
        self.adaptive_branch_weight_beta = 4.0
        self.adaptive_branch_weight_min = 0.3
        self.late_color_priority = True
        self.late_color_priority_start_iter = 18_000
        self.late_color_priority_ramp_iters = 6_000
        self.late_color_priority_rgb_weight = 0.56
        self.use_ema_export = True
        self.ema_export_start_iter = 20_000
        self.ema_export_decay = 0.995
        self.bgfc_stability_weight = 5e-6
        self.bgfc_gate_sparsity_weight = 5e-5
        self.bgfc_gate_collapse_weight = 1e-4
        self.bgfc_gate_overlap_weight = 7.5e-5
        self.bgfc_gate_target_std = 0.1
        self.cmo_state_ema = 0.95
        self.save_cmo_states = False
        self.cmo_state_warmup_iters = 0
        self.use_cmo = True
        self.cmo_warmup_iters = 5_000
        self.cmo_local_residual_pool_kernel = 5
        self.cmo_split_rgb_weight = 1.0
        self.cmo_split_th_weight = 1.1
        self.cmo_split_bgfc_boost = 0.04
        self.cmo_split_at_gom_boost = 0.04
        self.cmo_split_score_threshold = 1.1
        self.cmo_split_partner_balance_min = 0.3
        self.cmo_split_max_extra_ratio = 0.15
        self.cmo_max_point_ratio = 5
        self.cmo_prune_visibility_thresh = 0.05
        self.cmo_prune_contribution_thresh = 0.25
        self.cmo_prune_residual_thresh = 0.25
        self.cmo_prune_bgfc_veto_thresh = 1.25
        self.cmo_prune_at_gom_veto_thresh = 1.25
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
