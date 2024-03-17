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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            # if shorthand:
            #     if t == bool:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
            #     else:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            # else:
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
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
        self.K = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.deform_type = 'node'
        self.skinning = False
        self.hyper_dim = 8
        self.node_num = 1024
        self.pred_opacity = False
        self.pred_color = False
        self.use_hash = False
        self.hash_time = False
        self.d_rot_as_rotmat = False # Debug!!!
        self.d_rot_as_res = True # Debug!!!
        self.local_frame = False
        self.progressive_brand_time = False
        self.gs_with_motion_mask = False
        self.init_isotropic_gs_with_all_colmap_pcl = False
        self.as_gs_force_with_motion_mask = False  # Only for scenes with both static and dynamic parts and without alpha mask
        self.max_d_scale = -1.
        self.is_scene_static = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if not g.model_path.endswith(g.deform_type):
            g.model_path = os.path.join(os.path.dirname(os.path.normpath(g.model_path)), os.path.basename(os.path.normpath(g.model_path)) + f'_{g.deform_type}')
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 80_000
        self.warm_up = 3_000
        self.dynamic_color_warm_up = 20_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 50_000
        self.densify_grad_threshold = 0.0002
        self.oneupSHdegree_step = 1000
        self.random_bg_color = False

        self.deform_lr_scale = 1.
        self.deform_downsamp_strategy = 'samp_hyper'
        self.deform_downsamp_with_dynamic_mask = False
        self.node_enable_densify_prune = False
        self.node_densification_interval = 5000
        self.node_densify_from_iter = 1000
        self.node_densify_until_iter = 25_000
        self.node_force_densify_prune_step = 10_000
        self.node_max_num_ratio_during_init = 16

        self.random_init_deform_gs = False
        self.node_warm_up = 2_000
        self.iterations_node_sampling = 7500
        self.iterations_node_rendering = 10000

        self.progressive_train = False
        self.progressive_train_node = False
        self.progressive_stage_ratio = .2  # The ratio of the number of images added per stage
        self.progressive_stage_steps = 3000  # The training steps of each stage

        self.lambda_optical_landmarks = [1e-1, 1e-1,   1e-3,        0]
        self.lambda_optical_steps =     [0,    15_000, 25_000, 25_001]

        self.lambda_motion_mask_landmarks = [5e-1,      1e-2,      0]
        self.lambda_motion_mask_steps =     [0,       10_000, 10_001]
        self.no_motion_mask_loss = False  # Camera pose may be inaccurate and should model the whole scene motion

        self.gt_alpha_mask_as_scene_mask = False
        self.gt_alpha_mask_as_dynamic_mask = False
        self.no_arap_loss = False  # For large scenes arap is too slow
        self.with_temporal_smooth_loss = False

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    if not args_cmdline.model_path.endswith(args_cmdline.deform_type):
        args_cmdline.model_path = os.path.join(os.path.dirname(os.path.normpath(args_cmdline.model_path)), os.path.basename(os.path.normpath(args_cmdline.model_path)) + f'_{args_cmdline.deform_type}')

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
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
