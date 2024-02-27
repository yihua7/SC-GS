from gaussian_renderer import render


def render_cur_cam(self, cur_cam):
    fid = cur_cam.fid
    if self.deform.name == 'node':
        if 'Node' in self.visualization_mode:
            gaussians = self.deform.deform.as_gaussians  # if self.iteration_node_rendering < self.opt.iterations_node_rendering else self.deform.deform.as_gaussians_visualization
            time_input = fid.unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1)
            d_values = self.deform.deform.query_network(x=gaussians.get_xyz.detach(), t=time_input)
            if self.motion_animation_d_values is not None:
                for key in self.motion_animation_d_values:
                    d_values[key] = self.motion_animation_d_values[key]
            d_xyz, d_opacity, d_color = d_values['d_xyz'] * gaussians.motion_mask, d_values['d_opacity'] * gaussians.motion_mask if d_values['d_opacity'] is not None else None, d_values['d_color'] * gaussians.motion_mask if d_values['d_color'] is not None else None
            d_rotation, d_scaling = 0., 0.
            if self.animation_trans_bias is not None:
                d_xyz = d_xyz + self.animation_trans_bias
            gs_rot_bias = None
            vis_scale_const = self.vis_scale_const
        else:
            time_input = self.deform.deform.expand_time(fid)
            d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, is_training=False, node_trans_bias=self.animation_trans_bias, node_rot_bias=self.animation_rot_bias, motion_mask=self.gaussians.motion_mask, camera_center=cur_cam.camera_center, animation_d_values=self.motion_animation_d_values)
            gaussians = self.gaussians
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
            gs_rot_bias = d_values['gs_rot_bias']  # GS rotation bias
            vis_scale_const = None
    else:
        vis_scale_const = None
        if self.iteration < self.opt.warm_up:
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
            gaussians = self.gaussians
        else:
            N = self.gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            gaussians = self.gaussians
            d_values = self.deform.step(self.gaussians.get_xyz.detach(), time_input, feature=self.gaussians.feature, camera_center=cur_cam.camera_center)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        gs_rot_bias = None
    
    render_motion = "Motion" in self.visualization_mode
    if render_motion:
        vis_scale_const = self.vis_scale_const
    if type(d_rotation) is not float and gaussians._rotation.shape[0] != d_rotation.shape[0]:
        d_xyz, d_rotation, d_scaling = 0, 0, 0
        print('Async in Gaussian Switching')
    out = render(viewpoint_camera=cur_cam, pc=gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, render_motion=render_motion, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=self.deform.d_rot_as_res, gs_rot_bias=gs_rot_bias, scale_const=vis_scale_const)

    buffer_image = out[self.mode]  # [3, H, W]
    return buffer_image
