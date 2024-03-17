from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.ops
import numpy as np
import os
from utils.deform_utils import cal_connectivity_from_points, cal_arap_error, arap_deformation_loss

try:
    from torch_batch_svd import svd
    print('Using speed up torch_batch_svd!')
except:
    svd = torch.svd
    print('Use original torch svd!')

def log1p_safe(x):
  """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))


def expm1_safe(x):
  """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))


def robust_lossfunc(squared_x, alpha, scale):
    r"""Implements the general form of the loss.
    This implements the rho(x, \alpha, c) function described in "A General and
    Adaptive Robust Loss Function", Jonathan T. Barron,
    https://arxiv.org/abs/1701.03077.
    Args:
        squared_x: The residual for which the loss is being computed. x can have
                any shape, and alpha and scale will be broadcasted to match x's shape if
                necessary.
        alpha: The shape parameter of the loss (\alpha in the paper), where more
            negative values produce a loss with more robust behavior (outliers "cost"
            less), and more positive values produce a loss with less robust behavior
            (outliers are penalized more heavily). Alpha can be any value in
            [-infinity, infinity], but the gradient of the loss with respect to alpha
            is 0 at -infinity, infinity, 0, and 2. Must be a tensor of floats with the
            same precision as `x`. Varying alpha allows
            for smooth interpolation between a number of discrete robust losses:
            alpha=-Infinity: Welsch/Leclerc Loss.
            alpha=-2: Geman-McClure loss.
            alpha=0: Cauchy/Lortentzian loss.
            alpha=1: Charbonnier/pseudo-Huber loss.
            alpha=2: L2 loss.
        scale: The scale parameter of the loss. When |x| < scale, the loss is an
            L2-like quadratic bowl, and when |x| > scale the loss function takes on a
            different shape according to alpha. Must be a tensor of single-precision
            floats.
        approximate: a bool, where if True, this function returns an approximate and
            faster form of the loss, as described in the appendix of the paper. This
            approximation holds well everywhere except as x and alpha approach zero.
        epsilon: A float that determines how inaccurate the "approximate" version of
            the loss will be. Larger values are less accurate but more numerically
            stable. Must be great than single-precision machine epsilon.
    Returns:
        The losses for each element of x, in the same shape and precision as x.
    """
    epsilon = torch.tensor(torch.finfo(torch.float32).eps).to(squared_x.device)
    alpha = torch.tensor(alpha).to(squared_x.dtype).to(squared_x.device)
    scale = torch.tensor(scale).to(squared_x.dtype).to(squared_x.device)
    # Compute the exact loss.
    # This will be used repeatedly.
    squared_scaled_x = squared_x / (scale ** 2)
    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.max(epsilon, torch.abs(alpha - 2.))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha)) * torch.max(epsilon, torch.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
            torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
            alpha == -float('inf'), loss_neginf,
            torch.where(
                    alpha == 0, loss_zero,
                    torch.where(
                            alpha == 2, loss_two,
                            torch.where(alpha == float('inf'), loss_posinf,
                                                    loss_otherwise))))
    return scale * loss


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class ProgressiveBandFrequency(nn.Module):
    def __init__(self, in_channels: int, n_frequencies=12, no_masking_step=5000):
        super().__init__()
        self.N_freqs = n_frequencies
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = no_masking_step
        self.cur_step = nn.Parameter(torch.tensor(-1), requires_grad=False)
        self.update_step(0)

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq * x) * mask]
        return torch.cat(out, -1)

    def update_step(self, global_step):
        if global_step > self.cur_step.item():
            if self.n_masking_step <= 0 or global_step is None or not self.training:
                self.mask = torch.ones(self.N_freqs, dtype=torch.float32, device=torch.device("cuda:0"))
            else:
                self.mask = (1.0 - torch.cos(torch.pi* (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs, device=torch.device("cuda:0"))).clamp(0, 1))) / 2.0
                # print(f"Update mask of Freq: {global_step}/{self.n_masking_step} {self.mask}")
            self.cur_step.data = torch.ones_like(self.cur_step) * global_step


class StaticNetwork(nn.Module):
    def __init__(self, return_tensors=False, *args, **kwargs) -> None:
        super().__init__()
        self.name = 'static'
        self.param = nn.Parameter(torch.zeros([1]).cuda())
        self.reg_loss = 0.
        self.return_tensors = return_tensors

    def forward(self, x, t, **kwargs):
        if self.return_tensors:
            return_dict = {'d_xyz': torch.zeros_like(x), 'd_rotation': torch.zeros_like(x[..., [0, 0, 0, 0]]), 'd_scaling': torch.zeros_like(x), 'local_rotation': torch.zeros_like(x[..., [0, 0, 0, 0]]), 'hidden': None, 'd_opacity':None, 'd_color': None}
        else:
            return_dict = {'d_xyz': 0., 'd_rotation': 0., 'd_scaling': 0., 'hidden': 0., 'd_opacity':None, 'd_color': None}
        return return_dict
    
    def trainable_parameters(self):
        return [{'params': [self.param], 'name': 'deform'}]
    
    def update(self, *args, **kwargs):
        return


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, t_multires=6, multires=10,
                 is_blender=False, local_frame=False, pred_opacity=False, pred_color=False, resnet_color=True, hash_color=False, color_wrt_dir=False, progressive_brand_time=False, max_d_scale=-1, **kwargs):  # t_multires 6 for D-NeRF; 10 for HyperNeRF
        super(DeformNetwork, self).__init__()
        self.name = 'mlp'
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.progressive_brand_time = progressive_brand_time
        if self.progressive_brand_time:
            self.embed_time_fn = ProgressiveBandFrequency(in_channels=1, n_frequencies=self.t_multires)
            time_input_ch = self.embed_time_fn.n_output_dims
        else:
            self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        self.pred_opacity = pred_opacity
        self.pred_color = pred_color
        self.resnet_color = resnet_color
        self.hash_color = not resnet_color and hash_color
        self.color_wrt_dir = color_wrt_dir
        self.max_d_scale = max_d_scale

        self.reg_loss = 0.

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out))

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W)
                    for i in range(D - 1)]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)] + [
                    nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                    for i in range(D - 1)]
            )

        self.is_blender = is_blender

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_scaling = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)

        self.local_frame = local_frame
        if self.local_frame:
            self.local_rotation = nn.Linear(W, 4)
            nn.init.normal_(self.local_rotation.weight, mean=0, std=1e-4)
            nn.init.zeros_(self.local_rotation.bias)
        
        for layer in self.linear:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        nn.init.normal_(self.gaussian_warp.weight, mean=0, std=1e-5)
        nn.init.normal_(self.gaussian_scaling.weight, mean=0, std=1e-8)
        nn.init.normal_(self.gaussian_rotation.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.gaussian_warp.bias)
        nn.init.zeros_(self.gaussian_scaling.bias)
        nn.init.zeros_(self.gaussian_rotation.bias)

        if self.pred_opacity:
            self.gaussian_opacity = nn.Linear(W, 1)
            nn.init.normal_(self.gaussian_opacity.weight, mean=0, std=1e-5)
            nn.init.zeros_(self.gaussian_opacity.bias)
        if self.pred_color:
            if self.resnet_color:
                in_dim = xyz_input_ch + W if self.color_wrt_dir else self.linear[0].weight.shape[-1] + W  # Color depends on Direction or Position-Time
                # self.gaussian_color = MLP(dim_in=in_dim, dim_out=3, n_neurons=W, n_hidden_layers=3, skip=[])
                self.gaussian_color = nn.Sequential(nn.Linear(in_dim, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
                for layer in self.gaussian_color:
                    if hasattr(layer, 'weight'):
                        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                nn.init.normal_(self.gaussian_color[-1].weight, mean=0, std=1e-5)
            elif self.hash_color:
                self.color_hash_encoding = ProgressiveBandHashGrid(in_channels=4, start_level=16, n_levels=16)
                self.gaussian_color = MLP(dim_in=self.color_hash_encoding.n_output_dims+W, dim_out=3, n_neurons=W//4, n_hidden_layers=3)
            else:
                self.gaussian_color = nn.Linear(W, 3)
                nn.init.normal_(self.gaussian_color.weight, mean=0, std=1e-5)
                nn.init.zeros_(self.gaussian_color.bias)
    
    def trainable_parameters(self):
        return [{'params': list(self.parameters()), 'name': 'mlp'}]

    def forward(self, x, t, **kwargs):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        if self.max_d_scale > 0:
            scaling = torch.tanh(scaling) * np.log(self.max_d_scale)

        return_dict = {'d_xyz': d_xyz, 'd_rotation': rotation, 'd_scaling': scaling, 'hidden': h}
        if self.pred_opacity:
            return_dict['d_opacity'] = self.gaussian_opacity(h)
        else:
            return_dict['d_opacity'] = None
        if self.pred_color:
            if self.resnet_color:
                if self.color_wrt_dir:
                    if 'camera_center' in kwargs:
                        dir_pp = (x - kwargs['camera_center'].repeat(x.shape[0], 1))
                        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        return_dict['d_color'] = self.gaussian_color(torch.cat([self.embed_fn(dir_pp_normalized), h], dim=-1))
                    else:
                        return_dict['d_color'] = None
                else:
                    return_dict['d_color'] = self.gaussian_color(torch.cat([x_emb, t_emb, h], dim=-1))
            elif self.hash_color:
                return_dict['d_color'] = self.gaussian_color(torch.cat([self.color_hash_encoding(torch.cat([x, t], dim=-1)), h], dim=-1))
            else:
                return_dict['d_color'] = self.gaussian_color(h)
        else:
            return_dict['d_color'] = None
        if self.local_frame:
            return_dict['local_rotation'] = self.local_rotation(h)
        return return_dict
    
    def update(self, iteration, *args, **kwargs):
        if self.progressive_brand_time:
            self.embed_time_fn.update_step(iteration)
        return


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def landmark_interpolate(landmarks, steps, step, interpolation='log'):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    elif stage == 0:
        return 0
    else:
        ldm1, ldm2 = landmarks[stage-1], landmarks[stage]
        if ldm2 <= 0:
            return 0
        step1, step2 = steps[stage-1], steps[stage]
        ratio = (step - step1) / (step2 - step1)
        if interpolation == 'log':
            return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
        elif interpolation == 'linear':
            return ldm1 * (1 - ratio) + ldm2 * ratio
        else:
            print(f'Unknown interpolation type: {interpolation}')
            raise NotImplementedError


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


class ProgressiveBandHashGridCosine(nn.Module):
    def __init__(self, in_channels, start_level=6, n_levels=12, start_step=1000, update_steps=1000, dtype=torch.float32):
        super().__init__()

        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0,
            "interpolation": "Linear",
            "start_level": start_level,
            "start_step": start_step,
            "update_steps": update_steps,
        }
        import tinycudann as tcnn # import when necessary

        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = encoding_config["n_levels"]
        self.n_features_per_level = encoding_config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            encoding_config["start_level"],
            encoding_config["start_step"],
            encoding_config["update_steps"],
        )
        
        self.register_buffer('current_step', torch.tensor(0, dtype=torch.int32))
        self.n_masking_step = (n_levels - start_level) * update_steps

        self.mask = torch.ones(self.n_output_dims, dtype=torch.float32, device=torch.device("cuda:0"))
        self.update_step(self.current_step)

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, global_step):
        if global_step < self.current_step.item() or global_step < 0:
            return
        if self.n_masking_step <= 0 or global_step is None or global_step < 0:
            self.mask = torch.ones(self.n_output_dims, dtype=torch.float32, device=torch.device("cuda:0"))
        else:
            self.current_step.data = global_step * torch.ones_like(self.current_step)
            ratio = global_step / self.n_masking_step
            start_idx = self.start_level * self.n_features_per_level
            band_len = self.n_output_dims - start_idx
            self.mask[start_idx:] = (1.0 - torch.cos(torch.pi* (ratio* band_len - torch.arange(0, band_len, device=torch.device("cuda:0"))).clamp(0, 1))) / 2.0
            # print(f"Update mask of Freq: {global_step}/{self.n_masking_step} {self.mask}")


class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels, start_level=6, n_levels=12, start_step=1000, update_steps=1000, dtype=torch.float32):
        super().__init__()

        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,  # 16 for complex motions
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0,
            "interpolation": "Linear",
            "start_level": start_level,
            "start_step": start_step,
            "update_steps": update_steps,
        }
        import tinycudann as tcnn # import when necessary

        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = encoding_config["n_levels"]
        self.n_features_per_level = encoding_config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            encoding_config["start_level"],
            encoding_config["start_step"],
            encoding_config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )
        self.mask[: self.current_level * self.n_features_per_level] = 1.0

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask + enc.detach() * (1 - self.mask)
        return enc

    def update_step(self, global_step):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            print(f"Update current level of HashGrid to {current_level}")
            self.current_level = current_level
            self.mask[: self.current_level * self.n_features_per_level] = 1.0


class TCNNNetwork(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_neurons:int=32, n_hidden_layers:int=1, out_act=nn.Identity()) -> None:
        super().__init__()
        config = {
            "otype": "FullyFusedMLP",    # Component type.
            "activation": "ReLU",        # Activation of hidden layers.
            "output_activation": "None", # Activation of the output layer.
            "n_neurons": n_neurons,             # Neurons in each hidden layer.
                                         # May only be 16, 32, 64, or 128.
            "n_hidden_layers": n_hidden_layers,        # Number of hidden layers.
        }
        import tinycudann as tcnn # import when necessary
        with torch.cuda.device(get_rank()):
            self.network = tcnn.Network(dim_in, dim_out, config)
        self.out_act = out_act

    def forward(self, x):
        return self.out_act(self.network(x))  # transform to float32
    

class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_neurons:16, n_hidden_layers=1, out_act=nn.Identity(), skip=[]):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = n_neurons, n_hidden_layers
        self.skip = skip
        self.skip_idx = []
        self.layers = nn.ModuleList([])
        self.layers.append(self.make_linear(dim_in, self.n_neurons))
        self.layers.append(self.make_activation())
        for i in range(self.n_hidden_layers - 1):
            if i in self.skip:
                self.skip_idx.append((i+1)*2)
            hidden_in_dim = self.n_neurons + dim_in if i in self.skip else self.n_neurons
            self.layers.append(self.make_linear(hidden_in_dim, self.n_neurons))
            self.layers.append(self.make_activation())
        self.layers.append(self.make_linear(self.n_neurons, dim_out, bias=True))
        self.output_activation = out_act

    def forward(self, x):
        h = x
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i in self.skip_idx:
                h = torch.cat([h, x], dim=-1)
            h = layer(h)
        y = self.output_activation(h)
        return y

    def make_linear(self, dim_in, dim_out, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        if bias:
            nn.init.zeros_(layer.bias)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)


def scale_tensor(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, torch.Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def contract_to_unisphere(x, bbox=(-1, 1), unbounded: bool = False):
    if unbounded:
        x = scale_tensor(x, bbox, (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag > 1
        x = torch.where(mask.expand_as(x), (2 - 1 / mag) * (x / mag), x)
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        x = scale_tensor(x, bbox, (0, 1))
    return x


class HashDeformNetwork(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=256, hash_time=False, tcnn_mlp=False, t_multires=6, bbox=None, scale_range=2., local_frame=False, pred_opacity=False, pred_color=False):
        super().__init__()
        self.hash_time = hash_time
        self.tcnn_mlp = tcnn_mlp
        self.register_buffer('bbox', torch.tensor([-2, 2]).float().cuda() if bbox is None else torch.tensor(bbox).float().cuda())
        self.hashgrid = ProgressiveBandHashGridCosine(in_channels=4 if hash_time else 3, start_level=6, n_levels=12, start_step=1000, update_steps=1000)
        mlp_type = TCNNNetwork if tcnn_mlp else MLP
        if not hash_time:
            self.time_embed_func, self.time_embed_dim = get_embedder(t_multires, 1)
        self.mlp = mlp_type(dim_in=self.hashgrid.n_output_dims if hash_time else self.hashgrid.n_output_dims + self.time_embed_dim, dim_out=hidden_dim, n_neurons=hidden_dim, n_hidden_layers=num_layers if hash_time else num_layers+2, out_act=nn.Identity(), skip=[2] if not hash_time else [])
        self.translate_layer = MLP(dim_in=hidden_dim, dim_out=3, n_neurons=64, n_hidden_layers=1)
        self.rotation_layer = MLP(dim_in=hidden_dim, dim_out=4, n_neurons=64, n_hidden_layers=1)
        self.scaling_layer = MLP(dim_in=hidden_dim, dim_out=3, n_neurons=64, n_hidden_layers=1)
        self.log_scale_range = np.log(scale_range)
        if not self.hash_time:
            nn.init.normal_(self.mlp.layers[-1].weight, mean=0., std=1e-5)
        self.local_frame = local_frame
        if self.local_frame:
            self.local_rotation_layer = MLP(dim_in=hidden_dim, dim_out=4, n_neurons=64, n_hidden_layers=1)
        self.pred_opacity = pred_opacity
        if self.pred_opacity:
            self.opacity_layer = MLP(dim_in=hidden_dim, dim_out=1, n_neurons=64, n_hidden_layers=1)
        self.pred_color = pred_color
        if self.pred_color:
            self.color_layer = MLP(dim_in=hidden_dim, dim_out=3, n_neurons=64, n_hidden_layers=1)

    def forward(self, x, t, **kwargs):
        x = contract_to_unisphere(x, bbox=self.bbox, unbounded=False)
        if self.hash_time:
            xt = torch.cat([x, t], dim=-1)
            embed = self.hashgrid(xt)
        else:
            x_embed = self.hashgrid(x)
            t_embed = self.time_embed_func(t)
            t_embed = t_embed / self.time_embed_dim * self.hashgrid.n_output_dims  # Align the scale
            embed = torch.cat([x_embed, t_embed], dim=-1)
        hidden = self.mlp(embed).float()
        translate = self.translate_layer(hidden)
        rotation = self.rotation_layer(hidden) + torch.tensor([1., 0, 0, 0]).float().to(hidden.device)
        scaling = torch.tanh(self.scaling_layer(hidden)) * self.log_scale_range
        return_dict = {'d_xyz': translate, 'd_rotation': rotation, 'd_scaling': scaling}
        if self.pred_opacity:
            return_dict['d_opacity'] = self.opacity_layer(hidden)
        else:
            return_dict['d_opacity'] = None
        if self.pred_color:
            return_dict['d_color'] = self.color_layer(hidden)
        else:
            return_dict['d_color'] = None
        if self.local_frame:
            return_dict['local_rotation'] = self.local_rotation_layer(hidden)
        return return_dict
    
    def update(self, global_step):
        self.hashgrid.update_step(global_step=global_step)


class ControlNodeWarp(nn.Module):
    def __init__(self, is_blender, init_pcl=None, node_num=512, K=3, use_hash=False, hash_time=False, enable_densify_prune=False, pred_opacity=False, pred_color=False, with_arap_loss=False, with_node_weight=True, local_frame=False, d_rot_as_res=True, skinning=False, hyper_dim=2, progressive_brand_time=False, max_d_scale=-1, is_scene_static=False, **kwargs):
        super().__init__()
        self.K = K
        self.use_hash = use_hash
        self.hash_time = hash_time
        self.enable_dp = enable_densify_prune
        self.name = 'node'
        self.with_node_weight = with_node_weight
        self.reg_loss = 0.
        self.local_frame = local_frame
        self.d_rot_as_res = d_rot_as_res
        self.hyper_dim = hyper_dim if not skinning else 0  # skinning should not be with hyper
        self.is_blender = is_blender
        self.pred_opacity = pred_opacity
        self.pred_color = pred_color
        self.max_d_scale = max_d_scale
        self.is_scene_static = is_scene_static
        
        self.skinning = skinning  # As skin model, discarding KNN weighting
        if with_arap_loss and not self.is_scene_static:
            self.lambda_arap_landmarks = [ 1e-4,  1e-4,  1e-5,  1e-5,     0]
            self.lambda_arap_steps =     [    0,  5000, 10000, 20000, 20001]
        else:
            self.lambda_arap_landmarks = [0]
            self.lambda_arap_steps =     [0]

        # Initialize Network
        if self.is_scene_static:
            self.network = StaticNetwork(return_tensors=True)
        elif use_hash:
            self.network = HashDeformNetwork(local_frame=local_frame, pred_opacity=pred_opacity, pred_color=pred_color, hash_time=hash_time).cuda()
        else:
            self.network = DeformNetwork(is_blender=is_blender, local_frame=local_frame, pred_opacity=pred_opacity, pred_color=pred_color, progressive_brand_time=progressive_brand_time, max_d_scale=max_d_scale).cuda()
        
        self.register_buffer('inited', torch.tensor(False))
        self.nodes = nn.Parameter(torch.randn(node_num, 3+self.hyper_dim))
        if not self.skinning:
            self._node_radius = nn.Parameter(torch.randn(node_num))
            if self.with_node_weight:
                self._node_weight = nn.Parameter(torch.zeros_like(self.nodes[:, :1]), requires_grad=with_node_weight)
        if init_pcl is not None:
            self.init(init_pcl)

        # Node colors for visualization
        self.nodes_color_visualization = torch.ones_like(self.nodes)

        # Cached nn_weight to speed up
        self.cached_nn_weight = False
        self.nn_weight, self.nn_dist, self.nn_idxs = None, None, None
    
    def update(self, iteration):
        self.network.update(iteration)
    
    def trainable_parameters(self):
        if self.skinning:
            return [{'params': list(self.network.parameters()), 'name': 'deform'},
                    {'params': [self.nodes], 'name': 'nodes'}]
        elif self.with_node_weight:
            return [{'params': list(self.network.parameters()), 'name': 'deform'},
                    {'params': [self.nodes, self._node_radius, self._node_weight], 'name': 'nodes'}]
        else:
            return [{'params': list(self.network.parameters()), 'name': 'deform'},
                    {'params': [self.nodes, self._node_radius], 'name': 'nodes'}]
    
    @property
    def param_names(self):
        if self.skinning:
            param_names = ['nodes', 'deform']
        elif self.with_node_weight:
            param_names = ['nodes', '_node_radius', '_node_weight']
        else:
            param_names = ['nodes', '_node_radius']
        return param_names

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        param_names = self.param_names
        for key in state_dict:
            if key in param_names:
                node_param = state_dict[key]
                if getattr(self, key).shape != node_param.shape:
                    print(f'Loading nodes mismatching the original setting: {getattr(self, key).shape} and {node_param.shape}')
                    setattr(self, key, nn.Parameter(node_param))
                else:
                    getattr(self, key).data = node_param
            elif key.startswith('gs_'):
                name = key[3:]
                try:
                    getattr(self.as_gaussians, name).data = state_dict[key]
                except:
                    print(f'Directly set as values for {key} when loading deform gaussians')
                    setattr(self.as_gaussians, name, state_dict[key])
        for key in param_names:
            if key in state_dict:
                state_dict.pop(key)
        super().load_state_dict(state_dict=state_dict, strict=False)

    def state_dict(self):
        state_dict = super().state_dict()
        if hasattr(self, 'gs') and self.gs is not None:
            for name in self.gs.param_names():
                state_dict['gs_'+name] = getattr(self.gs, name)
        return state_dict

    @property
    def node_radius(self):
        return torch.exp(self._node_radius)
    
    @property
    def node_weight(self):
        return torch.sigmoid(self._node_weight)
    
    @property
    def node_num(self):
        return self.nodes.shape[0]
    
    def init(self, opt, init_pcl, hyper_pcl=None, keep_all=False, force_init=False, as_gs_force_with_motion_mask=False, force_gs_keep_all=False, reset_bbox=True, **kwargs):
        # keep_all: initialize nodes with all init_pcl given. it happens when sample nodes from the isotropic Gaussians right after the node training
        # force_gs_keep_all: initialize isotropic Gaussians with all init_pcl given. it happens in the very beginning of the training.

        # Initialize Nodes
        if self.inited and not force_init:
            return
        self.inited.data = torch.ones_like(self.inited)
        # Decide the bbox of hashgrid
        if self.use_hash and reset_bbox:
            self.network.bbox.data = torch.tensor([init_pcl.min()-.1, init_pcl.max()+.1]).float().cuda()
        self.register_buffer('inited', torch.tensor(True))
        if keep_all or self.node_num > init_pcl.shape[0]:
            self.nodes = nn.Parameter(torch.cat([init_pcl.float(), 1e-2 * torch.ones([init_pcl.shape[0], self.hyper_dim]).float().cuda()], dim=-1))
            init_nodes_idx = None
            print('Initialization with all pcl. Need to reset the optimizer.')
        else:
            pcl_to_samp = init_pcl if hyper_pcl is None else hyper_pcl
            init_nodes_idx = farthest_point_sample(pcl_to_samp.detach()[None], self.node_num)[0]
            self.nodes.data = nn.Parameter(torch.cat([init_pcl[init_nodes_idx].float(), 1e-2 * torch.ones([self.node_num, self.hyper_dim]).float().cuda()], dim=-1))
        scene_range = init_pcl.max() - init_pcl.min()
        if self.skinning:
            if 'feature' in kwargs:
                gs_weights = kwargs['feature']
                radius = .1 * scene_range + 1e-7
                initial_weights = - torch.log((init_pcl[:, None] - self.nodes[None,...,:3]).square().sum(dim=-1) / radius ** 2)
                gs_weights.data = initial_weights
        else:
            if keep_all or self.node_num > init_pcl.shape[0]:
                self._node_radius = nn.Parameter(torch.log(.1 * scene_range + 1e-7) * torch.ones([self.node_num]).float().to(scene_range.device))
                self._node_weight = nn.Parameter(torch.zeros_like(torch.zeros_like(self.nodes[:, :1])))
            else:
                self._node_radius.data = nn.Parameter(torch.log(.1 * scene_range + 1e-7) * torch.ones([self.node_num]).float().to(scene_range.device))
                self._node_weight.data = torch.zeros_like(torch.zeros_like(self.nodes[:, :1]))
        self.gs = None
        if force_gs_keep_all:
            self.init_gaussians(init_pcl=init_pcl, with_motion_mask=as_gs_force_with_motion_mask)
        else:
            self.init_gaussians(init_pcl=self.nodes[..., :3],with_motion_mask=as_gs_force_with_motion_mask)
        self.as_gaussians.training_setup(opt)
        print(f'Control node initialized with {self.nodes.shape[0]} from {init_pcl.shape[0]} points.')
        return init_nodes_idx

    def expand_time(self, t):
        N = self.nodes.shape[0]
        t = t.unsqueeze(0).expand(N, -1)
        return t

    def cal_nn_weight(self, x:torch.Tensor, K=None, feature=None, nodes=None, gs_kernel=True, temperature=1.):
        if self.skinning:
            nn_weight = torch.softmax(feature, dim=-1)
            nn_idx = torch.arange(0, self.node_num, dtype=torch.long).cuda()
            return nn_weight, None, nn_idx
        else:
            if self.cached_nn_weight and self.nn_weight is not None:
                return self.nn_weight, self.nn_dist, self.nn_idxs
            else:
                if self.hyper_dim > 0 and feature is not None:
                    x = torch.cat([x.detach(), feature[..., :self.hyper_dim]], dim=-1)  # cat with hyper coor
                K = self.K if K is None else K
                # Weights of control nodes
                nodes = self.nodes[..., :3].detach() if nodes is None else nodes[..., :3]
                if feature is not None:
                    nodes = torch.cat([nodes[..., :3].detach(), self.nodes[..., 3:]], dim=-1)  # Freeze the first 3 coordinates for deformation mlp input
                nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)  # N, K
                nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]  # N, K
                if gs_kernel:
                    nn_radius = self.node_radius[nn_idxs]  # N, K
                    nn_weight = torch.exp(- nn_dist / (2 * nn_radius ** 2))  # N, K
                    if self.with_node_weight:
                        nn_node_weight = self.node_weight[nn_idxs]
                        nn_weight = nn_weight * nn_node_weight[..., 0]
                    nn_weight = nn_weight + 1e-7
                    nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
                    if self.cached_nn_weight:
                        self.nn_weight = nn_weight
                        self.nn_dist = nn_dist
                        self.nn_idxs = nn_idxs
                    return nn_weight, nn_dist, nn_idxs
                else:
                    nn_weight = torch.softmax(- nn_dist / temperature, dim=-1)
                    return nn_weight, nn_dist, nn_idxs
    
    def cal_nn_weight_floyd(self, x:torch.Tensor, t0:torch.Tensor, cur_node:torch.Tensor, K=None, GraphK=2, temperature=1., cache_name='floyd', XisNode=False):
        if not hasattr(self, f'{cache_name}_nn_dist') or (t0 is not None and (getattr(self, f'{cache_name}_t') - t0).abs().max() > 1e-2):
            node_dist_mat = self.geodesic_distance_floyd(cur_node=cur_node, K=GraphK)
            floyd_nn_dist, floyd_nn_idx = node_dist_mat.sort(dim=1)
            offset = 1 if XisNode else 0
            floyd_nn_dist = floyd_nn_dist[:, offset:K+offset]
            floyd_nn_idx = floyd_nn_idx[:, offset:K+offset]
            setattr(self, f'{cache_name}_nn_dist', floyd_nn_dist)
            setattr(self, f'{cache_name}_nn_idx', floyd_nn_idx)
            if t0 is not None:
                setattr(self, f'{cache_name}_t', t0.clone())
        nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], cur_node[None], None, None, K=1)  # N, K
        nn_dist, nn_idxs = nn_dist[0, :, 0], nn_idxs[0, :, 0]  # N
        knn_dist, knn_idxs = getattr(self, f'{cache_name}_nn_dist')[nn_idxs] + nn_dist[:, None], getattr(self, f'{cache_name}_nn_idx')[nn_idxs]
        knn_weight = torch.softmax(- knn_dist / temperature, dim=-1)
        return knn_weight, knn_dist, knn_idxs

    def query_network(self, x, t, **kwargs):
        values = self.network(x=x, t=t, **kwargs)
        return values
    
    def node_deform(self, t, detach_node=True, **kwargs):
        tshape = t.shape
        if t.dim() == 3:
            assert t.shape[0] == self.node_num, f'Shape of t {t.shape} does not match the shape of nodes {self.nodes.shape}'
            nodes = self.nodes[:, None, ..., :3].expand(self.node_num, t.shape[1], 3).reshape(-1, 3)
            t = t.reshape(-1, 1)
        else:
            nodes = self.nodes[..., :3]
        if detach_node:
            nodes = nodes.detach()
        values = self.query_network(x=nodes, t=t, **kwargs)
        values = {key: values[key].view(*tshape[:-1], values[key].shape[-1]) if values[key] is not None else None for key in values}
        return values
    
    @torch.no_grad()
    def sample_node_deform(self, samp_num=512, sv_path='./deform'):
        t = torch.linspace(0, 1, samp_num).float().cuda()
        chunk = 16
        start = 0
        values = {}
        while start < samp_num:
            end = min(start + chunk, samp_num)
            t_ = t[None, start: end, None].expand(self.node_num, end-start, 1)
            values_ = self.node_deform(t_)
            for key in values_:
                if values_[key] is not None:
                    values_[key] = values_[key].permute(1,0,2)
                    if key not in values:
                        values[key] = values_[key]
                    else:
                        values[key] = torch.cat([values[key], values_[key]], dim=0)
            start = end
        values_np = {key: values[key].detach().cpu().numpy() for key in values if key != 'hidden'}
        np.savez(sv_path, **values_np, nodes=self.nodes.detach().cpu().numpy())
        print(f"Successfully save {values_np.keys()} into {sv_path}! Without hidden features!")
    
    def get_trajectory(self, t_samp_num=8):
        t_samp = torch.linspace(0, 1, t_samp_num).cuda()
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)  # M, T, 1
        node_deform = self.node_deform(t=t_samp)
        trajectory = self.nodes[:, None, :3].detach() + node_deform['d_xyz']  # M, T, 3
        for key in node_deform:
            node_deform[key] = node_deform[key][:, 0] if node_deform[key] is not None else None
        return trajectory.detach(), node_deform

    def arap_loss_with_rot(self, t_samp_num=8):
        t_samp = torch.rand(t_samp_num).cuda()
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)  # M, T, 1
        node_deform = self.node_deform(t=t_samp)
        trajectory = self.nodes[:, None, :3].detach() + node_deform['d_xyz']  # M, T, 3
        trajectory_rot = node_deform['d_rotation'] if not self.d_rot_as_res else None
        arap_error, rot_error = arap_deformation_loss(trajectory=trajectory, node_radius=self.node_radius.detach(), trajectory_rot=trajectory_rot, with_rot=not self.d_rot_as_res)
        return arap_error + rot_error
    
    def p2dR(self, p, p0=None, K=8, as_quat=True, mode='trajectory', t0=None):
        p = p.detach()
        nodes = self.nodes[..., :3].detach()
        if mode == 'trajectory':
            trajectory, t0_deform = self.get_trajectory(t_samp_num=4)
            t0_nodes = trajectory[:, 0] if p0 is None else p0
            trajectory = trajectory.reshape([trajectory.shape[0], -1])
            nn_dist, nn_idx, _ = pytorch3d.ops.knn_points(trajectory[None], trajectory[None], None, None, K=K+1, return_nn=False)
            nn_dist, nn_idx = nn_dist[0, :, 1:], nn_idx[0, :, 1:]
            nn_weight = torch.softmax(nn_dist/nn_dist.mean(), dim=-1)
            edges = torch.gather(t0_nodes[:, None].expand([nodes.shape[0], K, nodes.shape[-1]]), dim=0, index=nn_idx[..., None].expand([nodes.shape[0], K, nodes.shape[-1]])) - t0_nodes[:, None]
        elif mode == 'floyd':
            nn_weight, _, nn_idx = self.cal_nn_weight_floyd(x=p, t0=t0, cur_node=p0, K=K+1, GraphK=4, temperature=1e-1, cache_name='p2dR', XisNode=True)
            nn_weight, nn_idx = nn_weight[:, 1:], nn_idx[:, 1:]
            edges = torch.gather(p0[:, None].expand([nodes.shape[0], K, nodes.shape[-1]]), dim=0, index=nn_idx[..., None].expand([nodes.shape[0], K, nodes.shape[-1]])) - p0[:, None]
            t0_deform = None
        else:
            nn_dist, nn_idx, nn_nodes = pytorch3d.ops.knn_points(nodes[None], nodes[None], None, None, K=K+1, return_nn=True)
            nn_dist, nn_idx, nn_nodes = nn_dist[0, :, 1:], nn_idx[0, :, 1:], nn_nodes[0, :, 1:]
            nn_weight = torch.softmax(nn_dist/nn_dist.mean(), dim=-1)
            if p0 is None:
                edges = nn_nodes - nodes[:, None]
            else:
                edges = torch.gather(p0[:, None].expand([nodes.shape[0], K, nodes.shape[-1]]), dim=0, index=nn_idx[..., None].expand([nodes.shape[0], K, nodes.shape[-1]])) - p0[:, None]
            t0_deform = None
        edges_t = torch.gather(p[:, None].expand([p.shape[0], K, p.shape[-1]]), dim=0, index=nn_idx[..., None].expand([p.shape[0], K, p.shape[-1]])) - p[:, None]
        edges, edges_t = edges / (edges.norm(dim=-1, keepdim=True) + 1e-5), edges_t / (edges_t.norm(dim=-1, keepdim=True) + 1e-5)
        W = torch.zeros([edges.shape[0], K, K], dtype=torch.float32, device=edges.device)
        W[:, range(K), range(K)] = nn_weight
        S = torch.einsum('nka,nkg,ngb->nab', edges, W, edges_t)
        U, _, V = svd(S)
        dR = torch.matmul(V, U.permute(0, 2, 1))
        if as_quat:
            dR = matrix_to_quaternion(dR)
        return dR, t0_deform

    def arap_loss(self, t=None, delta_t=0.05, t_samp_num=2):
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)  # M, T, 1
        node_trans = self.node_deform(t=t_samp)['d_xyz']
        nodes_t = self.nodes[:, None, :3].detach() + node_trans  # M, T, 3
        hyper_nodes = nodes_t[:,0]  # M, 3
        ii, jj, nn, weight = cal_connectivity_from_points(hyper_nodes, K=10)  # connectivity of control nodes
        error = cal_arap_error(nodes_t.permute(1,0,2), ii, jj, nn)
        return error
    
    def elastic_loss(self, t=None, delta_t=0.005, K=2, t_samp_num=8):
        # Calculate nodes translate
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)
        node_trans = self.node_deform(t=t_samp)['d_xyz']
        nodes_t = self.nodes[:, None, :3].detach() + node_trans  # M, T, 3

        # Calculate weights of nodes NN
        nn_weight, _, nn_idx = self.cal_nn_weight(x=self.nodes[..., :3].detach(), feature=self.nodes[..., 3:], K=K+1)
        nn_weight, nn_idx = nn_weight[:, 1:], nn_idx[:, 1:]  # M, K

        # Calculate edge deform loss
        edge_t = (nodes_t[nn_idx] - nodes_t[:, None]).norm(dim=-1)  # M, K, T
        edge_t_var = edge_t.var(dim=2)  # M, K
        edge_t_var = edge_t_var / (edge_t_var.detach() + 1e-5)
        arap_loss = (edge_t_var * nn_weight).sum(dim=1).mean()
        return arap_loss
    
    def acc_loss(self, t=None, delta_t=.005):
        # Calculate nodes translate
        t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        t = torch.stack([t-delta_t, t, t+delta_t])
        t = t[None, :, None].expand(self.node_num, 3, 1)
        node_trans = self.node_deform(t=t)['d_xyz']
        nodes_t = self.nodes[:, None, :3].detach() + node_trans  # M, 3, 3
        acc = (nodes_t[:, 0] + nodes_t[:, 2] - 2 * nodes_t[:, 1]).norm(dim=-1)  # M
        acc = acc / (acc.detach() + 1e-5)
        acc_loss = acc.mean()
        return acc_loss
    
    def geodesic_distance_floyd(self, cur_node, K=8):
        node_num = cur_node.shape[0]
        nn_dist, nn_idx, _ = pytorch3d.ops.knn_points(cur_node[None], cur_node[None], None, None, K=K+1)
        nn_dist, nn_idx = nn_dist[0]**.5, nn_idx[0]
        dist_mat = torch.inf * torch.ones([node_num, node_num], dtype=torch.float32, device=cur_node.device)
        dist_mat.scatter_(dim=1, index=nn_idx, src=nn_dist)
        dist_mat = torch.minimum(dist_mat, dist_mat.T)
        for i in range(nn_dist.shape[0]):
            dist_mat = torch.minimum((dist_mat[:, i, None] + dist_mat[None, i, :]), dist_mat)
        return dist_mat
    
    def forward(self, x, t, feature, motion_mask, iteration=0, is_training=True, node_trans_bias=None, node_scaling_bias=None, animation_d_values=None, **kwargs):
        if t.dim() == 0:
            t = self.expand_time(t)
        x = x.detach()
        rot_bias = torch.tensor([1., 0, 0, 0]).float().to(x.device)
        # Calculate nn weights: [N, K]
        nn_weight, _, nn_idx = self.cal_nn_weight(x=x, feature=feature)
        node_attrs = self.node_deform(t=t, **kwargs)
        # Animation
        if animation_d_values is not None:
            for key in animation_d_values:
                node_attrs[key] = animation_d_values[key]
        node_trans, node_rot, node_scale = node_attrs['d_xyz'], node_attrs['d_rotation'], node_attrs['d_scaling']
        
        # Obtain translation
        if self.local_frame:
            local_rot = node_attrs['local_rotation'] + rot_bias
            local_rot_matrix = quaternion_to_matrix(local_rot)
            nn_nodes = self.nodes[nn_idx,...,:3].detach()
            Ax = torch.einsum('nkab,nkb->nka', local_rot_matrix[nn_idx], x[:, None] - nn_nodes) + nn_nodes + node_trans[nn_idx]
            Ax_avg = (Ax * nn_weight[..., None]).sum(dim=1)
            translate = Ax_avg - x
        else:
            translate = (node_trans[nn_idx] * nn_weight[..., None]).sum(dim=1)
        translate = translate * motion_mask

        ##############################################################################
        if not self.d_rot_as_res:
            # Add rot bias [1, 0, 0, 0] to node rot since the initialization is 0
            node_rot = node_rot + rot_bias
            # Predict rotation by SVD on node positions rather than MLP
            if node_trans_bias is not None:
                with torch.no_grad():
                    nodes_t = (self.nodes[..., :3] + node_trans).detach()
                    nodes_t_init = nodes_t
                    nodes_t = nodes_t + node_trans_bias
                    node_rot_bias, _ = self.p2dR(p=nodes_t, p0=nodes_t_init, K=8, as_quat=True, mode='trajectory', t0=t)
                    node_rot = quaternion_multiply(node_rot_bias, node_rot)

            # Interpolate to obtain rotation of Gaussians
            rotation = ((node_rot[nn_idx] * nn_weight[..., None]).sum(dim=1) - rot_bias) * motion_mask + rot_bias
            if node_trans_bias is not None:
                # Calculate nodes and gs at t0
                with torch.no_grad():
                    cur_node = self.nodes[..., :3] + node_trans
                    cur_gs = x + translate
                    cur_nn_weight, _, cur_nn_idx = self.cal_nn_weight(x=cur_gs, feature=None, nodes=cur_node, K=32)

                    node_rot_R = quaternion_to_matrix(node_rot)[cur_nn_idx]
                    gs_init = x + translate
                    # Aligh the relative distance considering the rotation
                    gs_t = nodes_t[cur_nn_idx] + torch.einsum('gkab,gkb->gka', node_rot_R, (gs_init[:, None] - nodes_t_init[cur_nn_idx]))
                    gs_t_avg = (gs_t * cur_nn_weight[..., None]).sum(dim=1)
                    translate = (gs_t_avg - x) * motion_mask
            # Scale residual
            scale = (node_scale[nn_idx] * nn_weight[..., None]).sum(dim=1) * motion_mask
            return_dict = {'d_xyz': translate, 'd_rotation': rotation, 'd_scaling': scale}
        else:
            rotation = (node_rot[nn_idx] * nn_weight[..., None]).sum(dim=1)
            rotation = rotation * motion_mask
            scale = (node_scale[nn_idx] * nn_weight[..., None]).sum(dim=1) * motion_mask
            return_dict = {'d_xyz': translate, 'd_rotation': rotation, 'd_scaling': scale}
            # Animation deformation
            if node_trans_bias is not None:
                with torch.no_grad():
                    cur_node = self.nodes[..., :3] + node_trans
                    cur_gs = x + translate
                    # cur_nn_weight, _, cur_nn_idx = self.cal_nn_weight(x=cur_gs, feature=None, nodes=cur_node, K=16, gs_kernel=False, temperature=1e-3)
                    cur_nn_weight, _, cur_nn_idx = self.cal_nn_weight_floyd(x=cur_gs, t0=t, cur_node=cur_node, K=8, GraphK=3, temperature=1e-3, XisNode=False)

                    nodes_t = cur_node + node_trans_bias
                    node_rot_bias, _ = self.p2dR(p=nodes_t, p0=cur_node, K=8, as_quat=True, mode='trajectory', t0=t)
                    d_rotation_bias = (node_rot_bias[cur_nn_idx] * cur_nn_weight[..., None]).sum(dim=1)
                    d_nn_node_rot_R = quaternion_to_matrix(node_rot_bias)[cur_nn_idx]
                    gs_init = x + translate
                    # Aligh the relative distance considering the rotation
                    gs_t = nodes_t[cur_nn_idx] + torch.einsum('gkab,gkb->gka', d_nn_node_rot_R, (gs_init[:, None] - cur_node[cur_nn_idx]))
                    gs_t_avg = (gs_t * cur_nn_weight[..., None]).sum(dim=1)
                    translate = gs_t_avg - x
                    return_dict['d_xyz'] = translate * motion_mask
                    return_dict['d_rotation_bias'] = ((node_rot_bias[cur_nn_idx] * cur_nn_weight[..., None]).sum(dim=1) - rot_bias) * motion_mask + rot_bias
        
        if self.pred_opacity:
            node_opacity = node_attrs['d_opacity']
            d_opacity = (node_opacity[nn_idx] * nn_weight[..., None]).sum(dim=1) * motion_mask
            return_dict['d_opacity'] = d_opacity
        else:
            return_dict['d_opacity'] = None
        if self.pred_color:
            node_color = node_attrs['d_color']
            d_color = (node_color[nn_idx] * nn_weight[..., None]).sum(dim=1) * motion_mask
            return_dict['d_color'] = d_color
        else:
            return_dict['d_color'] = None

        self.reg_loss = 0.
        lambda_arap = landmark_interpolate(landmarks=self.lambda_arap_landmarks, steps=self.lambda_arap_steps, step=iteration)
        if self.training and lambda_arap > 0 and is_training:
            arap_loss = self.arap_loss()
            self.reg_loss = self.reg_loss + arap_loss * lambda_arap
        return return_dict
   
    @property
    def as_gaussians(self):
        if not hasattr(self, 'gs') or self.gs is None:
            # Building Learnable Gaussians for Nodes
            print('Building Learnable Gaussians for Nodes!')
            from scene.gaussian_model import GaussianModel, BasicPointCloud, StandardGaussianModel
            pcd = BasicPointCloud(points=self.nodes[..., :3].detach(), colors=torch.zeros_like(self.nodes[..., :3]), normals=self.nodes[..., :3].detach())
            self.gs = StandardGaussianModel(sh_degree=0, all_the_same=True, with_motion_mask=False)  # blender datas are all dynamic
            self.gs.create_from_pcd(pcd=pcd, spatial_lr_scale=0., print_info=False)
            self.gs._scaling.data = torch.log(1e-2 * torch.ones_like(self.gs._scaling))
            self.gs._xyz.data = self.nodes[..., :3]
        return self.gs
    
    def init_gaussians(self, init_pcl, with_motion_mask):
        if not hasattr(self, 'gs') or self.gs is None:
            # Building Learnable Gaussians for Nodes
            print('Initialize Learnable Gaussians for Nodes with Point Clouds!')
            from scene.gaussian_model import GaussianModel, BasicPointCloud, StandardGaussianModel
            pcd = BasicPointCloud(points=init_pcl.detach(), colors=torch.zeros_like(init_pcl), normals=torch.zeros_like(init_pcl))
            self.gs = StandardGaussianModel(sh_degree=0, all_the_same=True, with_motion_mask=with_motion_mask)  # blender datas are all dynamic
            self.gs.create_from_pcd(pcd=pcd, spatial_lr_scale=0., print_info=False)
        return self.gs
    
    @property
    def as_gaussians_visualization(self):
        from scene.gaussian_model import GaussianModel, BasicPointCloud
        pcd = BasicPointCloud(points=self.nodes[..., :3].detach(), colors=self.nodes_color_visualization, normals=self.nodes[..., :3].detach())
        gs = GaussianModel(sh_degree=0, with_motion_mask=False)
        gs.create_from_pcd(pcd=pcd, spatial_lr_scale=0., print_info=False)
        gs._scaling.data = torch.log(1e-2 * torch.ones_like(gs._scaling))
        gs._opacity.data = 1e3 * torch.ones_like(gs._opacity)
        return gs
    
    @torch.no_grad()
    def cal_node_importance(self, x:torch.Tensor, K=None, weights=None, feature=None):
        # Calculate the weights of Gaussians on nodes as importance
        if self.hyper_dim > 0:
            x = torch.cat([x, feature[..., :self.hyper_dim]], dim=-1)
        K = self.K if K is None else K
        nn_weight, _, nn_idxs = self.cal_nn_weight(x=x[..., :3], K=K, feature=feature)  # N, K
        node_importance = torch.zeros_like(self.nodes[:, 0]).view(-1)
        node_edge_count = torch.zeros_like(self.nodes[:, 0]).view(-1)
        avg_affected_x = torch.zeros_like(self.nodes)
        weights = torch.ones_like(x[:, 0]) if weights is None else weights
        node_importance.index_add_(dim=0, index=nn_idxs.view(-1), source=(nn_weight * weights[:, None]).view(-1))
        node_edge_count.index_add_(dim=0, index=nn_idxs.view(-1), source=nn_weight.view(-1))
        avg_affected_x.index_add_(dim=0, index=nn_idxs.view(-1), source=((nn_weight * weights[:, None]).view(-1, 1) * x[:, None].expand(*nn_weight.shape, x.shape[-1]).reshape(-1, x.shape[-1])))
        avg_affected_x = avg_affected_x / node_importance[:, None]
        node_importance = node_importance / (node_edge_count + 1e-7)
        return node_importance, avg_affected_x, node_edge_count
    
    @torch.no_grad()
    def densify(self, max_grad, optimizer, x:torch.Tensor, x_grad: torch.Tensor, feature=None, K=None, use_gaussians_grad=False, force_dp=False):
        if not self.enable_dp and not force_dp:
            return
        if not self.inited:
            print('No need to densify nodes before initialization.')
            return
        if self.skinning:
            print('No need to densify for skinning type')
            return

        x_grad[x_grad.isnan()] = 0.
        K = self.K if K is None else K
        weights = x_grad.norm(dim=-1)
        
        # Calculate the avg importance and coor
        node_avg_xgradnorm, node_avg_x, node_edge_count = self.cal_node_importance(x=x, K=K, weights=weights, feature=feature)
        
        # Picking pts to densify
        if use_gaussians_grad or not hasattr(self, 'nodes_accumulated_grad'):
            selected_pts_mask = torch.logical_and(node_avg_xgradnorm > max_grad, node_avg_x.isnan().logical_not().all(dim=-1))
        else:
            avg_nodes_norm = self.nodes_accumulated_grad / self.denom
            selected_pts_mask = avg_nodes_norm > max_grad
            self.nodes_accumulated_grad.data = 0
            self.denom = 0

        # For visualization
        self.nodes_color_visualization = torch.ones_like(self.nodes[..., :3])

        pruned_pts_mask = node_edge_count == 0
        if selected_pts_mask.sum() > 0 or pruned_pts_mask.sum() > 0:
            print(f'Add {selected_pts_mask.sum()} nodes and prune {pruned_pts_mask.sum()} nodes. ', end='')
        else:
            return

        # Densify
        if selected_pts_mask.sum() > 0:
            new_nodes = node_avg_x[selected_pts_mask]
            new_node_radius = self._node_radius[selected_pts_mask]
            new_param_list = [new_nodes, new_node_radius]
            if self.with_node_weight:
                new_node_weight = self._node_weight[selected_pts_mask]
                new_param_list.append(new_node_weight)
            param_list = self.param_names
            param_idx = np.arange(len(param_list))
            for group in optimizer.param_groups:
                if group["name"] != 'nodes':
                    continue
                for i in param_idx:
                    stored_state = optimizer.state.get(group['params'][i], None)
                    extension_tensor = new_param_list[i]
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                        del optimizer.state[group['params'][i]]
                        group["params"][i] = nn.Parameter(torch.cat((group["params"][i], extension_tensor), dim=0).requires_grad_(True))
                        optimizer.state[group['params'][i]] = stored_state
                        setattr(self, param_list[i], group["params"][i])
                    else:
                        group["params"][i] = nn.Parameter(torch.cat((group["params"][i], extension_tensor), dim=0).requires_grad_(True))
                        setattr(self, param_list[i], group["params"][i])
            self.nodes_color_visualization = torch.cat([self.nodes_color_visualization, torch.ones_like(new_nodes[..., :3])], dim=0)
            self.nodes_color_visualization[-new_nodes.shape[0]:, 1:] = 0  # Set as red
        
        # Prune
        if pruned_pts_mask.shape[0] < self.nodes.shape[0]:
            pruned_pts_mask = torch.cat([pruned_pts_mask, torch.zeros([self.nodes.shape[0] - pruned_pts_mask.shape[0]]).to(pruned_pts_mask.device).to(pruned_pts_mask.dtype)])
        if pruned_pts_mask.sum() > 0:
            pruned_pts_mask = ~pruned_pts_mask
            self.nodes_color_visualization = self.nodes_color_visualization[pruned_pts_mask]
            optimizable_tensors = {}
            param_list = self.param_names
            param_idx = np.arange(len(param_list))
            for group in optimizer.param_groups:
                if group["name"] != 'nodes':
                    continue
                for i in param_idx:
                    stored_state = optimizer.state.get(group['params'][i], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][pruned_pts_mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][pruned_pts_mask]
                        del optimizer.state[group['params'][i]]
                        group["params"][i] = nn.Parameter((group["params"][i][pruned_pts_mask].requires_grad_(True)))
                        optimizer.state[group['params'][i]] = stored_state
                        optimizable_tensors[param_list[i]] = group["params"][i]
                    else:
                        group["params"][i] = nn.Parameter(group["params"][i][pruned_pts_mask].requires_grad_(True))
                        optimizable_tensors[param_list[i]] = group["params"][i]
            for key in optimizable_tensors:
                setattr(self, key, optimizable_tensors[key])
        else:
            pruned_pts_mask = ~pruned_pts_mask
        
        if not self.with_node_weight:
            self._node_weight = torch.zeros_like(self.nodes[..., :1])
        
        self.gs.densify_and_split(selected_pts_mask=selected_pts_mask, N=1, without_prune=True)
        self.gs.prune_points(~pruned_pts_mask)
        self.gs._xyz.data = self.nodes[..., :3]
        print(f'With {self.nodes.shape[0]} nodes left.')

