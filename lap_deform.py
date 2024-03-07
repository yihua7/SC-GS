import torch
import torch.nn as nn
import pytorch3d.ops
from utils.arap_deform import ARAPDeformer
from utils.deform_utils import cal_arap_error


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


class LapDeform(nn.Module):
    def __init__(self, init_pcl, K=4, trajectory=None, node_radius=None):
        super().__init__()
        self.K = K
        self.N = init_pcl.shape[0]
        nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(init_pcl[None], init_pcl[None], None, None, K=K+1)  # N, K
        nn_dist, nn_idxs = nn_dist[0,:,1:], nn_idxs[0,:,1:]
        nn_dist = 1 / (nn_dist + 1e-7)
        self.nn_idxs = nn_idxs
        self._weight = nn.Parameter(torch.log(nn_dist / (nn_dist.sum(dim=1, keepdim=True) + 1e-5) + 1e-5))
        self.init_pcl = init_pcl
        self.init_pcl_copy = init_pcl.clone()
        self.tensors = {}
        # self.optimizer = torch.optim.Adam([self._weight], lr=1e-5)
        self.mask_control_points = False
        if self.mask_control_points:
            self.generate_mask_init_pcl()
            radius = torch.linalg.norm(self.init_pcl_reduced.max(dim=0).values - self.init_pcl_reduced.min(dim=0).values) / 10 * 3
            print("Set ball query radius to %f" % radius.item())
            self.arap_deformer = ARAPDeformer(self.init_pcl_reduced, radius=radius, K=30, point_mask=self.init_pcl_mask, trajectory=trajectory, node_radius=node_radius)
        else:
            radius = torch.linalg.norm(self.init_pcl.max(dim=0).values - self.init_pcl.min(dim=0).values) / 8
            print("Set ball query radius to %f" % radius.item())
            self.arap_deformer = ARAPDeformer(init_pcl, radius=radius, K=16, trajectory=trajectory, node_radius=node_radius)

        self.optimizer = torch.optim.Adam([self.arap_deformer.weight], lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)
        self.optim_step = 0

    def generate_mask_init_pcl(self):
        init_pcl_mask = torch.linalg.norm(self.init_pcl, dim=-1) < 5
        self.init_pcl_mask = init_pcl_mask
        # init_pcl[~init_pcl_mask] = 0
        self.init_pcl_reduced = self.init_pcl[self.init_pcl_mask]


    def reset(self, ):
        self.init_pcl = self.init_pcl_copy.clone()
        self.arap_deformer.reset()
        self.optim_step = 0
        self.generate_mask_init_pcl()
    
    @property
    def weight(self):
        return torch.softmax(self._weight, dim=-1)

    @property
    def L(self):
        L = torch.eye(self.N).cuda()
        L.scatter_add_(dim=1, index=self.nn_idxs, src=-self.weight)
        return L
    
    def add_one_ring_nbs(self, idxs):
        if type(idxs) is list:
            idxs = torch.tensor(idxs).cuda()
        elif idxs.dim() == 0:
            idxs = idxs[None]
        nn_idxs = self.nn_idxs[idxs].reshape([-1])
        return torch.unique(torch.cat([nn_idxs, idxs]))
    
    def add_n_ring_nbs(self, idxs, n=2):
        for i in range(n):
            idxs = self.add_one_ring_nbs(idxs)
        return idxs

    def initialize(self, pcl):
        b = self.L @ pcl
        self.tensors['b'] = b

    def estimate_R(self, pcl, return_quaternion=True):
        old_edges = torch.gather(input=self.init_pcl[:, None].repeat(1,self.K,1), dim=0, index=self.nn_idxs[..., None].repeat(1,1,3)) - self.init_pcl[:, None]  # N, K, 3
        edges = torch.gather(input=pcl[:, None].repeat(1,self.K,1), dim=0, index=self.nn_idxs[..., None].repeat(1,1,3)) - pcl[:, None]  # N, K, 3
        D = torch.diag_embed(self.weight, dim1=1, dim2=2)  # N, K, K
        S = torch.bmm(old_edges.permute(0, 2, 1), torch.bmm(D, edges))  # N, 3, 3
        unchanged = torch.unique(torch.where((edges == old_edges).all(dim=1))[0])
        S[unchanged] = 0
        U, _, W = torch.svd(S)
        R = torch.bmm(W, U.permute(0, 2, 1))
        if return_quaternion:
            q = matrix_to_quaternion(R)
            return q
        else:
            return R

    def energy(self, pcl, prev_pcl=None):
        if prev_pcl is None:
            if 'b' not in self.tensors:
                print('Have not initialized yet and start with init pcl')
                self.initialize(self.init_pcl)
            b = self.tensors['b']
        else:
            b = self.L @ prev_pcl
        loss = (self.L @ pcl - b).square().mean()
        return loss
    
    def energy_arap(self, pcl, prev_pcl):
        # loss = (self.arap_deformer.L_opt @ pcl - b).square().mean()
        self.optim_step += 1
        self.arap_deformer.cal_L_opt()
        node_seq = torch.stack([prev_pcl, pcl], dim=0)
        # print(self.arap_deformer.weight)
        loss = cal_arap_error(node_seq, self.arap_deformer.ii, self.arap_deformer.jj, self.arap_deformer.nn, K=self.arap_deformer.K, weight=self.arap_deformer.normalized_weight)
        return loss

    def deform(self, handle_idx, handle_pos, static_idx=None):
        if 'b' not in self.tensors:
            print('Have not initialized yet and start with init pcl')
            self.initialize(self.init_pcl)
        b = self.tensors['b']
        handle_pos = torch.tensor(handle_pos).float().cuda()
        if static_idx is not None:
            static_pos = self.init_pcl[static_idx]
            handle_idx = handle_idx + static_idx
            handle_pos = torch.cat([handle_pos.cuda(), static_pos.cuda()], dim=0)
        return lstsq_with_handles(A=self.L, b=b, handle_idx=handle_idx, handle_pos=handle_pos)
    
    def deform_arap(self, handle_idx, handle_pos, init_verts=None, return_R=False):
        handle_idx = torch.tensor(handle_idx).long().cuda()
        if type(handle_pos) is not torch.Tensor:
            handle_pos = torch.from_numpy(handle_pos).float().cuda()
        deformed_p, deformed_r, deformed_s = self.arap_deformer.deform(handle_idx, handle_pos, init_verts=init_verts, return_R=return_R)
        if self.mask_control_points:
            deformed_p_all = self.init_pcl.clone()
            deformed_p_all[self.init_pcl_mask] = deformed_p
            deformed_r_all = torch.tensor([[1,0,0,0]]).to(deformed_r.dtype).to(deformed_r.device).repeat(deformed_p_all.shape[0],1)
            deformed_r_all[self.init_pcl_mask] = deformed_r
            return deformed_p_all, deformed_r_all, deformed_s
        else:
            return deformed_p, deformed_r, deformed_s


def lstsq_with_handles(A, b, handle_idx, handle_pos):
    b = b - A[:, handle_idx] @ handle_pos
    handle_mask = torch.zeros_like(A[:, 0], dtype=bool)
    handle_mask[handle_idx] = 1
    L = A[:, handle_mask.logical_not()]
    x = torch.linalg.lstsq(L, b)[0]
    x_out = torch.zeros_like(b)
    x_out[handle_idx] = handle_pos
    x_out[handle_mask.logical_not()] = x
    return x_out

