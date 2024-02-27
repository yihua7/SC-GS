import torch


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


def dualquaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_real, b_real = a[..., :4], b[..., :4]
    a_imag, b_imag = a[..., 4:], b[..., 4:]
    o_real = quaternion_multiply(a_real, b_real)
    o_imag = quaternion_multiply(a_imag, b_real) + quaternion_multiply(a_real, b_imag)
    o = torch.cat([o_real, o_imag], dim=-1)
    return o


def conjugation(q):
    if q.shape[-1] == 4:
        q = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    elif q.shape[-1] == 8:
        q = torch.cat([q[..., :1], -q[..., 1:4], q[..., 4:5], -q[..., 5:]], dim=-1)
    else:
        raise TypeError(f'q should be of [..., 4] or [..., 8] but got {q.shape}!')
    return q


def QT2DQ(q, t, rot_as_q=True):
    if not rot_as_q:
        q = matrix_to_quaternion(q)
    q = torch.nn.functional.normalize(q)
    real = q
    t = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)
    image = quaternion_multiply(t, q) / 2
    dq = torch.cat([real, image], dim=-1)
    return dq


def DQ2QT(dq, rot_as_q=False):
    real = dq[..., :4]
    imag = dq[..., 4:]
    real_norm = real.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    real, imag = real / real_norm, imag / real_norm

    w0, x0, y0, z0 = torch.unbind(real, -1)
    w1, x1, y1, z1 = torch.unbind(imag, -1)

    t = 2* torch.stack([- w1*x0 + x1*w0 - y1*z0 + z1*y0,
                        - w1*y0 + x1*z0 + y1*w0 - z1*x0,
                        - w1*z0 - x1*y0 + y1*x0 + z1*w0], dim=-1)
    R = torch.stack([1-2*y0**2-2*z0**2, 2*x0*y0-2*w0*z0, 2*x0*z0+2*w0*y0,
                     2*x0*y0+2*w0*z0, 1-2*x0**2-2*z0**2, 2*y0*z0-2*w0*x0,
                     2*x0*z0-2*w0*y0, 2*y0*z0+2*w0*x0, 1-2*x0**2-2*y0**2], dim=-1).reshape([*w0.shape, 3, 3])
    if rot_as_q:
        q = matrix_to_quaternion(R)
        return q, t
    else:
        return R, t
    

def DQBlending(q, t, weights, rot_as_q=True):
    '''
    Input:
        q: [..., k, 4]; t: [..., k, 3]; weights: [..., k]
    Output:
        q_: [..., 4]; t_: [..., 3]
    '''
    dq = QT2DQ(q=q, t=t)
    dq_avg = (dq * weights[..., None]).sum(dim=-2)
    q_, t_ = DQ2QT(dq_avg, rot_as_q=rot_as_q)
    return q_, t_


def interpolate(q0, t0, q1, t1, weight, rot_as_q=True):
    dq0 = QT2DQ(q=q0, t=t0)
    dq1 = QT2DQ(q=q1, t=t1)
    dq_avg = dq0 * weight + dq1 * (1 - weight)
    q, t = DQ2QT(dq=dq_avg, rot_as_q=rot_as_q)
    return q, t


def transformation_blending(transformations, weights):
    Rs, Ts = transformations[:, :3, :3], transformations[:, :3, 3]
    qs = matrix_to_quaternion(Rs)
    q, T = DQBlending(qs[None], Ts[None], weights)
    R = quaternion_to_matrix(q)
    transformation = torch.eye(4).to(transformations.device)[None].expand(weights.shape[0], 4, 4).clone()
    transformation[:, :3, :3] = R
    transformation[:, :3, 3] = T
    return transformation
