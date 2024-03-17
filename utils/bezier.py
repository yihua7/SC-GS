import numpy as np


class BezierCurve:
    def __init__(self, points: np.ndarray) -> None:
        if points.ndim == 2:
            points = points[None]
        self.points = points  # N, T, D
        self.T = points.shape[1]
    
    def __call__(self, t: float):
        assert 0 <= t <= 1, f't: {t} out of range [0, 1]!'
        return self.interpolate(t, self.points)

    def interpolate(self, t, points):
        if points.shape[1] < 2:
            raise ValueError(f"points shape error: {points.shape}")
        elif points.shape[1] == 2:
            point0, point1 = points[:, 0], points[:, 1]
        else:
            point0 = self.interpolate(t, points[:, :-1])
            point1 = self.interpolate(t, points[:, 1:])
        return (1 - t) * point0 + t * point1


class PieceWiseLinear:
    def __init__(self, points: np.ndarray) -> None:
        if points.ndim == 2:
            points = points[None]
        self.points = points  # N, T, D
        self.T = points.shape[1]
    
    def __call__(self, t: float):
        assert 0 <= t <= 1, f't: {t} out of range [0, 1]!'
        return self.interpolate(t, self.points)

    def interpolate(self, t, points):
        if points.shape[1] < 2:
            raise ValueError(f"points shape error: {points.shape}")
        else:
            t_scaled = t * (self.T - 1)
            t_floor = min(self.T - 2, max(0, int(np.floor(t_scaled))))
            t_ceil = t_floor + 1
            point0, point1 = points[:, t_floor], points[:, t_ceil]
            return (t_ceil - t_scaled) * point0 + (t_scaled - t_floor) * point1
