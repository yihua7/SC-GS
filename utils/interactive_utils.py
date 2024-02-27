import numpy as np
import torch


class DeformKeypoints:
    def __init__(self) -> None:
        self.keypoints3d_list = []  # list of keypoints group
        self.keypoints_idx_list = [] # keypoints index
        self.keypoints3d_delta_list = []
        self.selective_keypoints_idx_list = []  # keypoints index
        self.idx2group = {}

        self.selective_rotation_keypoints_idx_list = []
        # self.rotation_idx2group = {}

    def get_kpt_idx(self,):
        return self.keypoints_idx_list
    
    def get_kpt(self,):
        return self.keypoints3d_list
    
    def get_kpt_delta(self,):
        return self.keypoints3d_delta_list
    
    def get_deformed_kpt_np(self, rate=1.):
        return np.array(self.keypoints3d_list) + np.array(self.keypoints3d_delta_list) * rate

    def add_kpts(self, keypoints_coord, keypoints_idx, expand=False):
        # keypoints3d: [N, 3], keypoints_idx: [N,], torch.tensor
        # self.selective_keypoints_idx_list.clear()
        selective_keypoints_idx_list = [] if not expand else self.selective_keypoints_idx_list
        for idx in range(len(keypoints_idx)):
            if not self.contain_kpt(keypoints_idx[idx].item()):
                selective_keypoints_idx_list.append(len(self.keypoints_idx_list))
                self.keypoints_idx_list.append(keypoints_idx[idx].item())
                self.keypoints3d_list.append(keypoints_coord[idx].cpu().numpy())            
                self.keypoints3d_delta_list.append(np.zeros_like(self.keypoints3d_list[-1]))

        for kpt_idx in keypoints_idx:
            self.idx2group[kpt_idx.item()] = selective_keypoints_idx_list

        self.selective_keypoints_idx_list = selective_keypoints_idx_list

    def contain_kpt(self, idx):
        # idx: int
        if idx in self.keypoints_idx_list:
            return True
        else:
            return False
        
    def select_kpt(self, idx):
        # idx: int
        # output: idx list of this group
        if idx in self.keypoints_idx_list:
            self.selective_keypoints_idx_list = self.idx2group[idx]

    def select_rotation_kpt(self, idx):
        if idx in self.keypoints_idx_list:
            self.selective_rotation_keypoints_idx_list = self.idx2group[idx]

    def get_rotation_center(self,):
        selected_rotation_points = self.get_deformed_kpt_np()[self.selective_rotation_keypoints_idx_list]
        return selected_rotation_points.mean(axis=0)
    
    def get_selective_center(self,):
        selected_points = self.get_deformed_kpt_np()[self.selective_keypoints_idx_list]
        return selected_points.mean(axis=0)

    def delete_kpt(self, idx):
        pass

    def delete_batch_ktps(self, batch_idx):
        pass

    def update_delta(self, delta):
        # delta: [3,], np.array
        for idx in self.selective_keypoints_idx_list:
            self.keypoints3d_delta_list[idx] += delta

    def set_delta(self, delta):
        # delta: [N, 3], np.array
        for id, idx in enumerate(self.selective_keypoints_idx_list):
            self.keypoints3d_delta_list[idx] = delta[id]


    def set_rotation_delta(self, rot_mat):
        kpts3d = self.get_deformed_kpt_np()[self.selective_keypoints_idx_list]
        kpts3d_mean = self.get_rotation_center()
        kpts3d = (kpts3d - kpts3d_mean) @ rot_mat.T + kpts3d_mean
        delta = kpts3d - np.array(self.keypoints3d_list)[self.selective_keypoints_idx_list]
        for id, idx in enumerate(self.selective_keypoints_idx_list):
            self.keypoints3d_delta_list[idx] = delta[id]
