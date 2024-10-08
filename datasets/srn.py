import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_readers import readCamerasFromTxt, readDepthMinMax
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

from .shared_dataset import SharedDataset

SHAPENET_DATASET_ROOT = "C:\\Users\dinak\PyCharmsProjects\\try\CV_Project-Splatter_Image\SRN_Full" # Change this to your data directory
assert SHAPENET_DATASET_ROOT is not None, "Update the location of the SRN Shapenet Dataset"

class SRNDataset(SharedDataset):
    def __init__(self, cfg,
                 dataset_name="train"):
        super().__init__()
        self.cfg = cfg
        self.rgbdnn_data = False

        self.dataset_name = dataset_name
        if dataset_name == "vis":
            self.dataset_name = "test"

        self.base_path = os.path.join(SHAPENET_DATASET_ROOT, "srn_{}/{}_{}".format(cfg.data.category,
                                                                                   cfg.data.category,
                                                                                   self.dataset_name))

        is_chair = "chair" in cfg.data.category
        if is_chair and dataset_name == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        data_len = len(self.intrins)

        self.depths_min_max = sorted(
            glob.glob(os.path.join(self.base_path, "*", "output.txt"))
        )

        if cfg.data.subset != -1:
            subset_size = int(cfg.data.subset * data_len)
            self.intrins = self.intrins[:subset_size]
            self.depths_min_max = self.depths_min_max[:subset_size]

        print(len(self.intrins))

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj

        # in deterministic version the number of testing images
        # and number of training images are the same
        if self.cfg.data.input_images == 1:
            self.test_input_idxs = [49]
        elif self.cfg.data.input_images == 2:
            self.test_input_idxs = [64, 128]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.intrins)

    def load_example_id(self, example_id, intrin_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))
        rgbd_paths = sorted(glob.glob(os.path.join(dir_path, "rgbd", "*")))
        rgbdnn_paths = sorted(glob.glob(os.path.join(dir_path, "rgbdnn", "*")))
        depth_min_max_path = os.path.join(dir_path, "output.txt")
        assert len(rgb_paths) == len(pose_paths)
        if len(rgbdnn_paths) > 0:
            assert len(rgbdnn_paths) == len(pose_paths)
            self.rgbdnn_data = True
        else:
            self.rgbdnn_data = False

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_rgbdnns = {} # Store RGBD images
            self.all_rgbds = {} # Store RGBD images
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}
            self.all_depth_min_max = {}

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_rgbds[example_id] = []
            self.all_rgbdnns[example_id] = []  # initialize RGBD images
            self.all_world_view_transforms[example_id] = []
            self.all_full_proj_transforms[example_id] = []
            self.all_camera_centers[example_id] = []
            self.all_view_to_world_transforms[example_id] = []
            self.all_depth_min_max[example_id] = []

            cam_infos = readCamerasFromTxt(rgb_paths, rgbd_paths, pose_paths, [i for i in range(len(rgb_paths))])

            if self.rgbdnn_data is not None and self.dataset_name == "train":
                self.all_depth_min_max[example_id] = readDepthMinMax(depth_min_max_path)

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T

                # Load and process RGB image
                self.all_rgbs[example_id].append(PILtoTorch(cam_info.image, 
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])

                # Load and process Depth image
                if cam_info.rgbd is not None:
                    self.all_rgbdnns[example_id].append(PILtoTorch(cam_info.rgbd,
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:4, :, :])
                    self.all_rgbds[example_id].append(PILtoTorch(cam_info.rgbd,
                                                                   (self.cfg.data.training_resolution,
                                                                    self.cfg.data.training_resolution)).clamp(0.0, 1.0)[
                                                        :4, :, :])

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.all_world_view_transforms[example_id].append(world_view_transform)
                self.all_view_to_world_transforms[example_id].append(view_world_transform)
                self.all_full_proj_transforms[example_id].append(full_proj_transform)
                self.all_camera_centers[example_id].append(camera_center)
            
            self.all_world_view_transforms[example_id] = torch.stack(self.all_world_view_transforms[example_id])
            self.all_view_to_world_transforms[example_id] = torch.stack(self.all_view_to_world_transforms[example_id])
            self.all_full_proj_transforms[example_id] = torch.stack(self.all_full_proj_transforms[example_id])
            self.all_camera_centers[example_id] = torch.stack(self.all_camera_centers[example_id])
            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])
            if self.rgbdnn_data:
                self.all_rgbdnns[example_id] = torch.stack(self.all_rgbdnns[example_id])  # stack depth images
                self.all_rgbds[example_id] = torch.stack(self.all_rgbds[example_id])  # stack depth images
                self.all_depth_min_max[example_id] = torch.tensor(self.all_depth_min_max[example_id])

    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))

        self.load_example_id(example_id, intrin_path)
        if self.dataset_name == "train":
            frame_idxs = torch.randperm(
                    len(self.all_rgbs[example_id])
                    )[:self.imgs_per_obj]

            frame_idxs = torch.cat([frame_idxs[:self.cfg.data.input_images], frame_idxs], dim=0)

        else:
            input_idxs = self.test_input_idxs
            
            frame_idxs = torch.cat([torch.tensor(input_idxs), 
                                    torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0)

        if self.rgbdnn_data:
            images_and_camera_poses = {
                "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
                "gt_rgbds": self.all_rgbds[example_id][frame_idxs].clone(),
                "gt_rgbdnns": self.all_rgbdnns[example_id][frame_idxs].clone(),  # clone depth images
                "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
                "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
                "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
                "camera_centers": self.all_camera_centers[example_id][frame_idxs]
            }
            if self.dataset_name == "train" or self.dataset_name == "val":
                images_and_camera_poses["depths_min_max"] = self.all_depth_min_max[example_id][frame_idxs]
        else:
            images_and_camera_poses = {
                "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
                "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
                "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
                "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
                "camera_centers": self.all_camera_centers[example_id][frame_idxs]
            }

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses
