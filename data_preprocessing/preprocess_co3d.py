import torch
import torchvision
import numpy as np
import os
import tqdm
import math
from PIL import Image
from omegaconf import DictConfig
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2
from pytorch3d.implicitron.tools.config import expand_args_fields

# Set the paths to your dataset directories
CO3D_RAW_ROOT = "./CO3D/DOWNLOAD_FOLDER"  # Update this path
CO3D_OUT_ROOT = "./CO3D/OUTPUT_FOLDER"     # Update this path

assert CO3D_RAW_ROOT is not None, "Change CO3D_RAW_ROOT to where your raw CO3D data resides"
assert CO3D_OUT_ROOT is not None, "Change CO3D_OUT_ROOT to where you want to save the processed CO3D data"

def construct_depth_paths(rgb_image_path):
    """
    Construct the depth image path based on the RGB image path.
    """
    depth_image_path = rgb_image_path.replace('images', 'depths').replace('.jpg', '.jpg.geometric.png')
    return depth_image_path

def main(dataset_name, category):

    subset_name = "multisequence"

    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_root=CO3D_RAW_ROOT,
        dataset_JsonIndexDataset_args=DictConfig(
            {"remove_empty_masks": False, "load_point_clouds": True}
        ),
    ).get_dataset_map()

    created_dataset = dataset_map[dataset_name]
    sequence_names = [k for k in created_dataset.seq_annots.keys()]
    bkgd = 0.0  # black background

    out_folder_path = os.path.join(CO3D_OUT_ROOT, f"co3d_{category}_for_gs_depth_hyd", dataset_name)
    os.makedirs(out_folder_path, exist_ok=True)

    bad_sequences = []
    camera_Rs_all_sequences = {}
    camera_Ts_all_sequences = {}

    for sequence_name in tqdm.tqdm(sequence_names):
        folder_outname = os.path.join(out_folder_path, sequence_name)
        os.makedirs(folder_outname, exist_ok=True)

        frame_idx_gen = created_dataset.sequence_indices_in_order(sequence_name)
        frame_idxs = []
        focal_lengths_this_sequence = []
        rgb_full_this_sequence = []
        rgb_fg_this_sequence = []
        depth_full_this_sequence = []
        rgdb_full_this_sequence = []  # To store RGDB images
        fname_order = []

        # Preprocess cameras
        cameras_this_seq = read_seq_cameras(created_dataset, sequence_name)
        camera_Rs_all_sequences[sequence_name] = cameras_this_seq.R
        camera_Ts_all_sequences[sequence_name] = cameras_this_seq.T

        while True:
            try:
                frame_idx = next(frame_idx_gen)
                frame_idxs.append(frame_idx)
            except StopIteration:
                break

        # Process each frame
        for frame_idx in frame_idxs:
            frame = created_dataset[frame_idx]

            # RGB image processing
            rgb_image = torchvision.transforms.functional.pil_to_tensor(
                Image.open(frame.image_path)).float() / 255.0

            # Construct the corresponding depth path
            depth_image_path = construct_depth_paths(frame.image_path)

            # Load Depth image
            if os.path.exists(depth_image_path):
                depth_image = torchvision.transforms.functional.pil_to_tensor(
                    Image.open(depth_image_path)).float() / 255.0
            else:
                print(f"Warning: Depth image not found for {frame.image_path}")
                depth_image = None

            # ============= Foreground mask (for RGB) ================
            fg_probability = torch.zeros_like(rgb_image)[:1, ...]
            resized_image_mask_boundary_y = torch.where(frame.mask_crop > 0)[1].max() + 1
            resized_image_mask_boundary_x = torch.where(frame.mask_crop > 0)[2].max() + 1
            x0, y0, box_w, box_h = frame.crop_bbox_xywh
            resized_mask = torchvision.transforms.functional.resize(
                frame.fg_probability[:, :resized_image_mask_boundary_y, :resized_image_mask_boundary_x],
                (box_h, box_w),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            fg_probability[:, y0:y0+box_h, x0:x0+box_w] = resized_mask

            # Compute location of principal point for cropping
            principal_point_cropped = frame.camera.principal_point * 0.5 * frame.image_rgb.shape[1]
            scaling_factor = max(box_h, box_w) / 800
            principal_point_x = (frame.image_rgb.shape[2] * 0.5 - principal_point_cropped[0, 0]) * scaling_factor + x0
            principal_point_y = (frame.image_rgb.shape[1] * 0.5 - principal_point_cropped[0, 1]) * scaling_factor + y0
            max_half_side = get_max_box_side(
                frame.image_size_hw, principal_point_x, principal_point_y)

            # Crop and resize the RGB and depth images
            rgb = crop_image_at_non_integer_locations(rgb_image, max_half_side, principal_point_x, principal_point_y)
            if depth_image is not None:
                depth_image = crop_image_at_non_integer_locations(depth_image, max_half_side, principal_point_x, principal_point_y)

            # Resize RGB and depth to 128x128
            rgb_resized = torchvision.transforms.functional.resize(torchvision.transforms.functional.to_pil_image(rgb), 128)
            rgb_resized_tensor = torchvision.transforms.functional.pil_to_tensor(rgb_resized) / 255.0

            if depth_image is not None:
                depth_resized = torchvision.transforms.functional.resize(torchvision.transforms.functional.to_pil_image(depth_image), 128)
                depth_resized_tensor = torchvision.transforms.functional.pil_to_tensor(depth_resized) / 255.0

            # Resize and apply foreground mask to RGB
            fg_probability_resized = torchvision.transforms.functional.resize(fg_probability, (128, 128), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            rgb_fg = rgb_resized_tensor[:3, ...] * fg_probability_resized + bkgd * (1-fg_probability_resized)

            # Save processed RGB and depth data
            rgb_full_this_sequence.append(rgb_resized_tensor[:3, ...])
            rgb_fg_this_sequence.append(rgb_fg)
            if depth_image is not None:
                depth_full_this_sequence.append(depth_resized_tensor[:1, ...])  # Keep only 1 channel for depth

            # Stacking RGB and depth (RGBD)
            if depth_image is not None:
                rgdb_image = torch.cat((rgb_resized_tensor[:3, ...], depth_resized_tensor[:1, ...]), dim=0)  # Stack along channel dimension (3 + 1)
                rgdb_full_this_sequence.append(rgdb_image)

            fname_order.append(f"{frame_idx:05d}.png")

            # Transform focal lengths
            transformed_focal_lengths = frame.camera.focal_length * max(box_h, box_w) / (2 * max_half_side)
            focal_lengths_this_sequence.append(transformed_focal_lengths)

        # Save images and focal lengths
        focal_lengths_this_sequence = torch.stack(focal_lengths_this_sequence)
        np.save(os.path.join(folder_outname, "images_full.npy"), torch.stack(rgb_full_this_sequence).numpy())
        np.save(os.path.join(folder_outname, "images_fg.npy"), torch.stack(rgb_fg_this_sequence).numpy())
        if depth_full_this_sequence:
            np.save(os.path.join(folder_outname, "depth_images_full.npy"), torch.stack(depth_full_this_sequence).numpy())
        if rgdb_full_this_sequence:
            np.save(os.path.join(folder_outname, "rgdb_images_full.npy"), torch.stack(rgdb_full_this_sequence).numpy())
        np.save(os.path.join(folder_outname, "focal_lengths.npy"), focal_lengths_this_sequence.numpy())

        with open(os.path.join(folder_outname, "frame_order.txt"), "w+") as f:
            f.writelines([fname + "\n" for fname in fname_order])

    # Save camera data
    for dict_to_save, dict_name in zip([camera_Rs_all_sequences, camera_Ts_all_sequences],
                                       ["camera_Rs", "camera_Ts"]):
        np.savez(os.path.join(out_folder_path, f"{dict_name}.npz"),
                 **{k: v.detach().cpu().numpy() for k, v in dict_to_save.items()})

    return bad_sequences

def get_max_box_side(hw, principal_point_x, principal_point_y):
    max_x = hw[1]
    min_x = 0.0
    max_y = hw[0]
    min_y = 0.0
    max_half_w = min(principal_point_x - min_x, max_x - principal_point_x)
    max_half_h = min(principal_point_y - min_y, max_y - principal_point_y)
    return min(max_half_h, max_half_w)

def crop_image_at_non_integer_locations(img, max_half_side, principal_point_x, principal_point_y):
    max_pixel_number = math.floor(2 * max_half_side)
    half_pixel_side = 0.5 / max_pixel_number
    x_locations = torch.linspace(principal_point_x - max_half_side + half_pixel_side,
                                 principal_point_x + max_half_side - half_pixel_side,
                                 max_pixel_number)
    y_locations = torch.linspace(principal_point_y - max_half_side + half_pixel_side,
                                 principal_point_y + max_half_side - half_pixel_side,
                                 max_pixel_number)
    grid_locations = torch.stack(torch.meshgrid(x_locations, y_locations, indexing='ij'), dim=-1).transpose(0, 1)
    grid_locations[:, :, 1] = (grid_locations[:, :, 1] - img.shape[1] / 2) / (img.shape[1] / 2)
    grid_locations[:, :, 0] = (grid_locations[:, :, 0] - img.shape[2] / 2) / (img.shape[2] / 2)
    image_crop = torch.nn.functional.grid_sample(img.unsqueeze(0), grid_locations.unsqueeze(0), align_corners=True)
    return image_crop.squeeze(0)

def read_seq_cameras(dataset, sequence_name):
    frame_idx_gen = dataset.sequence_indices_in_order(sequence_name)
    frame_idxs = []
    while True:
        try:
            frame_idx = next(frame_idx_gen)
            frame_idxs.append(frame_idx)
        except StopIteration:
            break

    cameras_start = []
    for frame_idx in frame_idxs:
        cameras_start.append(dataset[frame_idx].camera)
    cameras_start = join_cameras_as_batch(cameras_start)
    cameras = cameras_start.clone()

    return cameras

if __name__ == "__main__":
    categories = ["car"]
    splits = ["train", "val", "test"]

    for category in categories:
        for split in splits:
            print(f"Processing category: {category}, split: {split}")
            bad_sequences = main(split, category)
            if bad_sequences:
                print(f"Warning! Bad sequences found in {split}: {bad_sequences}")
            else:
                print(f"Successfully processed {split} split for category {category}.")
