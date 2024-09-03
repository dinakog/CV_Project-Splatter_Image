import os
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import torchvision
import numpy as np

from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_factory import get_dataset
from utils.vis_utils import vis_image_preds

from huggingface_hub import hf_hub_download

# Configuration constants
# Change these values to customize the dataset creation process
NUM_VIEWS_PER_OBJECT = 5  # Number of views to process per object
DATASET_SPLIT = "train"  # Dataset split to use: "train", "val", or "test"

@torch.no_grad()
def create_training_dataset(model, dataloader, device, model_cfg, out_folder):
    """
    Create a training dataset with RGB images, depth maps, and pose information.
    
    Args:
    - model: The GaussianSplatPredictor model
    - dataloader: DataLoader for the dataset
    - device: torch.device for computation
    - model_cfg: Model configuration
    - out_folder: Output folder for the created dataset
    """
    for d_idx, data in enumerate(tqdm(dataloader, desc="Processing objects")):
        data = {k: v.to(device) for k, v in data.items()}
        example_id = dataloader.dataset.get_example_id(d_idx)
        example_folder = os.path.join(out_folder, example_id)
        
        # Create directories for RGB, depth, and pose data
        os.makedirs(os.path.join(example_folder, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(example_folder, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(example_folder, 'pose'), exist_ok=True)

        # Process each view of the object
        for r_idx in range(min(NUM_VIEWS_PER_OBJECT, data["gt_images"].shape[1])):
            # Prepare input data for the model
            rot_transform_quats = data["source_cv2wT_quat"][:, r_idx:r_idx+1, ...]
            focals_pixels_pred = data["focals_pixels"][:, r_idx:r_idx+1, :, :, :] if model_cfg.data.category in ["hydrants", "teddybears"] else None
            im = data["gt_images"][:, r_idx:r_idx+1, :, :, :]
            
            # Save RGB image
            rgb_path = os.path.join(example_folder, 'rgb', f'{r_idx:05d}.png')
            torchvision.utils.save_image(data["gt_images"][0, r_idx, ...], rgb_path)
            
            # Generate and save depth image
            reconstruction = model(im, data["view_to_world_transforms"][:, r_idx:r_idx+1, ...], rot_transform_quats, focals_pixels_pred)
            depth_folder = os.path.join(example_folder, 'depth')
            vis_image_preds({k: v[0].contiguous() for k, v in reconstruction.items()}, depth_folder, 'depth')
            
            # Rename the depth image to match the RGB image naming convention
            old_depth_path = os.path.join(depth_folder, 'depth.png')
            new_depth_path = os.path.join(depth_folder, f'{r_idx:05d}.png')
            if os.path.exists(old_depth_path):
                os.rename(old_depth_path, new_depth_path)
            
            # Save pose information
            pose_path = os.path.join(example_folder, 'pose', f'{r_idx:05d}.txt')
            pose = data["view_to_world_transforms"][0, r_idx, ...].cpu().numpy()
            np.savetxt(pose_path, pose, fmt='%.6f')

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download configuration and model files
    cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", filename="config_cars.yaml")
    model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", filename="model_cars.pth")
    
    # Load configuration
    training_cfg = OmegaConf.load(cfg_path)
    
    # Initialize and load the model
    model = GaussianSplatPredictor(training_cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    model.eval()
    print('Model loaded successfully!')

    # Create dataset and dataloader
    dataset = get_dataset(training_cfg, DATASET_SPLIT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Create the training dataset
    create_training_dataset(model, dataloader, device, training_cfg, args.out_folder)
    print(f"Dataset created in {args.out_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create training dataset with depth')
    parser.add_argument('--out_folder', type=str, default='train_dataset_with_depth', help='Output folder to save dataset')
    args = parser.parse_args()
    main(args)