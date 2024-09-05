import argparse
import json
import os
import sys
import tqdm
from omegaconf import OmegaConf

from huggingface_hub import hf_hub_download

import lpips as lpips_lib

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from gaussian_renderer import render_predicted
from scene.gaussian_predictor_depth import DepthGaussianSplatPredictor
from datasets.srn_depth import SRNDepthDataset
from utils.loss_utils import ssim as ssim_fn
from utils.vis_utils_depth import vis_image_preds_depth

class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
        lpips = self.lpips_net(image[:3].unsqueeze(0) * 2 - 1, target[:3].unsqueeze(0) * 2 - 1).item()
        psnr_rgb = -10 * torch.log10(torch.mean((image[:3] - target[:3]) ** 2)).item()
        psnr_depth = -10 * torch.log10(torch.mean((image[3:] - target[3:]) ** 2)).item()
        ssim = ssim_fn(image[:3], target[:3]).item()
        return psnr_rgb, psnr_depth, ssim, lpips

@torch.no_grad()
def evaluate_dataset(model, dataloader, device, model_cfg, save_vis=0, out_folder=None):
    if save_vis > 0:
        os.makedirs(out_folder, exist_ok=True)

    with open("scores.txt", "w+") as f:
        f.write("")

    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color + [0], dtype=torch.float32, device="cuda")  # Add 0 for depth background

    metricator = Metricator(device)

    psnr_all_examples_novel_rgb = []
    psnr_all_examples_novel_depth = []
    ssim_all_examples_novel = []
    lpips_all_examples_novel = []

    psnr_all_examples_cond_rgb = []
    psnr_all_examples_cond_depth = []
    ssim_all_examples_cond = []
    lpips_all_examples_cond = []

    for d_idx, data in enumerate(tqdm.tqdm(dataloader)):
        psnr_all_renders_novel_rgb = []
        psnr_all_renders_novel_depth = []
        ssim_all_renders_novel = []
        lpips_all_renders_novel = []
        psnr_all_renders_cond_rgb = []
        psnr_all_renders_cond_depth = []
        ssim_all_renders_cond = []
        lpips_all_renders_cond = []

        data = {k: v.to(device) for k, v in data.items()}

        rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]

        if model_cfg.data.category == "hydrants" or model_cfg.data.category == "teddybears":
            focals_pixels_pred = data["focals_pixels"][:, :model_cfg.data.input_images, ...]
        else:
            focals_pixels_pred = None

        input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]

        example_id = dataloader.dataset.get_example_id(d_idx)

        if d_idx < save_vis:
            out_example_gt = os.path.join(out_folder, "{}_".format(d_idx) + example_id + "_gt")
            out_example = os.path.join(out_folder, "{}_".format(d_idx) + example_id)
            os.makedirs(out_example_gt, exist_ok=True)
            os.makedirs(out_example, exist_ok=True)

        reconstruction = model(input_images,
                               data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                               rot_transform_quats,
                               focals_pixels_pred)

        for r_idx in range(data["gt_images"].shape[1]):
            if "focals_pixels" in data.keys():
                focals_pixels_render = data["focals_pixels"][0, r_idx]
            else:
                focals_pixels_render = None

            image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
                                     data["world_view_transforms"][0, r_idx],
                                     data["full_proj_transforms"][0, r_idx], 
                                     data["camera_centers"][0, r_idx],
                                     background,
                                     model_cfg,
                                     focals_pixels=focals_pixels_render)["render"]

            if d_idx < save_vis:
                torchvision.utils.save_image(image, os.path.join(out_example, '{0:05d}'.format(r_idx) + ".png"))
                torchvision.utils.save_image(data["gt_images"][0, r_idx, ...], os.path.join(out_example_gt, '{0:05d}'.format(r_idx) + ".png"))

            if not torch.all(data["gt_images"][0, r_idx, ...] == 0):
                psnr_rgb, psnr_depth, ssim, lpips = metricator.compute_metrics(image, data["gt_images"][0, r_idx, ...])
                if r_idx < model_cfg.data.input_images:
                    psnr_all_renders_cond_rgb.append(psnr_rgb)
                    psnr_all_renders_cond_depth.append(psnr_depth)
                    ssim_all_renders_cond.append(ssim)
                    lpips_all_renders_cond.append(lpips)
                else:
                    psnr_all_renders_novel_rgb.append(psnr_rgb)
                    psnr_all_renders_novel_depth.append(psnr_depth)
                    ssim_all_renders_novel.append(ssim)
                    lpips_all_renders_novel.append(lpips)

        psnr_all_examples_cond_rgb.append(sum(psnr_all_renders_cond_rgb) / len(psnr_all_renders_cond_rgb))
        psnr_all_examples_cond_depth.append(sum(psnr_all_renders_cond_depth) / len(psnr_all_renders_cond_depth))
        ssim_all_examples_cond.append(sum(ssim_all_renders_cond) / len(ssim_all_renders_cond))
        lpips_all_examples_cond.append(sum(lpips_all_renders_cond) / len(lpips_all_renders_cond))

        psnr_all_examples_novel_rgb.append(sum(psnr_all_renders_novel_rgb) / len(psnr_all_renders_novel_rgb))
        psnr_all_examples_novel_depth.append(sum(psnr_all_renders_novel_depth) / len(psnr_all_renders_novel_depth))
        ssim_all_examples_novel.append(sum(ssim_all_renders_novel) / len(ssim_all_renders_novel))
        lpips_all_examples_novel.append(sum(lpips_all_renders_novel) / len(lpips_all_renders_novel))

        with open("scores.txt", "a+") as f:
            f.write("{}_".format(d_idx) + example_id + \
                    " " + str(psnr_all_examples_novel_rgb[-1]) + \
                    " " + str(psnr_all_examples_novel_depth[-1]) + \
                    " " + str(ssim_all_examples_novel[-1]) + \
                    " " + str(lpips_all_examples_novel[-1]) + "\n")

    scores = {
        "PSNR_cond_rgb": sum(psnr_all_examples_cond_rgb) / len(psnr_all_examples_cond_rgb),
        "PSNR_cond_depth": sum(psnr_all_examples_cond_depth) / len(psnr_all_examples_cond_depth),
        "SSIM_cond": sum(ssim_all_examples_cond) / len(ssim_all_examples_cond),
        "LPIPS_cond": sum(lpips_all_examples_cond) / len(lpips_all_examples_cond),
        "PSNR_novel_rgb": sum(psnr_all_examples_novel_rgb) / len(psnr_all_examples_novel_rgb),
        "PSNR_novel_depth": sum(psnr_all_examples_novel_depth) / len(psnr_all_examples_novel_depth),
        "SSIM_novel": sum(ssim_all_examples_novel) / len(ssim_all_examples_novel),
        "LPIPS_novel": sum(lpips_all_examples_novel) / len(lpips_all_examples_novel)
    }

    return scores

@torch.no_grad()
def main(dataset_name, experiment_path, device_idx, split='test', save_vis=0, out_folder=None):
    device = torch.device("cuda:{}".format(device_idx))
    torch.cuda.set_device(device)

    if args.experiment_path is None:
        cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                 filename="config_{}.yaml".format(dataset_name))
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                            filename="model_{}.pth".format(dataset_name))
    else:
        cfg_path = os.path.join(experiment_path, ".hydra", "config.yaml")
        model_path = os.path.join(experiment_path, "model_latest.pth")
    
    training_cfg = OmegaConf.load(cfg_path)

    if args.experiment_path is not None:
        assert training_cfg.data.category == dataset_name, "Model-dataset mismatch"

    model = DepthGaussianSplatPredictor(training_cfg)
    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model = model.to(device)
    model.eval()
    print('Loaded model!')

    dataset = SRNDepthDataset(training_cfg, split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            persistent_workers=True, pin_memory=True, num_workers=1)
    
    print("Evaluating model")
    scores = evaluate_dataset(model, dataloader, device, training_cfg, save_vis=save_vis, out_folder=out_folder)

    print(scores)
    return scores

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate depth model')
    parser.add_argument('dataset_name', type=str, help='Dataset to evaluate on', 
                        choices=['cars', 'chairs', 'hydrants', 'teddybears'])
    parser.add_argument('--experiment_path', type=str, default=None, help='Path to the parent folder of the model')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'vis'],
                        help='Split to evaluate on (default: test)')
    parser.add_argument('--out_folder', type=str, default='out', help='Output folder to save renders (default: out)')
    parser.add_argument('--save_vis', type=int, default=0, help='Number of examples for which to save renders (default: 0)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    scores = main(args.dataset_name, args.experiment_path, 0, split=args.split, save_vis=args.save_vis, out_folder=args.out_folder)
    
    if args.split != "vis":
        if args.experiment_path is not None:
            score_out_path = os.path.join(args.experiment_path, 
                                   "{}_scores.json".format(args.split))
        else:
            score_out_path = "{}_{}_scores.json".format(args.dataset_name, args.split)
        with open(score_out_path, "w+") as f:
            json.dump(scores, f, indent=4)