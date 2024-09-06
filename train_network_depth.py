import glob
import hydra
import os
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader

from lightning.fabric import Fabric

from ema_pytorch import EMA
from omegaconf import DictConfig, OmegaConf

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, l2_loss
import lpips as lpips_lib

from eval_depth import evaluate_dataset
from gaussian_renderer import render_predicted
from scene.gaussian_predictor_depth import DepthGaussianSplatPredictor
from datasets.srn_depth import SRNDepthDataset

@hydra.main(version_base=None, config_path='configs', config_name="depth_config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    if cfg.general.mixed_precision:
        fabric = Fabric(accelerator="cuda", devices=cfg.general.num_devices, strategy="dp",
                        precision="16-mixed")
    else:
        fabric = Fabric(accelerator="cuda", devices=cfg.general.num_devices, strategy="dp")
    fabric.launch()

    if fabric.is_global_zero:
        vis_dir = os.getcwd()

        dict_cfg = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

        if os.path.isdir(os.path.join(vis_dir, "wandb")):
            run_name_path = glob.glob(os.path.join(vis_dir, "wandb", "latest-run", "run-*"))[0]
            print("Got run name path {}".format(run_name_path))
            run_id = os.path.basename(run_name_path).split("run-")[1].split(".wandb")[0]
            print("Resuming run with id {}".format(run_id))
            wandb_run = wandb.init(project=cfg.wandb.project, resume=True,
                            id = run_id, config=dict_cfg)
        else:
            wandb_run = wandb.init(project=cfg.wandb.project, reinit=True,
                            config=dict_cfg)

    first_iter = 0
    device = safe_state(cfg)

    gaussian_predictor = DepthGaussianSplatPredictor(cfg)
    gaussian_predictor = gaussian_predictor.to(memory_format=torch.channels_last)

    l = []
    if cfg.model.network_with_offset:
        l.append({'params': gaussian_predictor.network_with_offset.parameters(), 
         'lr': cfg.opt.base_lr})
    if cfg.model.network_without_offset:
        l.append({'params': gaussian_predictor.network_wo_offset.parameters(), 
         'lr': cfg.opt.base_lr})
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, 
                                 betas=cfg.opt.betas)

    # Resuming training
    if fabric.is_global_zero:
        if os.path.isfile(os.path.join(vis_dir, "model_latest.pth")):
            print('Loading an existing model from ', os.path.join(vis_dir, "model_latest.pth"))
            checkpoint = torch.load(os.path.join(vis_dir, "model_latest.pth"),
                                    map_location=device) 
            try:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                                strict=False)
                print("Warning, model mismatch - was this expected?")
            first_iter = checkpoint["iteration"]
            best_PSNR = checkpoint["best_PSNR"] 
            print('Loaded model')
        # Resuming from checkpoint
        elif cfg.opt.pretrained_ckpt is not None:
            pretrained_ckpt_dir = os.path.join(cfg.opt.pretrained_ckpt, "model_latest.pth")
            checkpoint = torch.load(pretrained_ckpt_dir,
                                    map_location=device) 
            try:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                gaussian_predictor.load_state_dict(checkpoint["model_state_dict"],
                                                strict=False)
            best_PSNR = checkpoint["best_PSNR"] 
            print('Loaded model from a pretrained checkpoint')
        else:
            best_PSNR = 0.0

    if cfg.opt.ema.use and fabric.is_global_zero:
        ema = EMA(gaussian_predictor, 
                  beta=cfg.opt.ema.beta,
                  update_every=cfg.opt.ema.update_every,
                  update_after_step=cfg.opt.ema.update_after_step)
        ema = fabric.to_device(ema)

    if cfg.opt.loss == "l2":
        loss_fn = l2_loss
    elif cfg.opt.loss == "l1":
        loss_fn = l1_loss

    if cfg.opt.lambda_lpips != 0:
        lpips_fn = fabric.to_device(lpips_lib.LPIPS(net='vgg'))
    lambda_lpips = cfg.opt.lambda_lpips
    lambda_l12 = 1.0 - lambda_lpips

    bg_color = [1, 1, 1, 0] if cfg.data.white_background else [0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    background = fabric.to_device(background)

    dataset = SRNDepthDataset(cfg, "train")
    dataloader = DataLoader(dataset, 
                            batch_size=cfg.opt.batch_size,
                            shuffle=True,
                            num_workers=0,
                            persistent_workers=False)

    # distribute model and training dataset
    gaussian_predictor, optimizer = fabric.setup(
        gaussian_predictor, optimizer
    )
    dataloader = fabric.setup_dataloaders(dataloader)
    
    gaussian_predictor.train()

    print("Beginning training")
    first_iter += 1
    iteration = first_iter

    for num_epoch in range((cfg.opt.iterations + 1 - first_iter)// len(dataloader) + 1):
        for data in dataloader:
            iteration += 1

            print("starting iteration {} on process {}".format(iteration, fabric.global_rank))

            # =============== Prepare input ================
            input_images = data["gt_images"][:, :cfg.data.input_images, ...]

            gaussian_splats = gaussian_predictor(input_images,
                                                data["view_to_world_transforms"][:, :cfg.data.input_images, ...])
            
            print("Gaussian splats keys:", gaussian_splats.keys())
            print("Gaussian splats shapes:")
            for key, value in gaussian_splats.items():
                print(f"{key}: {value.shape}")
                
            # Render
            l12_loss_sum = 0.0
            lpips_loss_sum = 0.0
            rendered_images = []
            gt_images = []
            for b_idx in range(data["gt_images"].shape[0]):
                gaussian_splat_batch = {k: v[b_idx].contiguous() for k, v in gaussian_splats.items()}
                for r_idx in range(cfg.data.input_images, data["gt_images"].shape[1]):
                    image = render_predicted(gaussian_splat_batch, 
                                            data["world_view_transforms"][b_idx, r_idx],
                                            data["full_proj_transforms"][b_idx, r_idx],
                                            data["camera_centers"][b_idx, r_idx],
                                            background,
                                            cfg)["render"]
                    # Ensure the rendered image is the correct size
                    if image.shape[-2:] != (64, 64):
                        print(f"Resizing image from {image.shape[-2:]} to (64, 64)")
                        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
                        
                    rendered_images.append(image)
                    gt_image = data["gt_images"][b_idx, r_idx]
                    gt_images.append(gt_image)
                    
            
            # After stacking rendered_images and gt_images
            rendered_images = torch.stack(rendered_images, dim=0)
            gt_images = torch.stack(gt_images, dim=0)

            print(f"Rendered images shape: {rendered_images.shape}")
            print(f"GT images shape: {gt_images.shape}")

            print(f"Rendered RGB min/max: {rendered_images[:, :3].min().item()}, {rendered_images[:, :3].max().item()}")
            print(f"GT RGB min/max: {gt_images[:, :3].min().item()}, {gt_images[:, :3].max().item()}")

            if rendered_images.shape[1] > 3:
                print(f"Rendered depth min/max: {rendered_images[:, 3:].min().item()}, {rendered_images[:, 3:].max().item()}")
            else:
                print("Rendered images do not contain depth information")

            print(f"GT depth min/max: {gt_images[:, 3:].min().item()}, {gt_images[:, 3:].max().item()}")

            # Check for NaN values
            if torch.isnan(rendered_images).any():
                print("NaN detected in rendered images")
                nan_indices = torch.nonzero(torch.isnan(rendered_images))
                print(f"NaN indices in rendered images: {nan_indices}")
                
            if torch.isnan(rendered_images).any() or torch.isnan(gt_images).any():
                print("NaN detected in rendered or GT images")
                print(f"Rendered images NaN locations: {torch.isnan(rendered_images).nonzero()}")
                print(f"GT images NaN locations: {torch.isnan(gt_images).nonzero()}")
            if torch.isnan(gt_images).any():
                print("NaN detected in ground truth images")
                nan_indices = torch.nonzero(torch.isnan(gt_images))
                print(f"NaN indices in ground truth images: {nan_indices}")

            # Loss computation
            l12_loss_sum = loss_fn(rendered_images[:, :3], gt_images[:, :3])  # RGB loss
            depth_loss = loss_fn(rendered_images[:, 3:], gt_images[:, 3:])  # Depth loss
            
            print(f"l12_loss_sum: {l12_loss_sum.item()}")
            print(f"depth_loss: {depth_loss.item()}")

            if cfg.opt.lambda_lpips != 0:
                lpips_loss_sum = torch.mean(
                    lpips_fn(rendered_images[:, :3] * 2 - 1, gt_images[:, :3] * 2 - 1),
                )
            else:
                lpips_loss_sum = torch.tensor(0.0, device=device)  # Set to 0 if not used

            print(f"lpips_loss_sum: {lpips_loss_sum.item()}")

            total_loss = l12_loss_sum * lambda_l12 + lpips_loss_sum * lambda_lpips + depth_loss

            print(f"total_loss: {total_loss.item()}")
            
            if torch.isnan(total_loss):
                print("NaN detected in total loss")
                print(f"l12_loss_sum: {l12_loss_sum}")
                print(f"depth_loss: {depth_loss}")
                print(f"lpips_loss_sum: {lpips_loss_sum}")
                
            assert not torch.isnan(total_loss), "Found NaN loss!"
            
            fabric.backward(total_loss)

            # ============ Optimization ===============
            optimizer.step()
            optimizer.zero_grad()

            if cfg.opt.ema.use and fabric.is_global_zero:
                ema.update()

            # ========= Additional Logging ===========
            if fabric.is_global_zero:
                wandb.log({
                    "training_loss": total_loss.item(),
                    "l12_loss": l12_loss_sum.item(),
                    "lpips_loss": lpips_loss_sum.item(),
                    "depth_loss": depth_loss.item(),
                }, step=iteration)

            # ======= Existing WandB Logging Section ========
            if (iteration % cfg.logging.loss_log == 0 and fabric.is_global_zero):
                wandb.log({"training_loss": np.log10(total_loss.item() + 1e-8)}, step=iteration)
                wandb.log({"training_l12_loss": np.log10(l12_loss_sum.item() + 1e-8)}, step=iteration)
                wandb.log({"training_depth_loss": np.log10(depth_loss.item() + 1e-8)}, step=iteration)
                if cfg.opt.lambda_lpips != 0:
                    wandb.log({"training_lpips_loss": np.log10(lpips_loss_sum.item() + 1e-8)}, step=iteration)

            print(f"finished iteration {iteration} on process {fabric.global_rank}")

    wandb_run.finish()

if __name__ == "__main__":
    main()
