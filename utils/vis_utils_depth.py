import os
import matplotlib.pyplot as plt
import torch
from utils.sh_utils import eval_sh

def vis_image_preds_depth(image_preds: dict, folder_out: str, save_m_out=None):
    """
    Visualises network's image predictions including depth.
    Args:
        image_preds: a dictionary of xyz, opacity, scaling, rotation, features_dc and features_rest
    """
    image_preds_reshaped = {}
    ray_dirs = (image_preds["xyz"].detach().cpu() / torch.norm(image_preds["xyz"].detach().cpu(), dim=-1, keepdim=True)).reshape(128, 128, 3)

    for k, v in image_preds.items():
        image_preds_reshaped[k] = v
        if k == "xyz":
            image_preds_reshaped["depth"] = image_preds_reshaped[k][..., 2]

            image_preds_reshaped["depth"] = (image_preds_reshaped["depth"] - torch.min(image_preds_reshaped["depth"])) / (
                torch.max(image_preds_reshaped["depth"]) - torch.min(image_preds_reshaped["depth"])
            )
            image_preds_reshaped["depth"] = image_preds_reshaped["depth"].reshape(128, 128).detach().cpu()

            image_preds_reshaped[k] = (image_preds_reshaped[k] - torch.min(image_preds_reshaped[k])) / (
                torch.max(image_preds_reshaped[k]) - torch.min(image_preds_reshaped[k])
            )

        if k == "scaling":
            image_preds_reshaped["scaling"] = (image_preds_reshaped["scaling"] - torch.min(image_preds_reshaped["scaling"])) / (
                torch.max(image_preds_reshaped["scaling"]) - torch.min(image_preds_reshaped["scaling"])
            )
        if k != "features_rest":
            image_preds_reshaped[k] = image_preds_reshaped[k].reshape(128, 128, -1).detach().cpu()
        else:
            image_preds_reshaped[k] = image_preds_reshaped[k].reshape(128, 128, 3, 3).detach().cpu().permute(0, 1, 3, 2)
        if k == "opacity":
            image_preds_reshaped[k] = image_preds_reshaped[k].expand(128, 128, 3) 

    colours = torch.cat([image_preds_reshaped["features_dc"].unsqueeze(-1), image_preds_reshaped["features_rest"]], dim=-1)
    colours = eval_sh(1, colours, ray_dirs)

    if save_m_out is None or save_m_out == 'all' or save_m_out == 'rgb':
        plt.imsave(os.path.join(folder_out, "rgb.png"), colours.numpy())
    if save_m_out is None or save_m_out == 'all' or save_m_out == 'opacity':
        plt.imsave(os.path.join(folder_out, "opacity.png"), image_preds_reshaped["opacity"].numpy())
    if save_m_out is None or save_m_out == 'all' or save_m_out == 'xyz':
        plt.imsave(os.path.join(folder_out, "xyz.png"),
                   (image_preds_reshaped["xyz"] * image_preds_reshaped["opacity"] + 1 - image_preds_reshaped["opacity"]).numpy())
    if save_m_out is None or save_m_out == 'all' or save_m_out == 'scaling':
        plt.imsave(os.path.join(folder_out, "scaling.png"),
                   (image_preds_reshaped["scaling"] * image_preds_reshaped["opacity"] + 1 - image_preds_reshaped["opacity"]).numpy())
    if save_m_out is None or save_m_out == 'all' or save_m_out == 'depth':
        plt.imsave(os.path.join(folder_out, "depth.png"), image_preds_reshaped["depth"].numpy(), cmap='viridis')