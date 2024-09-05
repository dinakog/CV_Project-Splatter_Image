import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import silu
from einops import rearrange, repeat
from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from utils.graphics_utils import fov2focal
from .gaussian_predictor import GaussianSplatPredictor, SingleImageSongUNetPredictor, SongUNet


class DepthSingleImageSongUNetPredictor(SingleImageSongUNetPredictor):
    def __init__(self, cfg, out_channels, bias, scale):
        super(SingleImageSongUNetPredictor, self).__init__()
        self.out_channels = out_channels
        self.cfg = cfg
        if cfg.cam_embd.embedding is None:
            in_channels = 4 if cfg.data.use_depth else 3
            emb_dim_in = 0
        else:
            in_channels = 4 if cfg.data.use_depth else 3
            emb_dim_in = 6 * cfg.cam_embd.dimension

        self.encoder = SongUNet(cfg.data.training_resolution, 
                                in_channels, 
                                sum(out_channels),
                                model_channels=cfg.model.base_dim,
                                num_blocks=cfg.model.num_blocks,
                                emb_dim_in=emb_dim_in,
                                channel_mult_noise=0,
                                attn_resolutions=cfg.model.attention_resolutions)
        self.out = nn.Conv2d(in_channels=sum(out_channels), 
                             out_channels=sum(out_channels),
                             kernel_size=1)

        start_channels = 0
        for out_channel, b, s in zip(out_channels, bias, scale):
            nn.init.xavier_uniform_(self.out.weight[start_channels:start_channels+out_channel, :, :, :], s)
            nn.init.constant_(self.out.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

    def forward(self, x, film_camera_emb=None, N_views_xa=1):
        print(f"Input shape: {x.shape}")
        print(f"Input min/max: {x.min().item()}, {x.max().item()}")
        
        x = self.encoder(x, 
                         film_camera_emb=film_camera_emb,
                         N_views_xa=N_views_xa)
        
        print(f"Encoder output shape: {x.shape}")
        print(f"Encoder output min/max: {x.min().item()}, {x.max().item()}")
        
        output = self.out(x)
        
        print(f"Final output shape: {output.shape}")
        print(f"Final output min/max: {output.min().item()}, {output.max().item()}")
        
        return output


class DepthGaussianSplatPredictor(GaussianSplatPredictor):
    def __init__(self, cfg):
        super(DepthGaussianSplatPredictor, self).__init__(cfg)
        self.cfg = cfg
        assert cfg.model.network_with_offset or cfg.model.network_without_offset, \
            "Need at least one network"

        if cfg.model.network_with_offset:
            split_dimensions, scale_inits, bias_inits = self.get_splits_and_inits(True, cfg)
            self.network_with_offset = DepthSingleImageSongUNetPredictor(cfg, 
                                        split_dimensions,
                                        scale=scale_inits,
                                        bias=bias_inits)
            assert not cfg.model.network_without_offset, "Can only have one network"
        if cfg.model.network_without_offset:
            split_dimensions, scale_inits, bias_inits = self.get_splits_and_inits(False, cfg)
            self.network_wo_offset = DepthSingleImageSongUNetPredictor(cfg, 
                                        split_dimensions,
                                        scale=scale_inits,
                                        bias=bias_inits)
            assert not cfg.model.network_with_offset, "Can only have one network"

    def forward(self, x, 
                source_cameras_view_to_world, 
                source_cv2wT_quat=None,
                focals_pixels=None,
                activate_output=True):

        B = x.shape[0]
        N_views = x.shape[1]
        
        # Cross-view attention
        if self.cfg.model.cross_view_attention:
            N_views_xa = N_views
        else:
            N_views_xa = 1

        # Camera embedding
        if self.cfg.cam_embd.embedding is not None:
            cam_embedding = self.get_camera_embeddings(source_cameras_view_to_world)
            assert self.cfg.cam_embd.method == "film"
            film_camera_emb = cam_embedding.reshape(B * N_views, cam_embedding.shape[2])
        else:
            film_camera_emb = None

        # Dataset specifics
        if self.cfg.data.category in ["hydrants", "teddybears"]:
            assert focals_pixels is not None
            focals_pixels = focals_pixels.reshape(B * N_views, *focals_pixels.shape[2:])
        else:
            assert focals_pixels is None, "Unexpected argument for non-co3d dataset"

        # Reshape inputs for the network
        x = x.reshape(B * N_views, *x.shape[2:])
        if self.cfg.data.origin_distances:
            const_offset = x[:, 4:, ...]
            x = x[:, :4, ...]
        else:
            const_offset = None

        source_cameras_view_to_world = source_cameras_view_to_world.reshape(B * N_views, *source_cameras_view_to_world.shape[2:])
        x = x.contiguous(memory_format=torch.channels_last)

        if self.cfg.model.network_with_offset:
            split_network_outputs = self.network_with_offset(x, 
                                                             film_camera_emb=film_camera_emb,
                                                             N_views_xa=N_views_xa)
            split_network_outputs = split_network_outputs.split(self.split_dimensions_with_offset, dim=1)
            depth, offset, opacity, scaling, rotation, features_dc = split_network_outputs[:6]
            if self.cfg.model.max_sh_degree > 0:
                features_rest = split_network_outputs[6]

            pos = self.get_pos_from_network_output(depth, offset, focals_pixels, const_offset=const_offset)
        else:
            split_network_outputs = self.network_wo_offset(x, 
                                                           film_camera_emb=film_camera_emb,
                                                           N_views_xa=N_views_xa).split(self.split_dimensions_without_offset, dim=1)
            depth, opacity, scaling, rotation, features_dc = split_network_outputs[:5]
            if self.cfg.model.max_sh_degree > 0:
                features_rest = split_network_outputs[5]

            pos = self.get_pos_from_network_output(depth, 0.0, focals_pixels, const_offset=const_offset)

        # Handling isotropic scaling
        if self.cfg.model.isotropic:
            scaling_out = torch.cat([scaling[:, :1, ...], scaling[:, :1, ...], scaling[:, :1, ...]], dim=1)
        else:
            scaling_out = scaling

        # Position prediction in camera space -> world space transformation
        pos = self.flatten_vector(pos)
        pos = torch.cat([pos, 
                         torch.ones((pos.shape[0], pos.shape[1], 1), device=pos.device, dtype=torch.float32)], dim=2)
        pos = torch.bmm(pos, source_cameras_view_to_world)
        pos = pos[:, :, :3] / (pos[:, :, 3:] + 1e-10)
        
        # Prepare the output dictionary
        out_dict = {
            "xyz": pos, 
            "rotation": self.flatten_vector(self.rotation_activation(rotation)),
            "features_dc": self.flatten_vector(features_dc).unsqueeze(2)
        }

        if activate_output:
            out_dict["opacity"] = self.flatten_vector(self.opacity_activation(opacity))
            out_dict["scaling"] = self.flatten_vector(self.scaling_activation(scaling_out))
        else:
            out_dict["opacity"] = self.flatten_vector(opacity)
            out_dict["scaling"] = self.flatten_vector(scaling_out)

        ## Handling quaternion multiplication
        if source_cv2wT_quat is None:
            print("Warning: source_cv2wT_quat is None. Setting a default identity matrix.")
            source_cv2wT_quat = torch.eye(4).unsqueeze(0).repeat(B*N_views, 1, 1).to(x.device)

        out_dict["rotation"] = self.transform_rotations(out_dict["rotation"], source_cv2wT_quat=source_cv2wT_quat)

        # Handling spherical harmonics if used
        if self.cfg.model.max_sh_degree > 0:
            features_rest = self.flatten_vector(features_rest)
            out_dict["features_rest"] = features_rest.reshape(*features_rest.shape[:2], -1, 3)
            assert self.cfg.model.max_sh_degree == 1, "Only accepting degree 1"
            out_dict["features_rest"] = self.transform_SHs(out_dict["features_rest"], source_cameras_view_to_world)
        else:    
            out_dict["features_rest"] = torch.zeros((out_dict["features_dc"].shape[0], 
                                                     out_dict["features_dc"].shape[1], 
                                                     (self.cfg.model.max_sh_degree + 1) ** 2 - 1,
                                                     3), dtype=out_dict["features_dc"].dtype, device="cuda")

        out_dict = self.multi_view_union(out_dict, B, N_views)
        out_dict = self.make_contiguous(out_dict)

        return out_dict
