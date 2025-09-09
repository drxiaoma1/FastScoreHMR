"""
Implementation of the shortcut denoising model.
Code adapted from https://github.com/statho/ScoreHMR/blob/master/score_hmr/models/denoising_model.py
"""

import torch
from torch import nn
from score_hmr.utils.model_blocks import SinusoidalPosEmb, ResMLPBlock

PREDICTORS = {
    "prohmr": {"thetas_emb_dim": 2048, "betas_emb_dim": 2048},
    "pare": {"thetas_emb_dim": 3072, "betas_emb_dim": 1536},
}

class FC_shortcut(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.use_betas = cfg.MODEL.USE_BETAS
        img_feats = cfg.MODEL.DENOISING_MODEL.IMG_FEATS
        hidden_dim = cfg.MODEL.DENOISING_MODEL.HIDDEN_LAYER_DIM
        
        # diffusion dimensions
        self.thetas_dim = cfg.MODEL.DENOISING_MODEL.POSE_DIM
        betas_dim = cfg.MODEL.DENOISING_MODEL.SHAPE_DIM if self.use_betas else 0
        self.diffusion_dim = self.thetas_dim + betas_dim

        
        # image features
        self.thetas_emb_dim = PREDICTORS[img_feats]["thetas_emb_dim"]
        self.betas_emb_dim = PREDICTORS[img_feats]["betas_emb_dim"]
        self.split_img_emb = self.use_betas and img_feats == "pare"

        # SMPL thetas
        time_dim = self.thetas_dim * 4
        sinu_pos_emb = SinusoidalPosEmb(self.thetas_dim)
        fourier_dim = self.thetas_dim
        
        # Modified time embedding to include step size
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # NEW block for step size embedding, ONLY for shortcut model
        self.step_size_mlp = nn.Sequential(
            sinu_pos_emb, 
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.init_mlp = nn.Linear(in_features=self.thetas_dim, out_features=self.thetas_dim)
        self.blocks = nn.ModuleList([])

        # Use several ResMLP blocks
        for _ in range(cfg.MODEL.DENOISING_MODEL.NUM_BLOCKS_POSE):
            self.blocks.append(
                ResMLPBlock(
                    input_dim=self.thetas_dim,
                    hidden_dim=hidden_dim,
                    time_emb_dim=time_dim * 2,  # Double for concatenated time+step emb
                    cond_emb_dim=self.thetas_emb_dim,
                )
            )
        self.final_mlp = nn.Linear(in_features=self.thetas_dim, out_features=self.thetas_dim)

        # SMPL betas (optionally)
        if self.use_betas:
            time_dim = betas_dim * 4
            sinu_pos_emb = SinusoidalPosEmb(betas_dim)
            fourier_dim = betas_dim
            
            self.time_mlp_betas = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            
            # Step size embedding for betas (new)
            self.step_mlp_betas = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            
            self.init_mlp_betas = nn.Linear(in_features=betas_dim, out_features=betas_dim)
            self.blocks_betas = nn.ModuleList([])
            for _ in range(cfg.MODEL.DENOISING_MODEL.NUM_BLOCKS_SHAPE):
                self.blocks_betas.append(
                    ResMLPBlock(
                        input_dim=betas_dim,
                        hidden_dim=hidden_dim,
                        time_emb_dim=time_dim * 2,  # Double for concatenated time+step emb
                        cond_emb_dim=self.betas_emb_dim,
                    )
                )
            self.final_mlp_betas = nn.Linear(in_features=betas_dim, out_features=betas_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor, step_size: torch.Tensor, cond_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape [B, P] containing the (noised) SMPL parameters
            time : Tensor of shape [B] containing timesteps
            step_size : Tensor of shape [B] containing desired step sizes (d in paper)
            cond_emb : Tensor of shape [B, cond_emb_dim] containing image features
        Returns:
            torch.Tensor : predicted noise with shape [B, P]
        """
        if self.use_betas:
            thetas = x[:, :-10]
            betas = x[:, -10:]
            if self.split_img_emb:
                thetas_emb = cond_emb[:, :3072]
                cam_shape_emb = cond_emb[:, 3072:]
        else:
            thetas = x

        thetas = self.init_mlp(thetas)
        tt = self.time_mlp(time)

        # step_size embedding
        ss = self.step_size_mlp(step_size)  # [B, step_size_emb_dim]

        # Concatenate time and step size embeddings
        time_step_emb = torch.cat([tt, ss], dim=-1)  # [B, time_dim*2]
        
        thetas = self.init_mlp(thetas)
        for block in self.blocks:
            thetas = block(thetas, time_step_emb, thetas_emb if self.split_img_emb else cond_emb)
        thetas = self.final_mlp(thetas)

        if self.use_betas:
            # Process betas with step size conditioning
            betas = self.init_mlp_betas(betas)
            tt_betas = self.time_mlp_betas(time)
            dt_betas = self.step_mlp_betas(step_size)
            t_emb_betas = torch.cat([tt_betas, dt_betas], dim=-1)
            
            for block in self.blocks_betas:
                betas = block(
                    betas, t_emb_betas, cam_shape_emb if self.split_img_emb else cond_emb
                )
            betas = self.final_mlp_betas(betas)

            thetas_betas = torch.cat((thetas, betas), dim=1)
            return thetas_betas
        else:
            return thetas
