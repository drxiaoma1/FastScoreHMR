import numpy as np
from typing import Dict, Tuple
from yacs.config import CfgNode
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
from functools import partial

from score_hmr.utils.utils import *
from score_hmr.utils.geometry import aa_to_rotmat, rot6d_to_rotmat
from score_hmr.utils.guidance_losses import keypoint_fitting_loss, multiview_loss, smoothness_loss



class Shortcut(nn.Module):
    """ Class for the Diffusion Process with Shortcut Model integration. """

    def __init__(self, cfg: CfgNode, model: nn.Module, **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = kwargs['device']
        
        # Add shortcut model specific parameters
        self.use_shortcut = cfg.MODEL.DENOISING_MODEL.USE_SHORTCUT
        self.self_consistency_ratio = cfg.TRAIN.SELF_CONSISTENCY_RATIO
        self.fm_weight = cfg.TRAIN.FM_WEIGHT
        self.sc_weight = cfg.TRAIN.SC_WEIGHT

        ### Rest of the initialization remains the same ###
        timesteps = cfg.MODEL.DIFFUSION_PROCESS.TIMESTEPS_SHORTCUT
        beta_schedule = cfg.MODEL.DIFFUSION_PROCESS.BETA_SCHEDULE
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # Helper function to register buffer from float64 to float32.
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # For q(x_t | x_{t-1}) and others.
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # For posterior q(x_{t-1} | x_t, x_0).
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_variance", posterior_variance)

        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        p2_loss_weight_gamma = 0.0
        p2_loss_weight_k = 1.0
        register_buffer("p2_loss_weight", (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        ## Denoising model
        self.model = model
        self.use_betas = self.model.use_betas
        if self.use_betas:
            betas_stats = np.load(cfg.MODEL.BETAS_STATS)
            self.betas_min = torch.from_numpy(betas_stats["betas_min"]).to(self.device)
            self.betas_max = torch.from_numpy(betas_stats["betas_max"]).to(self.device)
        self.loss_type = cfg.MODEL.DENOISING_MODEL.LOSS_TYPE
        self.objective = cfg.MODEL.DENOISING_MODEL.OBJECTIVE
        assert self.objective in {"pred_noise", "pred_x0"}, "must be pred_noise or pred_x0"

        ## Sampling
        self.use_guidance = cfg.MODEL.DENOISING_MODEL.USE_GUIDANCE
        # Score guidance.
        self.optim_iters = kwargs.get("optim_iters", cfg.GUIDANCE.OPTIM_ITERS_SHORTCUT)
        self.sample_start = kwargs.get("sample_start", cfg.GUIDANCE.SAMPLE_START_SHORTCUT)
        self.sample_step_size = kwargs.get("step_size", cfg.GUIDANCE.DDIM_STEP_SIZE_SHORTCUT)
        self.grad_scale = kwargs.get("grad_scale", cfg.GUIDANCE.GRADIENT_SCALE_SHORTCUT)
        self.use_hips = kwargs.get("use_hips", cfg.GUIDANCE.USE_HIPS)
        self.early_stop = kwargs.get("early_stopping", True)
        self.keypoint_guidance = kwargs.get("keypoint_guidance", True)

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")
        
    def predict_start_from_noise(self, x_t: torch.Tensor, t_: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Given the noised SMPL parameters x_t and the added noise, return the denoised parameters x_0.
        Args:
            x_t: Tensor of shape [B, P] containing the noised SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            t_: Tensor of shape [B] containing the timestep (noise level) for each sample in the batch.
            noise: Tensor of shape [B, P] containing the added noise.
        Returns:
            torch.Tensor of shape [B, P] with the denoised SMPL parameters.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t_, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t_, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t_: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Given the noised SMPL parameters x_t and the clean SMPL parameters x_0, return the added noise.
        Args:
            x_t: Tensor of shape [B, P] containing the noised SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            t_: Tensor of shape [B] containing the timestep (noise level) for each sample in the batch.
            x_0: Tensor of shape [B, P] with the clean SMPL parameters.
        Returns:
            torch.Tensor of shape [B, P] with added noise.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t_, x_t.shape) * x_t - x_0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t_, x_t.shape)

    def q_sample(self, x_start: torch.Tensor, t_: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Implements q(x_t | x_0).
        Args:
            x_start: Tensor of shape [B, P] with the clean SMPL parameters (B: batch_size, P: dimension of SMPL parameters).
            t_: Tensor of shape [B] containing the timestep (noise level) for each sample in the batch.
            noise: Tensor of shape [B, P] containing the added noise.
        Returns:
            torch.Tensor: Tensor of shape [B, P] containing the noised SMPL parameters.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t_, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t_, x_start.shape) * noise
        )

    def forward(self, batch: Dict, *args, **kwargs):
        batch_size = batch["img_feats"].size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        if self.use_shortcut:
            # 离散步长选择 (对应论文中的1/64, 1/32等)
            possible_step_sizes = torch.tensor(
                [1, 2, 4, 8, 16, 32, 64],  # 整数步长
                device=self.device
            )
            
            # 75%概率选单步(=1)，25%概率选大步长(>1)
            prob = torch.rand(batch_size, device=self.device)
            step_size = torch.where(
                prob > self.self_consistency_ratio,  # prob > 0.25 → 75%单步
                torch.ones(batch_size, device=self.device),  # 单步
                possible_step_sizes[
                    torch.randint(1, len(possible_step_sizes), (batch_size,))
                ]  # 25%多步
            )

            # print(f"step size: {step_size}")
            # print(f"step size length: {step_size.shape}")
            loss, pred_x_start = self.p_losses(batch, t, step_size, *args, **kwargs)
        else:
            # 原始单步训练
            loss, pred_x_start = self.p_losses(batch, t, None, *args, **kwargs)
        
        return loss, pred_x_start, t

    def p_losses(self, batch: Dict, t_: torch.Tensor, step_size: torch.Tensor = None, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified to support shortcut model training with step size conditioning.
        """
        batch_size = batch["img_feats"].size(0)
        global_orient_aa = batch["smpl_params"]["global_orient"]
        body_pose_aa = batch["smpl_params"]["body_pose"]

        # Get the 6D pose (same as before)
        global_orient_6d = (
            aa_to_rotmat(global_orient_aa.reshape(-1, 3))
            .reshape(batch_size, -1, 3, 3)[:, :, :, :2]
            .permute(0, 1, 3, 2)
            .reshape(batch_size, 1, -1)
        )  # (bs, 1, 6)
        body_pose_6d = (
            aa_to_rotmat(body_pose_aa.reshape(-1, 3))
            .reshape(batch_size, -1, 3, 3)[:, :, :, :2]
            .permute(0, 1, 3, 2)
            .reshape(batch_size, 23, -1)
        )  # (bs, 23, 6)
        pose_6d = torch.cat((global_orient_6d, body_pose_6d), dim=1)
        pose = pose_6d.reshape(batch_size, -1)  # (bs, 144)

        if self.use_betas:
            scaled_betas = normalize_betas(batch["smpl_params"]["betas"], self.betas_min, self.betas_max)
            params = torch.cat((pose, scaled_betas), dim=1)
        else:
            params = pose

        noise = default(noise, lambda: torch.randn_like(params))
        cond_feats = batch["img_feats"]

        noised_params = self.q_sample(x_start=params, t_=t_, noise=noise)

        if self.use_shortcut:
            # Forward pass with step size conditioning
            # print(f"shortcut prepared, Step size: {step_size}")
            pred_noise, pred_x_start = self.model_predictions(noised_params, t_, step_size, cond_feats)
            
            # Compute combined loss
            if step_size is not None:
                loss = self.compute_shortcut_loss(
                    x=noised_params,
                    t=t_,
                    step_size=step_size,
                    pred_noise=pred_noise,
                    cond_emb=cond_feats,
                    x_start=params,  # 真实的SMPL参数
                    noise=noise      # 真实的噪声
                )
            else:
                raise ValueError(f"step_size must be provided for shortcut model training.")
        else:
            # Original behavior
            pred_noise, pred_x_start = self.model_predictions(noised_params, t_, cond_feats)
            if self.objective == "pred_noise":
                target = noise
                model_output = pred_noise
            elif self.objective == "pred_x0":
                target = params
                model_output = pred_x_start
            loss = self.loss_fn(model_output, target, reduction="none")
            loss = reduce(loss, "b ... -> b (...)", "mean")
            loss = loss * extract(self.p2_loss_weight, t_, loss.shape)

        return loss, pred_x_start

    def compute_shortcut_loss(
        self,
        x: torch.Tensor,       
        t: torch.Tensor,       
        step_size: torch.Tensor,  
        pred_noise: torch.Tensor,  # model(xt, t, d) 的输出
        cond_emb: torch.Tensor,
        x_start: torch.Tensor,     
        noise: torch.Tensor        
    ) -> torch.Tensor:
        device = x.device
        bs = x.shape[0]
        
        # 1. Flow-matching loss (标准训练)
        fm_mask = (step_size <= 1)
        fm_loss = torch.zeros(bs, device=device)
        
        if fm_mask.any():
            alpha_t = self.alphas_cumprod[t[fm_mask]].unsqueeze(-1)
            true_noise = ((x[fm_mask] - x_start[fm_mask] * alpha_t.sqrt()) / 
                        (1 - alpha_t).sqrt())
            fm_loss[fm_mask] = self.loss_fn(
                pred_noise[fm_mask], 
                true_noise, 
                reduction='none'
            ).mean(dim=1) * self.fm_weight
        
        # 2. Self-consistency loss (Shortcut训练)
        sc_mask = (step_size > 1)
        sc_loss = torch.zeros(bs, device=device)
        
        if sc_mask.any():
            d = step_size[sc_mask].long()
            half_d = (d // 2).clamp(min=1).long()
            
            t_current = t[sc_mask]
            t_half = (t_current - half_d).clamp(min=0).long()
            
            alpha_current = self.alphas_cumprod[t_current].unsqueeze(-1)
            alpha_half = self.alphas_cumprod[t_half].unsqueeze(-1)
            
            # 使用真实噪声构造中间状态
            x_half = (x[sc_mask] * (alpha_half / alpha_current).sqrt() + 
                    noise[sc_mask] * (1 - alpha_half).sqrt())
            
            # 第一步预测：从 xt 去噪 d/2 步
            with torch.no_grad():
                noise_step1 = self.model(
                    x[sc_mask],
                    t_current,
                    half_d,
                    cond_emb[sc_mask]
                )
            
            # 计算第一步后的状态 (使用DDIM公式)
            alpha_step1 = self.alphas_cumprod[(t_current - half_d).clamp(min=0).long()].unsqueeze(-1)
            alpha_ratio_1 = (alpha_step1 / alpha_current).clamp(min=1e-8, max=1-1e-8)
            x_after_step1 = (alpha_ratio_1.sqrt() * x[sc_mask] + 
                        (1 - alpha_ratio_1).sqrt() * noise_step1)
            
            # 第二步预测：从中间状态继续去噪 d/2 步
            with torch.no_grad():
                noise_step2 = self.model(
                    x_after_step1,
                    (t_current - half_d).clamp(min=0).long(),
                    half_d,
                    cond_emb[sc_mask]
                )
            
            # 严格推导：通过两步过程计算最终状态，然后反推出等效噪声
            alpha_final = self.alphas_cumprod[(t_current - d).clamp(min=0).long()].unsqueeze(-1)
            alpha_ratio_total = (alpha_final / alpha_current).clamp(min=1e-8, max=1-1e-8)
            
            # 通过两步推导的最终状态
            x_final_two_steps = (alpha_ratio_total.sqrt() * x[sc_mask] + 
                            (1 - alpha_ratio_total).sqrt() * ((noise_step1 + noise_step2) / 2))
            
            # 反推出等效噪声
            epsilon_equivalent = (x_final_two_steps - alpha_ratio_total.sqrt() * x[sc_mask]) / \
                            (1 - alpha_ratio_total).sqrt().clamp(min=1e-8)
            
            target_noise = epsilon_equivalent.detach()
            
            # 计算损失：pred_noise vs target_noise
            sc_loss_sc = self.loss_fn(
                pred_noise[sc_mask],
                target_noise,
                reduction='none'
            ).mean(dim=1) * self.sc_weight
            
            sc_loss[sc_mask] = sc_loss_sc
        
        # 合并损失
        total_loss = fm_loss + sc_loss
        
        # 应用时间步加权
        p2_weight = extract(self.p2_loss_weight, t, total_loss.shape).squeeze()
        total_loss = total_loss * p2_weight
        
        return total_loss





    def model_predictions(
        self,
        x: torch.Tensor,
        t_: torch.Tensor,
        step_size: torch.Tensor,
        cond_feats: torch.Tensor,
        batch: Dict = None,
        clip_x_start: bool = True,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified to handle step size conditioning for shortcut models.
        Args:
            step_size: [B] 跳步大小，取值为1, 2, 4, 8, 16, 32, 64
        """
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        
        if self.use_shortcut:
            # Ensure step_size is in the correct shape
            # print(f"step size given, Step size: {step_size}")
            model_output = self.model(x, t_, step_size, cond_feats)
        else:
            model_output = self.model(x, t_, cond_feats)

        if self.objective == "pred_noise":
            pred_noise = model_output

            # TEST-TIME: Compute score guidance and the modified noise prediction.
            # if self.use_guidance and not inverse:
            #     timestep = t_[0]
            #     if timestep <= self.sample_start:
            #         # Compute score guidance, Eq. (8) in the paper.
            #         scaled_gradient, mask = self.cond_fn(x, t_, pred_noise, batch)
            #         # Compute modified noise prediction, Eq. (10) in the paper.
            #         pred_noise += self.sqrt_one_minus_alphas_cumprod[timestep].item() * scaled_gradient

            # One-step denoised result.
            x_start = self.predict_start_from_noise(x, t_, pred_noise)
            x_start = maybe_clip(x_start)

            # TEST-TIME: If using early stopping, cache the samples that converged in the current step.
            # if self.use_guidance and self.early_stop and not inverse and mask is not None and torch.any(mask).item():
            #     self.early_stop_obj.cache_results(mask, x_start, self.camera_translation)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t_, x_start)

        return pred_noise, x_start
    
    def cond_fn(
        self, x: torch.Tensor, t_: torch.Tensor, pred_noise: torch.Tensor, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Functions to compute score guidance.
        Args:
            x : Tensor of shape [B, P] containing the noised paramaters x_t (B: batch_size, P: dimension of SMPL parameters).
            t_ : Tensor of shape [B] containing the current timestep (noise level).
            pred_noise: Tensor of shape [B, P] containing the predicted noise by the denoising model.
            batch : dictionary containing the regression estimates and optionally information for model fitting.
        Returns:
            - The conditional score
            - A mask suggesting which samples coverged in the current step (if applying early stopping).
        """
        real_batch_size = batch["keypoints_2d"].size(0)
        num_samples = x.size(0) // real_batch_size

        loss = 0
        mask = None
        with torch.enable_grad():
            x_t = x.detach().requires_grad_(True)
            x_start = self.predict_start_from_noise(x_t, t_, pred_noise)
            bs_times_samples = x_start.size(0)
            pred_pose_6d = x_start[:, :-10] if self.use_betas else x_start

            if self.keypoint_guidance:
                # The SMPL betas can only contribute to KP fitting loss.
                pred_betas = (
                    unnormalize_betas(x_start[:, -10:], self.betas_min, self.betas_max)
                    if self.use_betas
                    else batch["pred_betas"].unsqueeze(1).repeat(1, num_samples, 1).reshape(real_batch_size * num_samples, -1)
                )
                pred_pose_rotmat = rot6d_to_rotmat(pred_pose_6d)

                pred_pose_rotmat = pred_pose_rotmat.view(bs_times_samples, 24, 3, 3)

                pred_smpl_params = {
                    "betas": pred_betas,
                    "global_orient": pred_pose_rotmat[:, [0]],
                    "body_pose": pred_pose_rotmat[:, 1:],
                }
                smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False)
                pred_keypoints_3d = smpl_output.joints

                # Compute the keypoint reprojection error.
                loss_kp = keypoint_fitting_loss(
                    model_joints=pred_keypoints_3d,
                    camera_translation= self.camera_translation,
                    joints_2d=batch["joints_2d"],
                    joints_conf=batch["joints_conf"],
                    camera_center=batch["camera_center"],
                    focal_length=batch["focal_length"],
                    img_size=batch["img_size"],
                )

                if self.early_stop:
                    mask = self.early_stop_obj.get_stop_mask(loss_kp)
                    loss_kp[mask] = 0.

                ## Update camera translation.
                self.camera_translation_optimizer.zero_grad()
                loss_kp.sum().backward(retain_graph=True)
                self.camera_translation_optimizer.step()

                loss = self.cfg.GUIDANCE.W_KP2D * loss_kp

            gradient = -torch.autograd.grad(loss.sum(), x_t, allow_unused=True)[0]
            return self.grad_scale * gradient, mask

    @torch.no_grad()
    def sample(self, batch: Dict, cond_feats: torch.Tensor, batch_size: int):
        """
        Run sampling with shortcut model.
        """
        shape = (batch_size, 154 if self.use_betas else 144)
        
        if self.use_guidance:
            return self.shortcut_sampling(batch, cond_feats, shape)
        else:
            return self.shortcut_vanilla(batch, cond_feats, shape, sampling_mode="from_regression")

    @torch.no_grad()
    def shortcut_vanilla(
        self,
        batch: Dict,
        cond_feats: torch.Tensor,
        shape: Tuple,
        clip_denoised: bool = True,
        eta: float = 0.0,
        sampling_mode: str = "from_regression"  # "from_regression" or "from_noise"
    ) -> Dict:
        """
        Vanilla shortcut sampling without guidance.
        Two modes:
        1. "from_regression": 从回归预测开始，先加噪再去噪
        2. "from_noise": 从纯高斯噪声开始直接去噪
        """
        device = self.device
        batch_size = shape[0]
        
        # 获取配置参数
        sample_start = min(self.sample_start, 256)  # 确保不超过最大时间步
        step_size = self.sample_step_size  # 例如 20
        
        # print(f"Vanilla sampling: mode={sampling_mode}, sample_start={sample_start}, step_size={step_size}")
        
        if sampling_mode == "from_regression":
            # 方案1: 从回归预测开始
            # 1. 获取回归预测的x_start
            x_start, _, _ = self.q_sample_verbose(batch, shape, timestep=0)  # 获取原始预测
            
            # 2. 加噪到高噪声状态 (t=sample_start)
            noise_timestep = sample_start
            time_cond_noise = torch.full((batch_size,), noise_timestep, device=device, dtype=torch.long)
            noise_for_forward = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t_=time_cond_noise, noise=noise_for_forward)
            
        else:
            # 方案2: 从纯高斯噪声开始
            x_t = torch.randn(shape, device=device)
            print("Starting from pure Gaussian noise")
        
        # 3. 构建时间步序列 (从高噪声到低噪声)
        times = list(range(0, sample_start + 1, step_size))
        if times[-1] != sample_start:
            times.append(sample_start)
        times_next = [-1] + list(times[:-1])
        time_pairs = list(reversed(list(zip(times[1:], times_next[1:]))))
        
        # print(f"Sampling time pairs: {time_pairs}")
        
        # 4. DDIM去噪循环
        for time, time_next in time_pairs:
            if time_next >= time:
                continue  # 确保是去噪方向
                
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred_noise = self.model(
                x_t, 
                time_cond,
                time_cond,
                cond_feats
            )
            x_start = self.predict_start_from_noise(x_t, time_cond, pred_noise)
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            c = (1 - alpha_next).sqrt()
            x_t = x_start * alpha_next.sqrt() + c * pred_noise
            
            # print(f"Denoising: t={time} -> t={time_next}")
        
        # 5. 最终预测的x_0
        final_x_start = self.predict_start_from_noise(
            x_t, 
            torch.full((batch_size,), 0, device=device, dtype=torch.long), 
            pred_noise
        )
        
        return {'x_0': final_x_start}

    @torch.no_grad()
    def shortcut_sampling(
        self,
        batch: Dict,
        cond_feats: torch.Tensor,
        shape: Tuple,
        num_steps: int = 256,
        step_size: int = 2,  # 跳步步长
        test_mode: bool = True
    ) -> Dict:
        """
        保留原warm-start流程的跳步采样
        Args:
            batch: 必须包含pred_pose等回归预测结果
            step_size: 跳步步长（整数），建议2的幂次如4/8/16
        """
        device = self.device
        batch_size = shape[0]
        sample_start = self.sample_start  # 低噪声阶段的采样起点

        # 加噪过程 均匀跳步
        times = list(range(0, self.sample_start + 1, self.sample_step_size))  # [0, 2, 4, ..., sample_start]
        times_next = [-1] + list(times[:-1])
        time_pairs = list(
            reversed(list(zip(times[1:], times_next[1:])))
        )  # [ ..., (20, 10), (10, 0)]
        time_pairs_inv = list(
            zip(times_next[1:], times[1:])
        )  # [(0, 10), (10, 20), ...]
        
        # 1. 从回归预测加噪初始化（原流程）
        # 初始加噪（从t=0开始）
        x_start, x_t, noise = self.q_sample_verbose(batch, shape, timestep=0)
        x_t = x_start.clone()
        x_0 = x_start.clone()
        t = torch.full((batch_size,), self.num_timesteps-1, device=device, dtype=torch.long)
        
        # 2. 初始化引导（原流程）
        self.camera_translation = None
        if self.keypoint_guidance and "joints_conf" in batch:
            # Ignore hips when fitting to 2D keypoints.
            if not self.use_hips:
                batch["joints_conf"][:, [8, 9, 12, 25 + 2, 25 + 3, 25 + 14]] *= 0.0

                # Ignore GT joints (the first 25 joints are from OpenPose, while the rest ones are GT joints, if they exist).
                batch["joints_conf"][:, 25:] = 0.0

                # Set up camera translation optimizer.
                self.camera_translation = batch["init_cam_t"]
                self.camera_translation.requires_grad_(True)
                self.camera_translation_optimizer = torch.optim.Adam([self.camera_translation], lr=1e-2)

            # Initialize object for early stopping.
            if self.early_stop:
                self.early_stop_obj = EarlyStopping(shape=shape, device=self.device, opt_trans=self.keypoint_guidance)
        
        # 3. 跳步去噪循环
        for _ in range(self.optim_iters):
            for time, time_next in time_pairs_inv:    
                # 预测噪声（带跳步条件）
                pred_noise = self.model(
                    x_t, 
                    torch.full((batch_size,), time, device=device),
                    torch.full((batch_size,), time, device=device),  
                    cond_feats
                )
                x_start = self.predict_start_from_noise(x_t, torch.full((batch_size,), time, device=device), pred_noise)
                alpha_next = self.alphas_cumprod[time_next]
                x_t = x_start * alpha_next.sqrt() + pred_noise * (1 - alpha_next).sqrt()

            for time, time_next in time_pairs:
                # 预测噪声（带跳步条件）
                pred_noise = self.model(
                    x_t, 
                    torch.full((batch_size,), time, device=device),
                    torch.full((batch_size,), time, device=device),  # 使用动态计算的步长
                    cond_feats
                )
                if self.use_guidance and "joints_conf" in batch:
                    # 计算引导梯度
                    scaled_gradient, mask = self.cond_fn(x_t, torch.full((batch_size,), time, device=device), pred_noise, batch)
                    # 更新噪声预测
                    pred_noise += self.sqrt_one_minus_alphas_cumprod[time].item() * scaled_gradient
                alpha_next = self.alphas_cumprod[time_next]
                c = (1 - alpha_next).sqrt()
                x_t = x_start * alpha_next.sqrt() + c * pred_noise
    
        # 返回格式与原代码一致
        output = {'x_0': x_t}
        if self.camera_translation is not None:
            output['camera_translation'] = self.camera_translation.detach()
        return output
    
    def q_sample_verbose(self, batch: Dict, shape: Tuple, timestep: int = 999) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Similar to q_sample(), but also returns the x_start and added noise.
        It also converts the SMPL from a regression model to the appropriate format used by the Diffusion Model.
        Args:
            batch : Dictionary containing SMPL parameter predtictions.
            shape : Tuple containing the shape of the diffusion model parameters.
            timestep : Timestep (noise level) to use.
        Returns:
            x_start : Tensor of shape [B*N, P] containing the input (of the diffusion model) for the regression estimate.
            x_t : Tensor of shape [B*N, P] containing the noised input at noise level t.
            noise: Tensor of shape [B*N, P] containing the the added noise to x_start to produce x_t.
        """
        batch_size = shape[0]
        pred_pose_rotmat = batch["pred_pose"]  # bs, 24, 3, 3
        pred_pose_6d = pred_pose_rotmat[:, :, :, :2].permute(0, 1, 3, 2)
        true_batch_size = pred_pose_6d.size(0)
        num_samples = batch_size // true_batch_size

        x_start = pred_pose_6d.reshape(true_batch_size, -1)
        # Potentially include SMPL betas in x_start.
        if self.use_betas:
            scaled_betas = normalize_betas(batch["pred_betas"], self.betas_min, self.betas_max)
            x_start = torch.cat((x_start, scaled_betas), dim=1)
        x_start = x_start.unsqueeze(1).repeat(1, num_samples, 1).reshape(batch_size, -1)

        time_cond = torch.full( (batch_size,), timestep, device=self.device, dtype=torch.long)
        # 0-th sample is mode, and if num_samples > 1 we also sample noise
        noise_mode = torch.zeros((true_batch_size, 1, x_start.size(-1)), device=self.device)
        noise_samples = torch.randn((true_batch_size, num_samples - 1, x_start.size(-1)), device=self.device)
        noise = torch.cat((noise_mode, noise_samples), dim=1).reshape(true_batch_size * num_samples, -1)

        x_t = self.q_sample(x_start=x_start, t_=time_cond, noise=noise)

        return x_start, x_t, noise
