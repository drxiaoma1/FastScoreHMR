import numpy as np
from typing import Dict, Tuple, Optional
from yacs.config import CfgNode
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
from functools import partial

from score_hmr.utils.utils import *
from score_hmr.utils.geometry import aa_to_rotmat, rot6d_to_rotmat
from score_hmr.utils.guidance_losses import keypoint_fitting_loss


class Shortcut(nn.Module):
    """
    Diffusion Process with Shortcut Model integration.
    This class implements a diffusion model with shortcut connections for faster sampling.
    """

    def __init__(self, cfg: CfgNode, model: nn.Module, **kwargs) -> None:
        """
        Initialize the Shortcut diffusion model.
        
        Args:
            cfg: Configuration node containing model parameters.
            model: Denoising model to use.
            **kwargs: Additional arguments including device information.
        """
        super().__init__()
        self.cfg = cfg
        self.device = kwargs['device']
        
        # Shortcut model specific parameters
        self.use_shortcut = cfg.MODEL.DENOISING_MODEL.USE_SHORTCUT
        self.self_consistency_ratio = cfg.TRAIN.SELF_CONSISTENCY_RATIO
        self.fm_weight = cfg.TRAIN.FM_WEIGHT
        self.sc_weight = cfg.TRAIN.SC_WEIGHT

        # Diffusion process parameters
        timesteps = cfg.MODEL.DIFFUSION_PROCESS.TIMESTEPS_SHORTCUT
        beta_schedule = cfg.MODEL.DIFFUSION_PROCESS.BETA_SCHEDULE
        
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # Helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # For q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_variance", posterior_variance)

        # Log calculation clipped because the posterior variance is 0 at the beginning
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        # P2 loss weight parameters
        p2_loss_weight_gamma = 0.0
        p2_loss_weight_k = 1.0
        register_buffer("p2_loss_weight", (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # Denoising model
        self.model = model
        self.use_betas = self.model.use_betas
        if self.use_betas:
            betas_stats = np.load(cfg.MODEL.BETAS_STATS)
            self.betas_min = torch.from_numpy(betas_stats["betas_min"]).to(self.device)
            self.betas_max = torch.from_numpy(betas_stats["betas_max"]).to(self.device)
        
        self.loss_type = cfg.MODEL.DENOISING_MODEL.LOSS_TYPE
        self.objective = cfg.MODEL.DENOISING_MODEL.OBJECTIVE
        assert self.objective in {"pred_noise", "pred_x0"}, "Objective must be pred_noise or pred_x0"

        # Sampling parameters
        self.use_guidance = cfg.MODEL.DENOISING_MODEL.USE_GUIDANCE
        # Score guidance
        self.optim_iters = kwargs.get("optim_iters", cfg.GUIDANCE.OPTIM_ITERS_SHORTCUT)
        self.sample_start = kwargs.get("sample_start", cfg.GUIDANCE.SAMPLE_START_SHORTCUT)
        self.sample_step_size = kwargs.get("step_size", cfg.GUIDANCE.DDIM_STEP_SIZE_SHORTCUT)
        self.grad_scale = kwargs.get("grad_scale", cfg.GUIDANCE.GRADIENT_SCALE_SHORTCUT)
        self.use_hips = kwargs.get("use_hips", cfg.GUIDANCE.USE_HIPS)
        self.early_stop = kwargs.get("early_stopping", True)
        self.keypoint_guidance = kwargs.get("keypoint_guidance", True)

    @property
    def loss_fn(self):
        """
        Get the loss function based on configuration.
        
        Returns:
            Loss function (L1 or L2).
        """
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"Invalid loss type {self.loss_type}")
        
    def predict_start_from_noise(self, x_t: torch.Tensor, t_: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Given the noised SMPL parameters x_t and the added noise, return the denoised parameters x_0.
        
        Args:
            x_t: Tensor of shape [B, P] containing the noised SMPL parameters.
            t_: Tensor of shape [B] containing the timestep for each sample.
            noise: Tensor of shape [B, P] containing the added noise.
            
        Returns:
            Tensor of shape [B, P] with the denoised SMPL parameters.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t_, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t_, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t_: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Given the noised SMPL parameters x_t and the clean SMPL parameters x_0, return the added noise.
        
        Args:
            x_t: Tensor of shape [B, P] containing the noised SMPL parameters.
            t_: Tensor of shape [B] containing the timestep for each sample.
            x_0: Tensor of shape [B, P] with the clean SMPL parameters.
            
        Returns:
            Tensor of shape [B, P] with added noise.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t_, x_t.shape) * x_t - x_0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t_, x_t.shape)

    def q_sample(self, x_start: torch.Tensor, t_: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implement q(x_t | x_0) - forward diffusion process.
        
        Args:
            x_start: Tensor of shape [B, P] with the clean SMPL parameters.
            t_: Tensor of shape [B] containing the timestep for each sample.
            noise: Optional tensor of shape [B, P] containing the noise to add.
            
        Returns:
            Tensor of shape [B, P] containing the noised SMPL parameters.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t_, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t_, x_start.shape) * noise
        )

    def forward(self, batch: Dict, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            batch: Input data batch.
            
        Returns:
            Tuple containing loss, predicted x_start, and timesteps.
        """
        batch_size = batch["img_feats"].size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        if self.use_shortcut:
            # Discrete step size selection (corresponding to 1/64, 1/32, etc. in the paper)
            possible_step_sizes = torch.tensor(
                [1, 2, 4, 8, 16, 32, 64],  # Integer step sizes
                device=self.device
            )
            
            # 75% probability for single step (=1), 25% for larger steps (>1)
            prob = torch.rand(batch_size, device=self.device)
            step_size = torch.where(
                prob > self.self_consistency_ratio,  # prob > 0.25 → 75% single step
                torch.ones(batch_size, device=self.device),  # Single step
                possible_step_sizes[
                    torch.randint(1, len(possible_step_sizes), (batch_size,))
                ]  # 25% multi-step
            )

            loss, pred_x_start = self.p_losses(batch, t, step_size, *args, **kwargs)
        else:
            # Original single-step training
            loss, pred_x_start = self.p_losses(batch, t, None, *args, **kwargs)
        
        return loss, pred_x_start, t

    def p_losses(self, batch: Dict, t_: torch.Tensor, step_size: Optional[torch.Tensor] = None, 
                noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute losses for the diffusion process with support for shortcut model training.
        
        Args:
            batch: Input data batch.
            t_: Timestep tensor.
            step_size: Step size tensor for shortcut model.
            noise: Optional noise tensor.
            
        Returns:
            Tuple containing loss and predicted x_start.
        """
        batch_size = batch["img_feats"].size(0)
        global_orient_aa = batch["smpl_params"]["global_orient"]
        body_pose_aa = batch["smpl_params"]["body_pose"]

        # Convert to 6D rotation representation
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
            pred_noise, pred_x_start = self.model_predictions(noised_params, t_, step_size, cond_feats)
            
            # Compute combined loss
            if step_size is not None:
                loss = self.compute_shortcut_loss(
                    x=noised_params,
                    t=t_,
                    step_size=step_size,
                    pred_noise=pred_noise,
                    cond_emb=cond_feats,
                    x_start=params,  # Ground truth SMPL parameters
                    noise=noise      # Ground truth noise
                )
            else:
                raise ValueError("Step size must be provided for shortcut model training.")
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
        pred_noise: torch.Tensor,
        cond_emb: torch.Tensor,
        x_start: torch.Tensor,     
        noise: torch.Tensor        
    ) -> torch.Tensor:
        """
        Compute the combined flow-matching and self-consistency loss for shortcut training.
        
        Args:
            x: Noised parameters.
            t: Timestep tensor.
            step_size: Step size tensor.
            pred_noise: Predicted noise from model.
            cond_emb: Conditional features.
            x_start: Ground truth parameters.
            noise: Ground truth noise.
            
        Returns:
            Combined loss tensor.
        """
        device = x.device
        bs = x.shape[0]
        
        # 1. Flow-matching loss (standard training)
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
        
        # 2. Self-consistency loss (Shortcut training)
        sc_mask = (step_size > 1)
        sc_loss = torch.zeros(bs, device=device)
        
        if sc_mask.any():
            d = step_size[sc_mask].long()
            half_d = (d // 2).clamp(min=1).long()
            
            t_current = t[sc_mask]
            t_half = (t_current - half_d).clamp(min=0).long()
            
            alpha_current = self.alphas_cumprod[t_current].unsqueeze(-1)
            alpha_half = self.alphas_cumprod[t_half].unsqueeze(-1)
            
            # Use ground truth noise to construct intermediate state
            x_half = (x[sc_mask] * (alpha_half / alpha_current).sqrt() + 
                    noise[sc_mask] * (1 - alpha_half).sqrt())
            
            # First step prediction: denoise d/2 steps from x_t
            with torch.no_grad():
                noise_step1 = self.model(
                    x[sc_mask],
                    t_current,
                    half_d,
                    cond_emb[sc_mask]
                )
            
            # Compute state after first step (using DDIM formula)
            alpha_step1 = self.alphas_cumprod[(t_current - half_d).clamp(min=0).long()].unsqueeze(-1)
            alpha_ratio_1 = (alpha_step1 / alpha_current).clamp(min=1e-8, max=1-1e-8)
            x_after_step1 = (alpha_ratio_1.sqrt() * x[sc_mask] + 
                        (1 - alpha_ratio_1).sqrt() * noise_step1)
            
            # Second step prediction: continue denoising d/2 steps from intermediate state
            with torch.no_grad():
                noise_step2 = self.model(
                    x_after_step1,
                    (t_current - half_d).clamp(min=0).long(),
                    half_d,
                    cond_emb[sc_mask]
                )
            
            # Strict derivation: compute final state through two-step process
            alpha_final = self.alphas_cumprod[(t_current - d).clamp(min=0).long()].unsqueeze(-1)
            alpha_ratio_total = (alpha_final / alpha_current).clamp(min=1e-8, max=1-1e-8)
            
            # Final state through two-step derivation
            x_final_two_steps = (alpha_ratio_total.sqrt() * x[sc_mask] + 
                            (1 - alpha_ratio_total).sqrt() * ((noise_step1 + noise_step2) / 2))
            
            # Derive equivalent noise
            epsilon_equivalent = (x_final_two_steps - alpha_ratio_total.sqrt() * x[sc_mask]) / \
                            (1 - alpha_ratio_total).sqrt().clamp(min=1e-8)
            
            target_noise = epsilon_equivalent.detach()
            
            # Compute loss: pred_noise vs target_noise
            sc_loss_sc = self.loss_fn(
                pred_noise[sc_mask],
                target_noise,
                reduction='none'
            ).mean(dim=1) * self.sc_weight
            
            sc_loss[sc_mask] = sc_loss_sc
        
        # Combine losses
        total_loss = fm_loss + sc_loss
        
        # Apply timestep weighting
        p2_weight = extract(self.p2_loss_weight, t, total_loss.shape).squeeze()
        total_loss = total_loss * p2_weight
        
        return total_loss

    def model_predictions(
        self,
        x: torch.Tensor,
        t_: torch.Tensor,
        step_size: torch.Tensor,
        cond_feats: torch.Tensor,
        batch: Optional[Dict] = None,
        clip_x_start: bool = True,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model predictions for given noised parameters.
        
        Args:
            x: Noised parameters.
            t_: Timestep tensor.
            step_size: Step size tensor for shortcut models.
            cond_feats: Conditional features.
            batch: Optional input batch.
            clip_x_start: Whether to clip predicted x_start.
            inverse: Whether this is for inverse process.
            
        Returns:
            Tuple containing predicted noise and predicted x_start.
        """
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        
        if self.use_shortcut:
            model_output = self.model(x, t_, step_size, cond_feats)
        else:
            model_output = self.model(x, t_, cond_feats)

        if self.objective == "pred_noise":
            pred_noise = model_output
            # One-step denoised result
            x_start = self.predict_start_from_noise(x, t_, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t_, x_start)

        return pred_noise, x_start
    
    def cond_fn(
        self, x: torch.Tensor, t_: torch.Tensor, pred_noise: torch.Tensor, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute score guidance for conditional sampling.
        
        Args:
            x: Noised parameters.
            t_: Timestep tensor.
            pred_noise: Predicted noise.
            batch: Input batch.
            
        Returns:
            Tuple containing conditional score and convergence mask.
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
                # The SMPL betas can only contribute to KP fitting loss
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

                # Compute keypoint reprojection error
                loss_kp = keypoint_fitting_loss(
                    model_joints=pred_keypoints_3d,
                    camera_translation=self.camera_translation,
                    joints_2d=batch["joints_2d"],
                    joints_conf=batch["joints_conf"],
                    camera_center=batch["camera_center"],
                    focal_length=batch["focal_length"],
                    img_size=batch["img_size"],
                )

                if self.early_stop:
                    mask = self.early_stop_obj.get_stop_mask(loss_kp)
                    loss_kp[mask] = 0.

                # Update camera translation
                self.camera_translation_optimizer.zero_grad()
                loss_kp.sum().backward(retain_graph=True)
                self.camera_translation_optimizer.step()

                loss = self.cfg.GUIDANCE.W_KP2D * loss_kp

            gradient = -torch.autograd.grad(loss.sum(), x_t, allow_unused=True)[0]
            return self.grad_scale * gradient, mask

    @torch.no_grad()
    def sample(self, batch: Dict, cond_feats: torch.Tensor, batch_size: int) -> Dict:
        """
        Run sampling with shortcut model.
        
        Args:
            batch: Input batch.
            cond_feats: Conditional features.
            batch_size: Batch size for sampling.
            
        Returns:
            Dictionary containing sampling results.
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
        sampling_mode: str = "from_regression"
    ) -> Dict:
        """
        Vanilla shortcut sampling without guidance.
        
        Args:
            batch: Input batch.
            cond_feats: Conditional features.
            shape: Shape of output tensor.
            clip_denoised: Whether to clip denoised values.
            eta: DDIM eta parameter.
            sampling_mode: Sampling mode ("from_regression" or "from_noise").
            
        Returns:
            Dictionary containing sampling results.
        """
        device = self.device
        batch_size = shape[0]
        
        # Get configuration parameters
        sample_start = min(self.sample_start, 256)  # Ensure it doesn't exceed max timestep
        step_size = self.sample_step_size  # e.g., 20
        
        if sampling_mode == "from_regression":
            # Option 1: Start from regression prediction
            x_start, _, _ = self.q_sample_verbose(batch, shape, timestep=0)  # Get original prediction
            
            # Add noise to high noise state (t=sample_start)
            noise_timestep = sample_start
            time_cond_noise = torch.full((batch_size,), noise_timestep, device=device, dtype=torch.long)
            noise_for_forward = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t_=time_cond_noise, noise=noise_for_forward)
            
        else:
            # Option 2: Start from pure Gaussian noise
            x_t = torch.randn(shape, device=device)
        
        # Build timestep sequence (from high to low noise)
        times = list(range(0, sample_start + 1, step_size))
        if times[-1] != sample_start:
            times.append(sample_start)
        times_next = [-1] + list(times[:-1])
        time_pairs = list(reversed(list(zip(times[1:], times_next[1:]))))
        
        # DDIM denoising loop
        for time, time_next in time_pairs:
            if time_next >= time:
                continue  # Ensure denoising direction
                
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
        
        # Final predicted x_0
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
        step_size: int = 2,
        test_mode: bool = True
    ) -> Dict:
        """
        Shortcut sampling with warm-start process.
        
        Args:
            batch: Input batch containing regression predictions.
            cond_feats: Conditional features.
            shape: Shape of output tensor.
            num_steps: Number of sampling steps.
            step_size: Step size for jumping.
            test_mode: Whether in test mode.
            
        Returns:
            Dictionary containing sampling results.
        """
        device = self.device
        batch_size = shape[0]
        sample_start = self.sample_start  # Sampling start point for low noise phase

        # Noise addition process with uniform jumping
        times = list(range(0, self.sample_start + 1, self.sample_step_size))  # [0, 2, 4, ..., sample_start]
        times_next = [-1] + list(times[:-1])
        time_pairs = list(reversed(list(zip(times[1:], times_next[1:]))))  # [ ..., (20, 10), (10, 0)]
        time_pairs_inv = list(zip(times_next[1:], times[1:]))  # [(0, 10), (10, 20), ...]
        
        # Initialize from regression prediction (original process)
        x_start, x_t, noise = self.q_sample_verbose(batch, shape, timestep=0)
        x_t = x_start.clone()
        
        # Initialize guidance (original process)
        self.camera_translation = None
        if self.keypoint_guidance and "joints_conf" in batch:
            # Ignore hips when fitting to 2D keypoints
            if not self.use_hips:
                batch["joints_conf"][:, [8, 9, 12, 25 + 2, 25 + 3, 25 + 14]] *= 0.0
                # Ignore GT joints
                batch["joints_conf"][:, 25:] = 0.0

                # Set up camera translation optimizer
                self.camera_translation = batch["init_cam_t"]
                self.camera_translation.requires_grad_(True)
                self.camera_translation_optimizer = torch.optim.Adam([self.camera_translation], lr=1e-2)

            # Initialize object for early stopping
            if self.early_stop:
                self.early_stop_obj = EarlyStopping(shape=shape, device=self.device, opt_trans=self.keypoint_guidance)
        
        # Jump-step denoising loop
        for _ in range(self.optim_iters):
            for time, time_next in time_pairs_inv:    
                # Predict noise (with step conditioning)
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
                # Predict noise (with step conditioning)
                pred_noise = self.model(
                    x_t, 
                    torch.full((batch_size,), time, device=device),
                    torch.full((batch_size,), time, device=device),
                    cond_feats
                )
                if self.use_guidance and "joints_conf" in batch:
                    # Compute guidance gradient
                    scaled_gradient, mask = self.cond_fn(x_t, torch.full((batch_size,), time, device=device), pred_noise, batch)
                    # Update noise prediction
                    pred_noise += self.sqrt_one_minus_alphas_cumprod[time].item() * scaled_gradient
                alpha_next = self.alphas_cumprod[time_next]
                c = (1 - alpha_next).sqrt()
                x_t = x_start * alpha_next.sqrt() + c * pred_noise
    
        # Return format consistent with original code
        output = {'x_0': x_t}
        if self.camera_translation is not None:
            output['camera_translation'] = self.camera_translation.detach()
        return output
    
    def q_sample_verbose(self, batch: Dict, shape: Tuple, timestep: int = 999) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extended version of q_sample() that also returns x_start and added noise.
        
        Args:
            batch: Dictionary containing SMPL parameter predictions.
            shape: Shape of diffusion model parameters.
            timestep: Timestep to use.
            
        Returns:
            Tuple containing x_start, x_t, and noise tensors.
        """
        batch_size = shape[0]
        pred_pose_rotmat = batch["pred_pose"]  # bs, 24, 3, 3
        pred_pose_6d = pred_pose_rotmat[:, :, :, :2].permute(0, 1, 3, 2)
        true_batch_size = pred_pose_6d.size(0)
        num_samples = batch_size // true_batch_size

        x_start = pred_pose_6d.reshape(true_batch_size, -1)
        # Potentially include SMPL betas in x_start
        if self.use_betas:
            scaled_betas = normalize_betas(batch["pred_betas"], self.betas_min, self.betas_max)
            x_start = torch.cat((x_start, scaled_betas), dim=1)
        x_start = x_start.unsqueeze(1).repeat(1, num_samples, 1).reshape(batch_size, -1)

        time_cond = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
        # 0-th sample is mode, and if num_samples > 1 we also sample noise
        noise_mode = torch.zeros((true_batch_size, 1, x_start.size(-1)), device=self.device)
        noise_samples = torch.randn((true_batch_size, num_samples - 1, x_start.size(-1)), device=self.device)
        noise = torch.cat((noise_mode, noise_samples), dim=1).reshape(true_batch_size * num_samples, -1)

        x_t = self.q_sample(x_start=x_start, t_=time_cond, noise=noise)

        return x_start, x_t, noise
