"""
Extract visualization data for 3DPW-TEST dataset (3 random images per sequence).
"""
import torch
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import random
from collections import defaultdict

from score_hmr.utils import *
from score_hmr.configs import dataset_config, model_config
from score_hmr.datasets import create_dataset
from score_hmr.models.model_utils import load_shortcut_model, load_pare

NUM_SAMPLES = 1

def get_sequence_indices(dataset_imgnames):
    """Group image indices by sequence."""
    sequence_dict = defaultdict(list)
    
    for idx, img_path in enumerate(dataset_imgnames):
        # img_path is like: "courtyard_arguing_00/image_00000.jpg"
        sequence_name = Path(img_path).parent.name
        sequence_dict[sequence_name].append(idx)
    
    return sequence_dict

def sample_indices_from_sequences(sequence_dict, samples_per_sequence=3):
    """Randomly sample indices from each sequence."""
    sampled_indices = []
    
    for sequence_name, indices in sequence_dict.items():
        if len(indices) <= samples_per_sequence:
            # If sequence has fewer images than samples needed, take all
            sampled_indices.extend(indices)
            print(f"Sequence {sequence_name}: taking all {len(indices)} images")
        else:
            # Randomly sample without replacement
            sampled = random.sample(indices, samples_per_sequence)
            sampled_indices.extend(sampled)
            print(f"Sequence {sequence_name}: sampled {samples_per_sequence} images from {len(indices)}")
    
    return sorted(sampled_indices)  # Keep order for consistency

def main():
    parser = argparse.ArgumentParser(description="Extract visualization data (3 random images per sequence).")
    parser.add_argument("--dataset", type=str, default="3DPW-TEST", help="Dataset to process.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--save_dir", type=str, default="diffusion_vis_data", help="Directory to save visualization data.")
    parser.add_argument("--samples_per_sequence", type=int, default=3, help="Number of samples per sequence.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_default_ckpt", action='store_true', default=False, help="Use pretrained checkpoint.")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load config
    model_cfg = model_config()
    model_cfg.defrost()
    model_cfg.MODEL.USE_BETAS = False
    model_cfg.EXTRA.LOAD_PREDICTIONS = "hmr2"
    model_cfg.freeze()

    # Dataset
    dataset_cfg = dataset_config()[args.dataset]
    full_dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    
    # Get all image names to analyze sequence structure
    print("Analyzing dataset structure...")
    dataset_imgnames = []
    for i in range(len(full_dataset)):
        dataset_imgnames.append(full_dataset[i]['imgname'])
    
    # Group indices by sequence
    sequence_dict = get_sequence_indices(dataset_imgnames)
    print(f"Found {len(sequence_dict)} sequences:")
    for seq_name, indices in list(sequence_dict.items())[:5]:  # Show first 5
        print(f"  {seq_name}: {len(indices)} images")
    if len(sequence_dict) > 5:
        print(f"  ... and {len(sequence_dict) - 5} more sequences")
    
    # Sample indices
    sampled_indices = sample_indices_from_sequences(sequence_dict, args.samples_per_sequence)
    print(f"\nTotal sampled indices: {len(sampled_indices)}")
    
    # Create subset dataset
    dataset = torch.utils.data.Subset(full_dataset, sampled_indices)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        args.batch_size,
        shuffle=False,  # Keep order consistent
        drop_last=False,
        num_workers=args.num_workers,
    )

    # Load models
    pare = load_pare(model_cfg.SMPL).to(device)
    pare.eval()

    img_feat_standarizer = StandarizeImageFeatures(
        backbone=model_cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
        use_betas=False,
        device=device,
    )

    # Load diffusion model
    extra_args = {
        "name": "keypoint_fitting",
        "keypoint_guidance": True,
        "early_stopping": True,
        "use_default_ckpt": args.use_default_ckpt,
        "device": device,
    }
    shortcut_model = load_shortcut_model(model_cfg, **extra_args)

    # Get SMPL faces
    smpl_faces = shortcut_model.smpl.faces
    print(f"SMPL faces shape: {smpl_faces.shape}")

    # Store visualization data
    vis_data = {
        'img_names': [],        # 图像文件名（完整路径，保持3DPW结构）
        'sample_indices': sampled_indices,  # 原始数据集中的索引
        'sequence_names': [],   # 序列名称
        'hmr2_verts': [],       # HMR2 顶点 [N, 6890, 3]
        'hmr2_cam_t': [],       # HMR2 相机平移 [N, 3]
        'opt_verts': [],        # 优化后顶点 [N, 6890, 3]
        'opt_cam_t': [],        # 优化后相机平移 [N, 3]
        'smpl_faces': smpl_faces,  # SMPL 面数据 [13776, 3]
    }

    print(f"\nProcessing {len(dataset)} samples ({args.samples_per_sequence} per sequence)...")
    
    processed_count = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        batch = recursive_to(batch, device)
        batch_size = batch["keypoints_3d"].size(0)

        # Get image features
        with torch.no_grad():
            pare_out = pare(batch["img"], get_feats=True)
        cond_feats = pare_out["pose_feats"].reshape(batch_size, -1)
        cond_feats = img_feat_standarizer(cond_feats)

        # Prepare for optimization
        batch["camera_center"] = 0.5 * batch["img_size"]
        gt_keypoints_2d = batch["orig_keypoints_2d"].clone()
        batch["joints_2d"] = gt_keypoints_2d[:, :, :2]
        batch["joints_conf"] = gt_keypoints_2d[:, :, [-1]]
        batch["focal_length"] = model_cfg.EXTRA.FOCAL_LENGTH * torch.ones_like(batch["camera_center"])
        batch["init_cam_t"] = batch["pred_cam_t"]

        # Get HMR2 results (baseline)
        reg_smpl_params = {
            "betas": batch["pred_betas"],
            "global_orient": batch["pred_pose"][:, [0]],
            "body_pose": batch["pred_pose"][:, 1:],
        }
        smpl_out_reg = shortcut_model.smpl(**reg_smpl_params, pose2rot=False)
        hmr2_verts = smpl_out_reg.vertices.cpu().numpy()  # [B, 6890, 3]
        hmr2_cam_t = batch["pred_cam_t"].cpu().numpy()    # [B, 3]

        # Run ScoreHMR optimization
        with torch.no_grad():
            dm_out = shortcut_model.sample(
                batch, cond_feats, batch_size=batch_size * NUM_SAMPLES
            )

        # Get optimized results
        pred_smpl_params = prepare_smpl_params(
            dm_out['x_0'],
            num_samples=NUM_SAMPLES,
            use_betas=False,
            pred_betas=batch["pred_betas"],
        )
        smpl_out = shortcut_model.smpl(**pred_smpl_params, pose2rot=False)
        opt_verts = smpl_out.vertices.cpu().numpy()       # [B, 6890, 3]
        opt_cam_t = dm_out['camera_translation'].cpu().numpy()  # [B, 3]

        # Store data
        img_names = [name for name in batch["imgname"]]  # 完整路径
        sequence_names = [Path(name).parent.name for name in img_names]  # 序列名称
        
        vis_data['img_names'].extend(img_names)
        vis_data['sequence_names'].extend(sequence_names)
        vis_data['hmr2_verts'].append(hmr2_verts)
        vis_data['hmr2_cam_t'].append(hmr2_cam_t)
        vis_data['opt_verts'].append(opt_verts)
        vis_data['opt_cam_t'].append(opt_cam_t)

        processed_count += batch_size

    # Concatenate all data
    for key in ['hmr2_verts', 'hmr2_cam_t', 'opt_verts', 'opt_cam_t']:
        if key in vis_data and len(vis_data[key]) > 0:
            vis_data[key] = np.concatenate(vis_data[key], axis=0)

    # Save visualization data
    save_path = save_dir / "shortcut_vis_data.npz"
    np.savez_compressed(save_path, **vis_data)
    print(f"\nVisualization data saved to {save_path}")
    print(f"Total samples processed: {len(vis_data['img_names'])}")
    print(f"Sample indices: {vis_data['sample_indices']}")
    print(f"Data keys: {list(vis_data.keys())}")
    
    # Print sampling summary
    print(f"\nSampling summary:")
    sequence_count = defaultdict(int)
    for seq_name in vis_data['sequence_names']:
        sequence_count[seq_name] += 1
    for seq_name, count in sequence_count.items():
        print(f"  {seq_name}: {count} images")

if __name__ == "__main__":
    main()
