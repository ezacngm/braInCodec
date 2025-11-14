import os
import sys
import signal
import argparse
import numpy as np
from itertools import cycle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import h5py

from torch.utils.data import DataLoader
from models.model import ImageFeatureRegressionModel
from dataset.multisubj_dataset import MultiSubjectDataset, Context_VoxelSampler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import random


def handler(signum, frame):
    print("KeyboardInterrupt caught. Cleaning up...")
    sys.exit(0)
@torch.no_grad()
def load_wnb_stats(backbone, subj_list=[1,2,5,7], ns_list=[30,70,100]):
    subj_id = random.choice(subj_list)
    ns = random.choice(ns_list)

    stats_path = f'/grp01/wnb_stats/{backbone}/stats_subj{subj_id}_ns{ns}.npz'
    stats = np.load(stats_path)
    mean_w = torch.tensor(stats['mean_w'], dtype=torch.float32).cuda()
    cov_w  = torch.tensor(stats['cov_w'], dtype=torch.float32).cuda()
    mean_b = torch.tensor(stats['mean_b'], dtype=torch.float32).cuda()
    std_b  = torch.sqrt(torch.tensor(stats['var_b'], dtype=torch.float32)).cuda()
    cov_w.add_(torch.eye(cov_w.shape[0], device=cov_w.device) * 1e-6)
    dist_w = MultivariateNormal(loc=mean_w, covariance_matrix=cov_w)
    dist_b = torch.distributions.Normal(loc=mean_b, scale=std_b)
    return dist_w, dist_b

def sim_batch(features, dist_w, dist_b, B, V, sigma_beta=0.1, device="cuda"):
    idx = torch.randint(0, len(features), (B,))
    img_emb = features[idx].float().to(device)
    img_emb = F.normalize(img_emb, dim=-1)
    weights = dist_w.sample((B, V))
    bias = dist_b.sample((B, V, 1))
    beta_clean = torch.bmm(weights, img_emb.unsqueeze(-1))
    beta = beta_clean + sigma_beta * torch.randn_like(beta_clean)
    return img_emb, weights, bias, beta

def load_coco_features_from_h5(data_path, train_ratio=0.9):

    with h5py.File(data_path, "r") as h5f:
        img_ids = list(h5f.keys())
        img_ids.sort()
        all_feats = [h5f[k][()] for k in img_ids]
        feats = np.stack(all_feats).astype("float32")
    dataset_len = len(img_ids)
    train_len = int(train_ratio * dataset_len)
    train_feats = feats[:train_len]
    val_feats = feats[train_len:]
    train_ids = img_ids[:train_len]
    val_ids = img_ids[train_len:]
    print(f"Loaded {len(img_ids)} embeddings from {data_path}")
    print(f"Train: {len(train_feats)}, Val: {len(val_feats)}")
    return torch.from_numpy(train_feats), torch.from_numpy(val_feats), train_ids, val_ids

def train_stage2(
    train_feats,
    val_feats,
    accelerator,
    val_loader = None,
    lr=1e-5,
    device="cuda",
    wandb_on=False,
    save_model_ckpt=True,
    run_name="mymodel",
    resume=False,
    root_path="./runs",
    B=2,
    epoch=100,
    backbone="CLIP",
    v_min=100,
    v_max=200,
    val_ids = None,
    train_num_samples_list = None,
    train_subjs=None,

):

    if backbone == "CLIP":
        feature_dim,internal_emb_dim = 512, 768
    elif backbone == "DINO":
        feature_dim,internal_emb_dim = 768, 1024
    elif backbone == "SIGLIP":
        feature_dim,internal_emb_dim = 1152, 1440
    else:
        raise NotImplementedError
    model = ImageFeatureRegressionModel(input_dim = feature_dim,output_dim=feature_dim,internal_emb_dim=internal_emb_dim, num_tsfm_layers=8, num_reg_tok=32, used_token_num=4).to(device)


    model_ckpt_path = os.path.join(root_path, run_name, "model_ckpt.pth")
    os.makedirs(os.path.join(root_path, run_name), exist_ok=True)

    if resume and os.path.exists(model_ckpt_path):
        model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
        print(f"Resumed model from {model_ckpt_path}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-6)

    model, optimizer, scheduler= accelerator.prepare(
        model, optimizer,scheduler
    )

    if wandb_on:
        if accelerator.is_main_process:
            import wandb
            wandb.init(project=f"pretraining_brainicl_{backbone}", name=run_name,
                       config=dict(B=B, init_lr=lr))

    loss_cossim = nn.CosineEmbeddingLoss()
    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()
    iter_single_epoch = len(train_feats) // B
    num_steps = epoch * iter_single_epoch
    model.train()
    progress_bar = tqdm(range(num_steps), desc="Training on synthetic data")
    for step in progress_bar:
        optimizer.zero_grad()
        dist_w, dist_b = load_wnb_stats(backbone=f"{backbone}", subj_list=train_subjs, ns_list=train_num_samples_list)
        V = torch.randint(v_min, v_max, (1,)).item()  # number of voxels
        img_emb, weights, bias, beta = sim_batch(train_feats, dist_w, dist_b, B, V, sigma_beta=0.1, device=device)


        img_emb = F.normalize(img_emb, dim=-1).to(device)
        weights = weights.to(device)
        bias    = bias.to(device)
        beta    = beta.to(device)

        pred = model(beta, weights, bias).squeeze()

        pred_n   = F.normalize(pred, dim=-1)
        target_n = img_emb.detach()

        y = torch.ones(pred_n.size(0), device=device)
        mse  = loss_mse(pred_n, target_n)
        cos  = loss_cossim(pred_n, target_n, y)

        tau = 0.2
        logits_p2t = pred_n @ target_n.t() / tau
        labels = torch.arange(pred_n.size(0), device=pred_n.device)
        contra_p2t = F.cross_entropy(logits_p2t, labels)
        logits_t2p = target_n @ pred_n.t() / tau
        contra_t2p = F.cross_entropy(logits_t2p, labels)
        contra = (contra_p2t + contra_t2p) / 2

        loss = cos + 0.1 * contra

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step() # if necessary

        with torch.no_grad():
            cos_sim   = F.cosine_similarity(pred, img_emb, dim=-1).mean().item()
            std_mrtic = pred.std(dim=0).mean().item()
            if pred_n.size(0) >= 2:
                pairwise_cos = F.cosine_similarity(pred_n[0].unsqueeze(0), pred_n[1:], dim=-1).mean().item()
            else:
                pairwise_cos = 0.0

        if wandb_on:
            if accelerator.is_main_process:
                wandb.log({
                    "train/loss_mse": mse.item(),
                    "train/loss_cossim": cos.item(),
                    "train/loss_contra": contra.item(),
                    "train/loss_total": loss.item(),
                    "train/cossim_raw": cos_sim,
                    "train/std_metric": std_mrtic,
                    "train/pairwise_cos": pairwise_cos,
                }, step=step)

        progress_bar.set_postfix({
            'lmse': f"{mse.item():.4f}",
            'lcos': f"{cos.item():.4f}",
            'contra': f"{contra.item():.4f}",
            'cossim': f"{cos_sim:.4f}",
            'std': f"{std_mrtic:.4f}",
            'pwcos': f"{pairwise_cos:.4f}",
        })
        progress_bar.update(1)

        if accelerator.is_main_process and step % 1000 == 0 and step > 0:
            print("Running synthetic validation...")
            model.eval()
            all_val_cossim = []

            dist_w, dist_b= load_wnb_stats(backbone=backbone, subj_list=[7], ns_list=[100])

            for _ in tqdm(range(30), desc="Validating on synthetic batches"):
                V = torch.randint(v_max - 10, v_max, (1,)).item()
                img_emb, weights, bias, beta = sim_batch(
                    val_feats, dist_w, dist_b, B, V, sigma_beta=0.1, device=device
                )
                with torch.no_grad():
                    pred = model(beta, weights, bias).squeeze()
                    pred_emb_n = F.normalize(pred, dim=-1)
                    target_emb_n = F.normalize(img_emb, dim=-1)
                    val_cossim = F.cosine_similarity(pred_emb_n, target_emb_n, dim=-1).mean().item()
                    all_val_cossim.append(val_cossim)

            avg_val_cossim = np.mean(all_val_cossim)
            print(f"Validation Cosine Similarity (avg over 10 synthetic batches): {avg_val_cossim:.4f}")

            if wandb_on and accelerator.is_main_process:
                wandb.log({"val/avg_cossim": avg_val_cossim})

            retrieval_img_save_path = os.path.join(root_path, run_name, f'val_retrieval_epoch{step//1000:04d}k.png')
            retrieval_figure = retrieval_exp(model,device,val_loader,retrieval_img_save_path,v_num=4000,backbone=backbone)
            model.train()
            if wandb_on and accelerator.is_main_process:
                wandb.log({
                    "val/retrieval_visualization": wandb.Image(retrieval_figure)
                })
                print("Logging retrieval image to wandb...")
            if save_model_ckpt:
                model_save_path = os.path.join(root_path, run_name, 'model_ckpt.pth')
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

        accelerator.wait_for_everyone()
    return model

# ---------------------------
# main
# ---------------------------


def main():
    args = build_argparser().parse_args()
    print(f"Starting experiment:")
    print(f"  Training on subjects: {args.train_subj} (using '{args.image_set_train}' images)")
    print(f"  Validating on subjects: {args.val_subj} (using '{args.image_set_val}' images)")

    seed_everything(args.seed)
    accelerator = Accelerator()
    data_path = f'/grp01/pretraining_data/unlabeled2017_img_emb_{args.backbone}.h5py'
    train_feats, val_feats, train_ids, val_ids = load_coco_features_from_h5(data_path, train_ratio=0.9)
    print(f"Train features: {train_feats.shape}, Val features: {val_feats.shape}")

    val_ds = MultiSubjectDataset(
        subj_ids=args.val_subj,
        image_set_type=args.image_set_val,
        num_samples_options=args.val_num_samples_list,
        preload_wnb=True,
        backbone=args.backbone

    )
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers_val, drop_last=True,collate_fn=None)

    _ = train_stage2(
        train_feats,
        val_feats,
        accelerator,
        val_loader = val_loader,
        lr=args.lr,
        device=args.device,
        wandb_on=args.wandb_on,
        save_model_ckpt=not args.no_save,
        run_name=args.run_name,
        resume=args.resume,
        root_path=args.root_path,
        B=args.batch_size,
        epoch=args.num_epochs,
        backbone = args.backbone,
        v_min = args.v_min,
        v_max = args.v_max,
        val_ids = val_ids,
        train_subjs = args.train_subj,
        train_num_samples_list = args.train_num_samples_list
    )

    print("Cleaning up resources...")
    accelerator.end_training()

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print('Note: not using cudnn.deterministic')

def build_argparser():
    p = argparse.ArgumentParser(description="Stage-2 training with CLI args (Brain-CLIP decoding).")

    p.add_argument("--cuda_visible_devices", type=str, default="1", help="Set CUDA_VISIBLE_DEVICES before training.")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Torch device.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--root_path", type=str, default="./model_ckpt_realdata", help="Directory to save runs and checkpoints.")
    p.add_argument("--run_name", type=str, default="mymodel", help="Run name for logging & saving.")
    p.add_argument("--num_iterations", type=int, default=10000, help="Number of optimization steps.")
    p.add_argument("--train_subj", type=int, nargs='+',default=[1,2,5])
    p.add_argument("--val_subj", type=int, nargs='+', default=[7])
    p.add_argument("--image_set_train", type=str, default="unique", choices=["unique", "common"])
    p.add_argument("--image_set_val", type=str, default="common", choices=["unique", "common"])
    p.add_argument("--train_num_samples_list", type=int, nargs='+', default=[30,70,100,200,300,400,500,600,700])
    p.add_argument("--val_num_samples_list", type=int, nargs='+', default=[200])
    p.add_argument("--backbone", type=str, default="CLIP", help="model backbones")
    p.add_argument("--num_epochs", type=int, default=1, help="training epochs.")
    p.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    p.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size.")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    p.add_argument("--v_min", type=int, default=100, help="Min #voxels sampled per step.")
    p.add_argument("--v_max", type=int, default=200, help="Max #voxels sampled per step.")
    p.add_argument("--num_workers_train", type=int, default=4, help="Dataloader workers (train).")
    p.add_argument("--num_workers_val", type=int, default=2, help="Dataloader workers (val).")
    p.add_argument("--wandb_on", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists.")
    p.add_argument("--no_save", action="store_true", help="Do not save final checkpoint.")
    p.add_argument("--topk", type=int, default=10, help="Top-K for retrieval visualization.")
    return p

if __name__ == "__main__":
    main()
