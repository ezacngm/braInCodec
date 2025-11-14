import os
import sys
import signal
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model_current import ImageFeatureRegressionModel
from dataset.multisubj_dataset import MultiSubjectDataset, Context_VoxelSampler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def handler(signum, frame):
    print("KeyboardInterrupt caught. Cleaning up...")
    sys.exit(0)

def train_stage(
    train_loader,
    val_loader,
    accelerator,
    lr=1e-5,
    device="cuda",
    wandb_on=False,
    save_model_ckpt=True,
    run_name="mymodel",
    resume=False,
    resume_from_ckpt='./model_ckpt_realdata/model_ckpt.pth',
    root_path="./runs",
    B=2,
    epoch=1,
    backbone="CLIP",
):
    if backbone == "CLIP":
        feature_dim,internal_emb_dim = 512, 768
    elif backbone == "DINO":
        feature_dim,internal_emb_dim = 768, 1024
    elif backbone == "SIGLIP":
        feature_dim,internal_emb_dim = 1152, 1440
    else:
        raise NotImplementedError
    model = ImageFeatureRegressionModel(input_dim = feature_dim,output_dim=feature_dim,internal_emb_dim=internal_emb_dim,
                                        num_tsfm_layers=8, num_reg_tok=32, used_token_num=4).to(device)

    model_ckpt_path = os.path.join(root_path, run_name, "model_ckpt.pth")
    os.makedirs(os.path.join(root_path, run_name), exist_ok=True)

    if resume:
        if os.path.exists(resume_from_ckpt):
            model.load_state_dict(torch.load(resume_from_ckpt, map_location="cpu"))
            print(f"Resumed model from custom checkpoint: {resume_from_ckpt}")
        elif os.path.exists(model_ckpt_path):
            model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
            print(f"Resumed model from default checkpoint: {model_ckpt_path}")
        else:
            print('\n!!!No valid checkpoint found. Starting training from scratch.\n')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    total_train_steps = len(train_loader) * epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=1e-6)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    device = accelerator.device

    if wandb_on:
        if accelerator.is_main_process:
            import wandb
            wandb.init(project=f"finetuning_{backbone}", name=run_name,
                       config=dict(B=B, init_lr=lr, loss="cos+infonce"))

    loss_cossim = nn.CosineEmbeddingLoss()
    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()
    step = 0
    for epoch_idx in range(epoch):
        model.train()
        progress_bar = tqdm(train_loader, desc="Training Stage",disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            step += 1
            optimizer.zero_grad()
            img_emb, weights, bias, beta, img_ids = batch

            num_voxels = beta.shape[1]
            target = img_emb.to(device)
            target_n = F.normalize(img_emb, dim=-1).to(device)
            weights = weights.to(device)
            bias = bias.to(device)
            beta = beta.to(device)

            pred = model(beta, weights, bias).squeeze()
            pred_n = F.normalize(pred, dim=-1)

            y = torch.ones(pred_n.size(0), device=device)
            mse = loss_mse(pred, target)
            mae = loss_l1(pred, target)
            cos = loss_cossim(pred_n, target_n, y)

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
            # scheduler.step() if necessary

            with torch.no_grad():
                cos_sim   = F.cosine_similarity(pred, img_emb, dim=-1).mean().item()
                std_mrtic = pred_n.std(dim=0).mean().item()
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
                        "train/mae": mae.item(),
                        "train/cossim_raw": cos_sim,
                        "train/std_metric": std_mrtic,
                        "train/pairwise_cos": pairwise_cos,
                    }, step=step)

            progress_bar.set_postfix({
                'lmse': f"{mse.item():.4f}",
                'lcos': f"{cos.item():.4f}",
                'contra': f"{contra.item():.4f}",
                'cossim': f"{cos_sim:.4f}",
            })
            progress_bar.update(1)

        model.eval()
        all_val_cossims = []
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validating",
                                  disable=not accelerator.is_local_main_process):
                img_emb, weights, bias, beta, img_ids = val_batch
                img_emb = F.normalize(img_emb.to(accelerator.device), dim=-1)
                weights = weights.to(accelerator.device)
                bias = bias.to(accelerator.device)
                beta = beta.to(accelerator.device)

                pred = model(beta, weights, bias).squeeze()
                pred_n = F.normalize(pred, dim=-1)
                sims = F.cosine_similarity(pred_n, img_emb, dim=-1)  # [batch]
                all_val_cossims.append(sims)

        all_val_cossim = torch.cat(all_val_cossims, dim=0)
        all_val_cossim = accelerator.gather_for_metrics(all_val_cossim)
        model.train()
        if accelerator.is_main_process:
            avg_val_cossim = all_val_cossim.mean().item()
            print(f"Validation Cosine Similarity: {avg_val_cossim:.4f}")
            if wandb_on and accelerator.is_main_process:
                wandb.log({"val/avg_cossim": avg_val_cossim})
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print("trained on GPU =", os.environ["CUDA_VISIBLE_DEVICES"])
    seed_everything(args.seed)
    accelerator = Accelerator()

    v_min, v_max = args.v_min, args.v_max
    collate_fn_train = Context_VoxelSampler(v_min=v_min, v_max=v_max)
    collate_fn_val = Context_VoxelSampler(v_min=v_max-10, v_max=v_max)

    train_ds = MultiSubjectDataset(
        subj_ids=args.train_subj,
        image_set_type=args.image_set_train,
        num_samples_options=args.train_num_samples_list,
        preload_wnb=False,
        backbone = args.backbone
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers_train, drop_last=False,collate_fn=collate_fn_train)

    val_ds = MultiSubjectDataset(
        subj_ids=args.val_subj,
        image_set_type=args.image_set_val,
        num_samples_options=args.val_num_samples_list,
        preload_wnb=False,
        backbone=args.backbone

    )
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers_val, drop_last=True,collate_fn=collate_fn_val)

    _ = train_stage2(
        train_loader,
        val_loader,
        accelerator,
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
        resume_from_ckpt=args.resume_from_ckpt,

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
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_path", type=str, default="./model_ckpt_realdata")
    p.add_argument("--run_name", type=str, default="mymodel")
    p.add_argument("--num_iterations", type=int, default=10000)
    p.add_argument("--train_subj", type=int, nargs='+',default=[1,2,5])
    p.add_argument("--val_subj", type=int, nargs='+', default=[7])
    p.add_argument("--image_set_train", type=str, default="unique", choices=["unique", "common"])
    p.add_argument("--image_set_val", type=str, default="common", choices=["unique", "common"])
    p.add_argument("--train_num_samples_list", type=int, nargs='+', default=[30,70,100,200,300,400,500,600,700,1000])
    p.add_argument("--val_num_samples_list", type=int, nargs='+', default=[200])
    p.add_argument("--backbone", type=str, default="CLIP", help="model backbones")
    p.add_argument("--num_epochs", type=int, default=100, help="training epochs.")
    p.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    p.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size.")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    p.add_argument("--v_min", type=int, default=100, help="Min #voxels sampled per step.")
    p.add_argument("--v_max", type=int, default=200, help="Max #voxels sampled per step.")
    p.add_argument("--num_workers_train", type=int, default=2, help="Dataloader workers (train).")
    p.add_argument("--num_workers_val", type=int, default=0, help="Dataloader workers (val).")
    p.add_argument("--wandb_on", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists.")
    p.add_argument("--resume_from_ckpt", type=str, default="./model_ckpt_realdata/ckpt.pt")
    p.add_argument("--no_save", action="store_true", help="Do not save final checkpoint.")
    p.add_argument("--topk", type=int, default=10, help="Top-K for retrieval visualization.")
    return p

if __name__ == "__main__":
    main()
