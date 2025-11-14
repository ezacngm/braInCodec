
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from io import BytesIO
from PIL import Image, ImageDraw
# ---------------------------
# Utilities
# ---------------------------

def load_pil_from_h5(h5: h5py.File, key: str) -> Image.Image:
    dset = h5[key]
    data = dset[()]
    if isinstance(data, (bytes, bytearray)):
        return Image.open(BytesIO(data)).convert("RGB")
    if isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 1:
        return Image.open(BytesIO(bytes(data))).convert("RGB")
    if isinstance(data, np.ndarray):
        return _to_pil_from_array(data)
    raise ValueError(f"Unsupported HDF5 image format for key={key}, type={type(data)}")

def _fit_into_cell(img: Image.Image, cell_size=(224, 224), bgcolor=(255, 255, 255)):
    W, H = cell_size
    w, h = img.size
    if w == 0 or h == 0:
        canvas = Image.new("RGB", cell_size, bgcolor)
        return canvas
    scale = min(W / w, H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", cell_size, bgcolor)
    off_x = (W - nw) // 2
    off_y = (H - nh) // 2
    canvas.paste(img_resized, (off_x, off_y))
    return canvas

def two_way_identification_from_sim_matrix(sim_matrix, return_avg=True) -> float:
    assert sim_matrix.dim() == 2 and sim_matrix.shape[0] == sim_matrix.shape[1], \
        "two_way_identification_from_sim (N x N)"

    N = sim_matrix.shape[0]
    congruent = torch.diag(sim_matrix).unsqueeze(1)  # [N,1]
    lt_mask = sim_matrix < congruent
    not_self = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    success_mask = lt_mask & not_self

    success_counts = success_mask.sum(dim=1).float()# [N]

    if return_avg:
        twoway_pc = (success_counts / (N - 1)).mean().item()
        return twoway_pc
    else:
        return success_counts, N-1

def two_way_identification_from_pearson(gt, pd, return_avg=True):

    assert gt.shape == pd.shape, "gt and pd must have the same shape"
    num_samples = gt.shape[0]

    # compute correlation matrix: (N×D) @ (D×N) -> (N×N)
    gt_mean = gt - gt.mean(dim=1, keepdim=True)
    pd_mean = pd - pd.mean(dim=1, keepdim=True)

    gt_norm = gt_mean / gt_mean.norm(dim=1, keepdim=True)
    pd_norm = pd_mean / pd_mean.norm(dim=1, keepdim=True)
    corr_mat = torch.mm(gt_norm, pd_norm.T)

    # extract diagonal (congruent correlations)
    congruent = torch.diag(corr_mat)  # shape [N]

    # compare correlations: whether congruent corr > others
    success = corr_mat < congruent.unsqueeze(0)  # [N, N]
    success_cnt = success.sum(dim=0)  # count per prediction

    if return_avg:
        return (success_cnt.float().mean() / (num_samples - 1)).item()
    else:
        return success_cnt, num_samples - 1



def two_way_identification_from_cosine(gt, pd, return_avg=True):

    assert gt.shape == pd.shape, "gt and pd must have the same shape"
    num_samples = gt.shape[0]

    # cosine similarity: L2-normalize each vector (no mean-centering)
    gt_norms = gt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pd_norms = pd.norm(dim=1, keepdim=True).clamp_min(1e-12)

    gt_unit = gt / gt_norms
    pd_unit = pd / pd_norms

    # similarity matrix: (N×D) @ (D×N) -> (N×N)
    sim_mat = torch.mm(gt_unit, pd_unit.T)

    # diagonal: congruent similarities
    congruent = torch.diag(sim_mat)  # [N]

    # success if congruent similarity is strictly greater than all others in the same column
    success = sim_mat < congruent.unsqueeze(0)  # [N, N], compare per column j
    success_cnt = success.sum(dim=0)            # counts per prediction j

    if return_avg:
        return (success_cnt.float().mean() / (num_samples - 1)).item()
    else:
        return success_cnt, num_samples - 1


def pairwise_cosine_similarity(A, B,return_avg=True):
    # A, B: [N, D]
    A_norm = A / (A.norm(dim=1, keepdim=True))
    B_norm = B / (B.norm(dim=1, keepdim=True))
    sims = (A_norm * B_norm).sum(dim=1)  # shape [N]
    if return_avg:
        return sims.float().mean().item()
    else:
        return sims, sims.shape[0]


def paired_pearson_correlation(A, B, eps=1e-8, return_avg=True):
    # A, B: [N, D]
    # mean-center each row (Pearson correlation = cosine of centered vectors)
    A_centered = A - A.mean(dim=1, keepdim=True)
    B_centered = B - B.mean(dim=1, keepdim=True)

    # normalize (prevent div-by-zero)
    A_norm = A_centered / (A_centered.norm(dim=1, keepdim=True).clamp_min(eps))
    B_norm = B_centered / (B_centered.norm(dim=1, keepdim=True).clamp_min(eps))

    # compute per-pair correlation (equivalent to cosine of centered vectors)
    corrs = (A_norm * B_norm).sum(dim=1)  # [N]

    if return_avg:
        return corrs.float().mean().item()
    else:
        return corrs, corrs.shape[0]


def compute_global_mean_std(h5_path):
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        print(f"Total {len(keys)} embeddings found.")

        all_embs = []
        for k in keys:
            emb = np.array(f[k])
            emb = emb.squeeze()
            all_embs.append(emb)

        all_embs = np.stack(all_embs, axis=0)  # (N, 512)
        mean = all_embs.mean(axis=0)           # (512,)
        std  = all_embs.std(axis=0, ddof=0)    # (512,)

    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t  = torch.tensor(std, dtype=torch.float32)
    return mean_t, std_t


def load_embeddings_dict(npz_path, device):
    data = np.load(npz_path)
    emb_dict = {k: torch.tensor(v, device=device) for k, v in data.items()}
    return emb_dict

@torch.no_grad()
def topk_by_free_texts(gallery_feats, gallery_ids, model, query_prompts, k=20, device="cuda"):
    txt_feats = encode_texts(model, query_prompts, device=device)  # [P, D]
    imgs = gallery_feats.to(device)
    sims_np = imgs @ txt_feats.T
    sims_max, _ = sims_np.max(dim=1)
    topk_scores, topk_idx = torch.topk(sims_max, k=min(k, sims_max.numel()), largest=True)
    topk_feats = imgs[topk_idx]
    topk_ids = [gallery_ids[i] for i in topk_idx.tolist()]
    return topk_feats, topk_ids, topk_scores