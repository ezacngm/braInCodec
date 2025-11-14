import clip
import h5py
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torchmetrics import PearsonCorrCoef
from torchvision import transforms
from torchvision.models import (AlexNet_Weights, EfficientNet_B1_Weights,
                                Inception_V3_Weights, alexnet, efficientnet_b1,
                                inception_v3)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from .mind_utils import *

import torch
import numpy as np
from scipy.stats import pearsonr

try:
    import torchvision
    from torchvision import transforms
    from torchvision.models import (
        alexnet, AlexNet_Weights,
        inception_v3, Inception_V3_Weights,
        efficientnet_b1, EfficientNet_B1_Weights,
    )
    from torchvision.models.feature_extraction import create_feature_extractor
    import clip as openai_clip  # pip install git+https://github.com/openai/CLIP.git
    _HAS_TORCHVISION_AND_CLIP = True
except Exception:
    _HAS_TORCHVISION_AND_CLIP = False


# === (A) Two-way identification: 直接基于 sim_matrix 计算 ===
def two_way_identification_from_sim(sim_matrix: torch.Tensor) -> float:
    """
    在不依赖原始图像的前提下，复现 MindEye 中 two-way identification 的统计量。
    这里输入是 N×N 的相似度矩阵：sim[i, j] 表示第 i 个预测 与 第 j 个真值的相似度。
    返回值是平均 2-way percent-correct（越高越好，范围[0,1]）。

    说明：
    - MindEye 原实现是先提取模型(Alex/Incep/CLIP等)的特征，再做 2-way identification。
      对于你目前“embedding 对 embedding”的设定，我们可以直接在相似度矩阵上做同构计算：
        congruent = diag(sim)  # 正确配对相似度
        success_i = count(sim[i, j] < sim[i, i] for j != i)
      平均后再除以 (N-1)，得到 percent-correct。
    """
    assert sim_matrix.dim() == 2 and sim_matrix.shape[0] == sim_matrix.shape[1], \
        "two_way_identification_from_sim 需要方阵相似度矩阵 (N x N)。"

    N = sim_matrix.shape[0]
    # 取对角（正确配对）的相似度，形状 [N, 1] 以便广播
    congruent = torch.diag(sim_matrix).unsqueeze(1)  # [N,1]

    # 与每一行的其它列比较（严格小于视为成功；等于时不计为成功，避免重复计数）
    # success_mask[i, j] = True  当且仅当 j != i 且 sim[i, j] < sim[i, i]
    lt_mask = sim_matrix < congruent
    not_self = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    success_mask = lt_mask & not_self

    success_counts = success_mask.sum(dim=1).float()            # [N]
    twoway_pc = (success_counts / (N - 1)).mean().item()        # 标量
    return twoway_pc


@torch.no_grad()
def _get_model_and_preprocess(model_name: str):
    """
    复用（并轻量改写）MindEye 的 get_model / get_preprocess 思路：
    仅当你要做“基于图像像素的感知度量”时调用。
    返回：
      model: 可前向得到特征的 nn.Module 或 函数（CLIP 返回 encode_image）
      preprocess: torchvision.transforms.Compose
      return_node: 对于支持 feature_extractor 的模型，给出节点名；CLIP 为 None
    """
    assert _HAS_TORCHVISION_AND_CLIP, "未安装 torchvision/CLIP，无法启用感知特征指标。"

    if model_name == 'Incep':
        weights = Inception_V3_Weights.DEFAULT
        model = create_feature_extractor(inception_v3(weights=weights), return_nodes={'avgpool': 'feat'})
        preprocess = transforms.Compose([
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return model.eval().requires_grad_(False), preprocess, 'feat'

    elif model_name == 'CLIP':
        model, _ = openai_clip.load("ViT-L/14", device='cuda' if torch.cuda.is_available() else 'cpu')
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        return model.encode_image, preprocess, None

    elif model_name == 'Eff':
        weights = EfficientNet_B1_Weights.DEFAULT
        model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes={'avgpool': 'feat'})
        preprocess = transforms.Compose([
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return model.eval().requires_grad_(False), preprocess, 'feat'

    elif model_name == 'SwAV':
        # 需提前把 SwAV resnet50 放到本地 torch hub 或直接从远端拉取
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        model = create_feature_extractor(model, return_nodes={'avgpool': 'feat'})
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return model.eval().requires_grad_(False), preprocess, 'feat'
    else:
        raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def perceptual_two_way_percent_correct(images_gt: torch.Tensor,
                                       images_pd: torch.Tensor,
                                       model_name: str,
                                       batch_size: int = 64) -> float:
    """
    MindEye 风格：把图像对 (gt, pd) 送入 {Incep, CLIP, Eff, SwAV} 得到特征，
    再做 two-way identification，返回平均 percent-correct。
    仅当你能拿到图像像素 (B,3,H,W) 时使用；与 embedding 流程相互独立。
    """
    device = images_gt.device
    model, preprocess, node = _get_model_and_preprocess(model_name)

    def _forward_feats(x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(0, x.size(0), batch_size):
            xb = preprocess(x[i:i+batch_size]).to(device)
            if node is None:  # CLIP encode_image
                fb = model(xb).float().flatten(1)
            else:
                fb = model(xb)[node].float().flatten(1)
            feats.append(F.normalize(fb, dim=-1))
        return torch.cat(feats, dim=0).cpu().numpy()

    gt_feat = _forward_feats(images_gt)
    pd_feat = _forward_feats(images_pd)

    # 复用 two-way identification 统计（等价于 eval_neuroflow.py 的实现）
    num = len(gt_feat)
    corr = np.corrcoef(gt_feat, pd_feat)[:num, num:]
    congruent = np.diag(corr)
    success = (corr < congruent[:, None]).sum(axis=1) / (num - 1)
    return float(success.mean())


def two_way_identification_from_pearson(gt, pd, return_avg=True):
    num_samples = len(gt)
    corr_mat = np.corrcoef(gt, pd)  # compute correlation matrix
    corr_mat = corr_mat[:num_samples, num_samples:]  # extract relevant quadrant of correlation matrix

    congruent = np.diag(corr_mat)
    success = corr_mat < congruent
    success_cnt = np.sum(success, axis=0)

    if return_avg:
        return np.mean(success_cnt) / (num_samples - 1)
    else:
        return success_cnt, num_samples - 1