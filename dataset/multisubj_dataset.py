import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import random
import os
import glob
from typing import List, Union

class MultiSubjectDataset(Dataset):
    def __init__(self,
                 subj_ids: list,
                 image_set_type = 'unique',
                 backbone = 'CLIP',
                 num_samples_options = 200,
                 fixed_ep=None,
                 preload_wnb = True,
                 img_key_path='/mnt/img_idx.json',
                 nrn_path_template='/mnt/cortex_data/cortex_subj_{subj_id}_masked.h5',
                 ):
        super().__init__()

        self.subj_ids = subj_ids
        self.fixed_ep = fixed_ep


        if isinstance(num_samples_options, int):
            self.num_samples_options = [num_samples_options]
        else:
            self.num_samples_options = sorted(list(set(num_samples_options)))

        with open(img_key_path, 'r') as f:
            self.img_keys_dict = json.load(f)

        self.img_emb_path = f'/mnt/img_emb_{backbone}.h5py'
        self.img_emb_handle = None


        self.wnb_arrays = {s: {} for s in self.subj_ids}
        self._wnb_paths = {s: {} for s in self.subj_ids}
        self.available_runs = {s: [] for s in subj_ids}
        self.nrn_paths = {s: nrn_path_template.format(subj_id=s) for s in self.subj_ids}
        self.nrn_handles = {}
        self.preload_wnb = preload_wnb


        print(f"Initializing dataset for subjects: {subj_ids} using '{image_set_type}' images.")
        print(f"num_samples options: {self.num_samples_options}, preload W&B: {self.preload_wnb}")

        for s_id in subj_ids:
            for ns in self.num_samples_options:
                wnb_root = f'/mnt/sample_img_pred_weights/{backbone}_te{s_id}'
                search_pattern = os.path.join(
                    wnb_root,
                    f"subj{s_id}_{backbone}_numic{ns}_ep*",
                    f"num_ic={ns}_ep=*_pred_weights.npy"
                )
                paths = sorted(glob.glob(search_pattern))
                if not paths:
                    print(f"WARNING: No W&B files found for subject {s_id}, num_samples {ns}.")
                    continue

                if preload_wnb:
                    arrs = [np.load(p) for p in paths]
                    self.wnb_arrays[s_id][ns] = arrs

                    if self.fixed_ep is not None:
                        if self.fixed_ep < len(arrs):
                            self.available_runs[s_id].append((ns, self.fixed_ep))
                        else:
                            print(f"Warning: subj {s_id} ns={ns} has only {len(arrs)} runs, "
                                f"fixed_ep={self.fixed_ep} ignored.")
                    else:
                        for ep_idx in range(len(arrs)):
                            self.available_runs[s_id].append((ns, ep_idx))

                else:
                    self._wnb_paths[s_id][ns] = paths

                    if self.fixed_ep is not None:
                        if self.fixed_ep < len(paths):
                            self.available_runs[s_id].append((ns, paths[self.fixed_ep]))
                        else:
                            print(f"Warning: subj {s_id} ns={ns} has only {len(paths)} runs, "
                                f"fixed_ep={self.fixed_ep} ignored.")
                    else:
                        for ep_idx, p in enumerate(paths):
                            self.available_runs[s_id].append((ns, p))
        # print(self._wnb_paths)
        self.data_index = []
        if image_set_type == 'unique':
            for s_id in subj_ids:
                keys = self.img_keys_dict.get(f's{s_id}_unique', [])
                self.data_index.extend((k, s_id) for k in keys)
        elif image_set_type == 'common':
            common_keys = self.img_keys_dict.get('common', [])
            self.data_index.extend((k, s) for s in subj_ids for k in common_keys)
        else:
            raise ValueError("image_set_type must be 'unique' or 'common'")

        total_available = sum(len(v) for v in self.available_runs.values())
        print(f"Dataset initialized with {len(self.data_index)} samples "
              f"and {total_available} available runs across all subjects.")

    def _lazy_load_data(self, subj_id: int, num_samples: int):

        if self.img_emb_handle is None:
            self.img_emb_handle = h5py.File(self.img_emb_path, 'r')
        if subj_id not in self.nrn_handles:
            self.nrn_handles[subj_id] = h5py.File(self.nrn_paths[subj_id], 'r')

        if num_samples not in self.wnb_arrays.get(subj_id, {}):
            paths = self._wnb_paths[subj_id][num_samples]
            self.wnb_arrays[subj_id][num_samples] = [np.load(p) for p in paths]

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        img_id_for_emb, subj_id = self.data_index[idx]
        img_id_for_nrn = str(int(img_id_for_emb))

        # Skip subjects with no available runs
        if subj_id not in self.available_runs or len(self.available_runs[subj_id]) == 0:
            raise RuntimeError(f"No available W&B runs for subject {subj_id}")

        # Randomly pick a (num_samples, ep)
        num_samples_choice, run_spec = random.choice(self.available_runs[subj_id])

        self._lazy_load_data(subj_id, num_samples_choice)

        # Select W&B weights
        if self.preload_wnb:
            wnb = self.wnb_arrays[subj_id][num_samples_choice][run_spec]
        else:
            wnb = np.load(run_spec)
        weights = wnb[:, :-1].astype(np.float32)
        bias = wnb[:, -1:].astype(np.float32)


        beta = np.array(self.nrn_handles[subj_id][img_id_for_nrn], dtype=np.float32).reshape(-1, 1)


        gt_img_emb = np.array(self.img_emb_handle[img_id_for_emb], dtype=np.float32).squeeze()

        return (
            torch.from_numpy(gt_img_emb).float(),
            torch.from_numpy(weights).float(),
            torch.from_numpy(bias).float(),
            torch.from_numpy(beta).float(),
            f"subj{subj_id}_{img_id_for_emb}",
        )

    def __del__(self):
        try:
            if self.img_emb_handle:
                self.img_emb_handle.close()
            for handle in self.nrn_handles.values():
                if handle:
                    handle.close()
        except Exception:
            pass

class Context_VoxelSampler:
    def __init__(self, v_min, v_max):
        self.v_min = v_min
        self.v_max = v_max

    def __call__(self, batch):
        min_voxels_in_batch = min(item[3].shape[0] for item in batch)
        v_num = random.randint(self.v_min, self.v_max)
        if v_num > min_voxels_in_batch:
            v_num = min_voxels_in_batch

        sampled_weights, sampled_biases, sampled_betas = [], [], []
        for _, weights, bias, beta, _ in batch:
            total_voxels = beta.shape[0]
            voxel_idx = torch.randperm(total_voxels)[:v_num]
            sampled_weights.append(weights[voxel_idx, :])
            sampled_biases.append(bias[voxel_idx, :])
            sampled_betas.append(beta[voxel_idx, :])

        gt_img_embs = torch.stack([item[0] for item in batch], dim=0)
        weights_batch = torch.stack(sampled_weights, dim=0)
        biases_batch = torch.stack(sampled_biases, dim=0)
        betas_batch = torch.stack(sampled_betas, dim=0)
        sample_ids = [item[4] for item in batch]

        return gt_img_embs, weights_batch, biases_batch, betas_batch, sample_ids


