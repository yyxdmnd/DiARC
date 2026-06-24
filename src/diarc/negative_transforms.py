#!/usr/bin/env python3
"""
DPO数据构建 - 不匹配模式消融实验

支持两种不匹配模式（通过 --mismatch-mode 参数选择）：

1. cross_task（跨任务不匹配）：
   - fewshot (6个): 来自当前任务 A（不换）
   - test_input: 来自当前任务 A 的第7个 example（不换）
   - chosen/rejected: 来自其他任务 B 的对应 epoch example
   → chosen/rejected 与 test_input/fewshot 来自完全不同的任务

2. same_task（同任务不同抽样不匹配）：
   - fewshot (6个): 来自当前任务 A（不换）
   - test_input: 来自当前任务 A 的第7个 example（不换）
   - chosen/rejected: 来自任务 A 的额外 example（不在21个里面的）
   → chosen/rejected 与 test_input 都来自 A，但是不同的 example

数据源：re_arc_21 数据集（400个基础任务，每个21个示例，分成3个子任务，每个7个：6 train + 1 test）
变换：
- all = 494种（28种单类 + 466种跨类组合）
- atomic_all = 28种四类原子操作并集（不含跨类组合）
"""

import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import random
from datetime import datetime, timedelta
import time

try:
    from .arc_loader import ArcDataset
    from .paths import DATA_DIR
except ImportError:  # pragma: no cover
    from arc_loader import ArcDataset
    from paths import DATA_DIR

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


# ARC标准颜色映射
ARC_COLORS = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 6), (255, 220, 0),
    (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37)
]


# ==============================================================================
# 四类原子操作定义（与 build_dpo.py 一致）
# ==============================================================================

def get_grid_block_ops():
    """网格分块操作 - 3种"""
    return [
        ('pooldownup2', ['pooldownup2']),
        ('modecenter3', ['modecenter3']),
        ('removesmall', ['removesmall']),
    ]


def get_rigid_shift_ops():
    """刚体位移操作 - 16种"""
    ops = []
    directions = ['U', 'D', 'L', 'R', 'UL', 'UR', 'DL', 'DR']
    percentages = [10, 30]
    for d in directions:
        for p in percentages:
            ops.append((f'shf{d}{p}', [f'shf{d}{p}']))
    return ops


def get_rigid_shift_1d_ops():
    """1D刚体位移操作：只保留左右方向，避免高度为1时上下平移退化成全背景。"""
    ops = []
    directions = ['L', 'R']
    percentages = [10, 30]
    for d in directions:
        for p in percentages:
            ops.append((f'shf{d}{p}', [f'shf{d}{p}']))
    return ops


def get_morphology_ops():
    """形态变换操作 - 7种"""
    return [
        ('erosion', ['erosion']),
        ('dilation', ['dilation']),
        ('skeleton', ['skeleton']),
        ('edge', ['edge']),
        ('boundary', ['boundary']),
        ('convexhull', ['convexhull']),
        ('fillholes', ['fillholes']),
    ]


def get_random_perturbation_ops():
    """随机扰动操作 - 2种"""
    return [
        ('noise5', ['noise5']),
        ('swap10', ['swap10']),
    ]


def get_random_perturbation_1d_ops():
    """1D随机扰动操作：noise5 + 一维相邻边界交换。"""
    return [
        ('noise5', ['noise5']),
        ('swap1d10', ['swap1d10']),
    ]


def get_all_category_ops():
    return {
        'grid_block': get_grid_block_ops(),
        'rigid_shift': get_rigid_shift_ops(),
        'morphology': get_morphology_ops(),
        'random_perturb': get_random_perturbation_ops(),
    }


def generate_all_transforms(category_filter='all'):
    """
    生成变换列表
    
    Args:
        category_filter: 'all' = 全部494种（28单类 + 466跨类组合）
                        'atomic_all' = 28种四类原子操作并集（不含跨类组合）
                        或指定单类: 'grid_block', 'rigid_shift', 'morphology', 'random_perturb'
    """
    if category_filter == 'rigid_shift_1d':
        return [(f'rigid_shift_1d__{name}', tokens, 'rigid_shift_1d') for name, tokens in get_rigid_shift_1d_ops()]
    if category_filter == 'random_perturb_1d':
        return [(f'random_perturb_1d__{name}', tokens, 'random_perturb_1d') for name, tokens in get_random_perturbation_1d_ops()]

    categories = get_all_category_ops()
    category_names = list(categories.keys())
    transforms = []
    
    if category_filter == 'atomic_all':
        for cat_name, ops in categories.items():
            for name, tokens in ops:
                transforms.append((f'{cat_name}__{name}', tokens, cat_name))
        return transforms

    if category_filter != 'all':
        # 单类消融：只用指定类别的操作
        if category_filter not in categories:
            raise ValueError(f"Unknown category: {category_filter}")
        ops = categories[category_filter]
        for name, tokens in ops:
            transforms.append((f'{category_filter}__{name}', tokens, category_filter))
        return transforms
    
    # 全部变换（默认）
    # 单类操作
    for cat_name, ops in categories.items():
        for name, tokens in ops:
            transforms.append((f'{cat_name}__{name}', tokens, cat_name))
    
    # 跨类两两组合
    for i, cat1 in enumerate(category_names):
        for cat2 in category_names[i+1:]:
            ops1 = categories[cat1]
            ops2 = categories[cat2]
            for name1, tokens1 in ops1:
                for name2, tokens2 in ops2:
                    transforms.append((f'{cat1}__{name1}__then__{cat2}__{name2}', tokens1 + tokens2, f'{cat1}+{cat2}'))
            for name2, tokens2 in ops2:
                for name1, tokens1 in ops1:
                    transforms.append((f'{cat2}__{name2}__then__{cat1}__{name1}', tokens2 + tokens1, f'{cat2}+{cat1}'))
    
    return transforms


# ==============================================================================
# 辅助函数
# ==============================================================================

def grid_to_string(grid):
    grid = np.asarray(grid)
    return '\n'.join([''.join([str(c) for c in row]) for row in grid])


def grid_to_key(grid):
    """将网格转为hashable的key用于去重"""
    return tuple(tuple(int(x) for x in row) for row in grid)


def grid_to_image_for_clip(grid, target_size=336):
    """将ARC网格转换为RGB图像（无网格线，与build_dpo.py一致）"""
    grid = np.array(grid, dtype=np.int32)
    h, w = grid.shape
    
    max_dim = max(h, w)
    square_grid = np.zeros((max_dim, max_dim), dtype=np.int32)
    start_h = (max_dim - h) // 2
    start_w = (max_dim - w) // 2
    square_grid[start_h:start_h+h, start_w:start_w+w] = grid
    
    scale = target_size // max_dim
    scale = max(scale, 1)
    
    scaled_size = max_dim * scale
    img = np.zeros((scaled_size, scaled_size, 3), dtype=np.uint8)
    for r in range(max_dim):
        for c in range(max_dim):
            color_idx = min(max(square_grid[r, c], 0), 9)
            img[r*scale:(r+1)*scale, c*scale:(c+1)*scale, :] = ARC_COLORS[color_idx]
    
    if scaled_size < target_size:
        final_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_start = (target_size - scaled_size) // 2
        final_img[pad_start:pad_start+scaled_size, pad_start:pad_start+scaled_size, :] = img
        img = final_img
    elif scaled_size > target_size:
        crop_start = (scaled_size - target_size) // 2
        img = img[crop_start:crop_start+target_size, crop_start:crop_start+target_size, :]
    
    return Image.fromarray(img)


def apply_transform(grid, tokens, color_candidates=None):
    """应用变换到网格"""
    try:
        result = ArcDataset.transform_array(
            np.array(grid, dtype=np.int32),
            tokens,
            apply_shift=True,
            color_candidates=color_candidates,
        )
        return result
    except Exception as e:
        return None


def collect_task_colors(task_data, solution_data=None):
    """收集任务中所有颜色"""
    all_colors = set()
    for ex in task_data.get('train', []):
        all_colors.update(np.unique(np.array(ex['input'])).tolist())
        all_colors.update(np.unique(np.array(ex['output'])).tolist())
    for ex in task_data.get('test', []):
        all_colors.update(np.unique(np.array(ex['input'])).tolist())
        if 'output' in ex:
            all_colors.update(np.unique(np.array(ex['output'])).tolist())
    if solution_data:
        for sol in solution_data:
            all_colors.update(np.unique(np.array(sol)).tolist())
    return sorted(all_colors)


# ==============================================================================
# CLIP计算器（使用transformers库）
# ==============================================================================

class CLIPCalculator:
    """CLIP相似度计算器 - 使用 ViT-L/14@336px"""
    
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Loading CLIP model ViT-L/14@336px on {device}...")
        from transformers import CLIPModel
        import torchvision.transforms as T

        clip_model_path = os.environ.get("POE_CLIP_MODEL_PATH", "openai/clip-vit-large-patch14-336")
        try:
            self.model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load CLIP model in offline mode. "
                "Please make sure the model is already cached locally, or set "
                "POE_CLIP_MODEL_PATH to a local directory containing "
                "openai/clip-vit-large-patch14-336."
            ) from exc
        self.model = self.model.to(device)
        self.model.eval()
        
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.to_tensor = T.ToTensor()
        print(f"Loaded transformers CLIP (offline): {clip_model_path}")
    
    def _prepare_image_tensor(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.normalize(self.to_tensor(image))
    
    def get_embedding(self, image):
        tensor = self._prepare_image_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(pixel_values=tensor)
        return emb.cpu().numpy().flatten()
    
    def get_embeddings_batch(self, images, batch_size=64):
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            tensors = torch.stack([self._prepare_image_tensor(img) for img in batch_images]).to(self.device)
            with torch.no_grad():
                emb = self.model.get_image_features(pixel_values=tensors)
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def similarity(self, emb1, emb2):
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1, emb2))


# ==============================================================================
# 网格变换处理（按网格去重）
# ==============================================================================

def grid_similarity(original_grid, transformed_grid):
    """Cheap fallback similarity used when CLIP is unavailable."""
    original = np.asarray(original_grid)
    transformed = np.asarray(transformed_grid)
    if original.shape == transformed.shape:
        return float(np.mean(original == transformed))

    h = max(original.shape[0], transformed.shape[0])
    w = max(original.shape[1], transformed.shape[1])
    original_pad = np.full((h, w), -1, dtype=np.int32)
    transformed_pad = np.full((h, w), -2, dtype=np.int32)
    original_pad[: original.shape[0], : original.shape[1]] = original
    transformed_pad[: transformed.shape[0], : transformed.shape[1]] = transformed
    return float(np.mean(original_pad == transformed_pad))


def process_grid_transforms(original_grid, transforms, clip_calc=None, color_candidates=None):
    """Apply transforms, deduplicate grids, and rank negatives by similarity."""
    original_emb = None
    if clip_calc is not None:
        original_img = grid_to_image_for_clip(original_grid)
        original_emb = clip_calc.get_embedding(original_img)
    original_key = grid_to_key(original_grid)
    
    # 按网格内容去重
    unique_grids = {}
    for item in transforms:
        name, tokens, combo_type = item if len(item) == 3 else (item[0], item[1], 'unknown')
        
        transformed = apply_transform(original_grid, tokens, color_candidates=color_candidates)
        if transformed is None:
            continue
        
        grid_key = grid_to_key(transformed)
        if grid_key == original_key:
            continue
        
        if grid_key not in unique_grids:
            unique_grids[grid_key] = {
                'grid': transformed, 
                'methods': [name], 
                'tokens': tokens, 
                'combo_type': combo_type
            }
        else:
            unique_grids[grid_key]['methods'].append(name)
    
    if len(unique_grids) == 0:
        return []
    
    grid_keys = list(unique_grids.keys())
    grid_data_list = [unique_grids[k] for k in grid_keys]
    if clip_calc is not None:
        images = [grid_to_image_for_clip(data['grid']) for data in grid_data_list]
        embeddings = clip_calc.get_embeddings_batch(images, batch_size=64)
        original_emb_norm = original_emb / (np.linalg.norm(original_emb) + 1e-8)
    else:
        embeddings = [None] * len(grid_data_list)
        original_emb_norm = None
    
    results = []
    for i, (grid_key, data) in enumerate(zip(grid_keys, grid_data_list)):
        if clip_calc is not None:
            emb = embeddings[i]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(original_emb_norm, emb_norm))
        else:
            sim = grid_similarity(original_grid, data['grid'])
        results.append({
            'methods': data['methods'],
            'tokens': data['tokens'],
            'similarity': sim,
            'grid': data['grid'],
            'combo_type': data['combo_type']
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results


# ==============================================================================
# 增广相关
# ==============================================================================

def generate_16_augmentations():
    """生成16种增广配置（8 D8 × 2 示例顺序）"""
    augmentations = []
    d8_list = []
    for tp_count in range(2):
        for rt_count in range(4):
            d8 = []
            if tp_count > 0:
                d8.append('tp')
            for _ in range(rt_count):
                d8.append('rt')
            d8_list.append(d8)
    
    for d8 in d8_list:
        augmentations.append({'d8': d8, 'shuffle': False})
        augmentations.append({'d8': d8, 'shuffle': True})
    
    return augmentations


def build_instruction(train_data, test_input, fmt_opts, d8_transforms=None, perm_str=None, shuffle_order=None):
    """构建instruction"""
    prefix = fmt_opts['preprompt']
    query_beg = fmt_opts['query_beg']
    reply_beg = fmt_opts['reply_beg']
    reply_end = fmt_opts['reply_end']
    lines_sep = fmt_opts['lines_sep']
    
    if shuffle_order is not None:
        ordered_train_data = [train_data[i] for i in shuffle_order if i < len(train_data)]
    else:
        ordered_train_data = train_data
    
    transforms = []
    if d8_transforms:
        transforms.extend(d8_transforms)
    if perm_str:
        transforms.append('perm' + perm_str)
    
    examples = []
    for train_ex in ordered_train_data:
        input_grid = np.asarray(train_ex['input'])
        output_grid = np.asarray(train_ex['output'])
        
        if transforms:
            input_grid = ArcDataset.transform_array(input_grid, transforms)
            output_grid = ArcDataset.transform_array(output_grid, transforms)
        
        input_str = lines_sep.join(''.join(map(str, row)) for row in input_grid)
        output_str = lines_sep.join(''.join(map(str, row)) for row in output_grid)
        examples.append(f"{query_beg}{input_str}{reply_beg}{output_str}{reply_end}")
    
    test_input_grid = np.asarray(test_input)
    if transforms:
        test_input_grid = ArcDataset.transform_array(test_input_grid, transforms)
    test_input_str = lines_sep.join(''.join(map(str, row)) for row in test_input_grid)
    
    instruction = prefix + ''.join(examples) + f"{query_beg}{test_input_str}{reply_beg}"
    return instruction, transforms


# ==============================================================================
# 主处理逻辑
# ==============================================================================

def process_task_mismatch_v2(base_id, epoch, rearc_data, transforms, clip_calc, fmt_opts, 
                              augmentation_list, top_k, no_augment, train_size,
                              mismatch_mode='normal', num_epochs=3, mix_ratio=0.5):
    """
    处理单个任务 - 消融实验
    所有数据都来自原始 re-arc
    
    六种模式：
    1. normal: chosen/rejected 来自当前组的第7个 example output（匹配，baseline）
    2. same_task: chosen/rejected 来自同一任务 A 的额外 example（不匹配）
    3. cross_task: chosen/rejected 来自随机选择的其他任务 B（不匹配）
    4. mixed: cross_task + same_task 混合（比例由 mix_ratio 控制）
    5. mixed_normal: normal + same_task 混合（比例由 mix_ratio 控制）
    6. dual_random: chosen 来自随机任务 B，rejected 来自另一个随机任务 C（无变换）
    
    共同点：
    - fewshot (6个): 来自当前任务 A 的当前组
    - test_input: 来自当前任务 A 的当前组第7个 example 的 input
    """
    samples = []
    group_size = train_size + 1  # 6 + 1 = 7
    
    # 获取当前任务的所有 examples
    if base_id not in rearc_data:
        return []
    all_examples = rearc_data[base_id]
    
    # 计算当前组的范围
    start_idx = epoch * group_size
    end_idx = start_idx + group_size
    if end_idx > len(all_examples):
        return []
    
    # fewshot 和 test_input 来自当前组
    current_group = all_examples[start_idx:end_idx]
    train_examples = current_group[:train_size]  # 前6个作为fewshot
    test_example = current_group[train_size]      # 第7个作为test
    test_input = test_example['input']
    
    # 根据 mismatch_mode 选择 chosen/rejected 的来源
    chosen_output = None
    source_info = None
    
    # 用于确定性随机的种子
    task_key = f"{base_id}_{epoch}"
    task_seed = hash(task_key) % (2**32)
    
    # mixed/mixed_normal 模式：根据 task_key 确定性地选择模式
    actual_mode = mismatch_mode
    if mismatch_mode == 'mixed':
        # 用 task_seed 生成确定性比例值，根据 mix_ratio 选择
        ratio_value = (task_seed % 1000) / 1000.0
        actual_mode = 'cross_task' if ratio_value < mix_ratio else 'same_task'
    elif mismatch_mode == 'mixed_normal':
        # normal + same_task 混合
        ratio_value = (task_seed % 1000) / 1000.0
        actual_mode = 'normal' if ratio_value < mix_ratio else 'same_task'
    
    if actual_mode == 'normal':
        # 正常模式：chosen 来自当前组的第7个 example（匹配 test_input）
        chosen_output = np.array(test_example['output'])
        source_info = f"{base_id}[{start_idx + train_size}] (matched)"
        
    elif actual_mode == 'cross_task':
        # 跨任务模式：随机选择其他任务 B
        other_base_ids = sorted([bid for bid in rearc_data.keys() if bid != base_id])
        if other_base_ids:
            # 每个 epoch 随机选择不同的任务 B（基于 task_key 确定性随机）
            rng = np.random.RandomState(task_seed)
            mismatch_base_id = other_base_ids[rng.randint(len(other_base_ids))]
            
            mismatch_examples = rearc_data[mismatch_base_id]
            if mismatch_examples:
                # 随机选择一个 example
                example_idx = rng.randint(len(mismatch_examples))
                chosen_output = np.array(mismatch_examples[example_idx]['output'])
                source_info = f"{mismatch_base_id}[{example_idx}] (cross)"
                
    elif actual_mode == 'same_task':
        # 同任务模式：chosen 来自任务 A 的额外 example
        # 额外 example 从 num_epochs * group_size 之后开始
        extra_start_idx = num_epochs * group_size
        extra_idx = extra_start_idx + epoch
        
        if len(all_examples) > extra_idx:
            chosen_output = np.array(all_examples[extra_idx]['output'])
            source_info = f"{base_id}[{extra_idx}] (same)"
        elif len(all_examples) > extra_start_idx:
            chosen_output = np.array(all_examples[extra_start_idx]['output'])
            source_info = f"{base_id}[{extra_start_idx}] (same)"
    
    elif actual_mode == 'dual_random':
        # dual_random 模式：chosen 来自任务 B，rejected 来自任务 C（无变换）
        # 这个模式的处理逻辑完全不同，直接生成样本并返回
        other_base_ids = sorted([bid for bid in rearc_data.keys() if bid != base_id])
        if len(other_base_ids) < 2:
            return []
        
        rng = np.random.RandomState(task_seed)
        
        # 随机选择两个不同的任务 B 和 C
        selected_indices = rng.choice(len(other_base_ids), size=2, replace=False)
        task_b_id = other_base_ids[selected_indices[0]]
        task_c_id = other_base_ids[selected_indices[1]]
        
        task_b_examples = rearc_data[task_b_id]
        task_c_examples = rearc_data[task_c_id]
        
        if not task_b_examples or not task_c_examples:
            return []
        
        # 为每个 rank 生成一个样本
        n_examples = len(train_examples)
        for rank in range(top_k):
            sample_seed = hash((task_key, rank)) % (2**32)
            sample_rng = np.random.RandomState(sample_seed)
            
            # 从 B 随机选一个作为 chosen
            b_idx = sample_rng.randint(len(task_b_examples))
            chosen_output = np.array(task_b_examples[b_idx]['output'])
            
            # 从 C 随机选一个作为 rejected
            c_idx = sample_rng.randint(len(task_c_examples))
            rejected_output = np.array(task_c_examples[c_idx]['output'])
            
            if no_augment:
                instruction, _ = build_instruction(train_examples, test_input, fmt_opts)
                sample = {
                    "instruction": instruction,
                    "input": "",
                    "chosen": grid_to_string(chosen_output),
                    "rejected": grid_to_string(rejected_output),
                    "rank": rank,
                    "combo_type": "dual_random",
                    "similarity": 0.0,  # 无变换，无相似度
                    "task_id": f"{base_id}_{epoch}",
                    "source": f"chosen:{task_b_id}[{b_idx}], rejected:{task_c_id}[{c_idx}]",
                    "mode": "dual_random",
                }
                samples.append(sample)
            else:
                np.random.seed(sample_seed)
                random.seed(sample_seed)
                
                aug_idx = np.random.randint(len(augmentation_list))
                aug_config = augmentation_list[aug_idx]
                perm_str = ArcDataset.rand_perm(10, '', keep_zero=False)
                
                if aug_config['shuffle']:
                    shuffle_order = np.random.permutation(n_examples).tolist()
                else:
                    shuffle_order = None
                
                instruction, aug_transforms = build_instruction(
                    train_examples, test_input, fmt_opts,
                    d8_transforms=aug_config['d8'],
                    perm_str=perm_str,
                    shuffle_order=shuffle_order
                )
                
                chosen_grid = ArcDataset.transform_array(chosen_output, aug_transforms)
                rejected_grid = ArcDataset.transform_array(rejected_output, aug_transforms)
                
                sample = {
                    "instruction": instruction,
                    "input": "",
                    "chosen": grid_to_string(chosen_grid),
                    "rejected": grid_to_string(rejected_grid),
                    "rank": rank,
                    "combo_type": "dual_random",
                    "similarity": 0.0,
                    "task_id": f"{base_id}_{epoch}",
                    "source": f"chosen:{task_b_id}[{b_idx}], rejected:{task_c_id}[{c_idx}]",
                    "mode": "dual_random",
                }
                samples.append(sample)
        
        return samples  # dual_random 模式直接返回
    
    if chosen_output is None:
        return []
    
    # 收集颜色
    color_candidates = set(np.unique(chosen_output).tolist())
    
    # 对 chosen_output 应用 494 变换，计算 CLIP 相似度
    results = process_grid_transforms(chosen_output, transforms, clip_calc, color_candidates)
    
    if len(results) == 0:
        return []
    
    # 过滤相似度为1的
    unique_results = [r for r in results if abs(r['similarity'] - 1.0) >= 1e-6]
    dpo_results = unique_results[:top_k]
    
    n_examples = len(train_examples)
    task_key = f"{base_id}_{epoch}"  # 用于生成确定性种子
    
    for rank, r in enumerate(dpo_results):
        if no_augment:
            instruction, _ = build_instruction(train_examples, test_input, fmt_opts)
            sample = {
                "instruction": instruction,
                "input": "",
                "chosen": grid_to_string(chosen_output),
                "rejected": grid_to_string(r['grid']),
                "rank": rank,
                "combo_type": r.get('combo_type', 'unknown'),
                "similarity": r['similarity'],
                "task_id": f"{base_id}_{epoch}",
                "source": source_info,
                "mode": actual_mode,
            }
            samples.append(sample)
        else:
            # 基于 task_key 和 rank 设置确定性种子
            sample_seed = hash((task_key, rank)) % (2**32)
            np.random.seed(sample_seed)
            random.seed(sample_seed)
            
            aug_idx = np.random.randint(len(augmentation_list))
            aug_config = augmentation_list[aug_idx]
            
            perm_str = ArcDataset.rand_perm(10, '', keep_zero=False)
            
            if aug_config['shuffle']:
                shuffle_order = np.random.permutation(n_examples).tolist()
            else:
                shuffle_order = None
            
            instruction, aug_transforms = build_instruction(
                train_examples, test_input, fmt_opts,
                d8_transforms=aug_config['d8'],
                perm_str=perm_str,
                shuffle_order=shuffle_order
            )
            
            chosen_grid = ArcDataset.transform_array(chosen_output, aug_transforms)
            rejected_grid = ArcDataset.transform_array(r['grid'], aug_transforms)
            
            sample = {
                "instruction": instruction,
                "input": "",
                "chosen": grid_to_string(chosen_grid),
                "rejected": grid_to_string(rejected_grid),
                "rank": rank,
                "combo_type": r.get('combo_type', 'unknown'),
                "similarity": r['similarity'],
                "task_id": f"{base_id}_{epoch}",
                "source": source_info,
                "mode": actual_mode,
            }
            samples.append(sample)
    
    return samples


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='跨任务DPO数据构建 - 故意不匹配模式（消融实验）')
    parser.add_argument('--rearc-path',
                       default=str(DATA_DIR / 're_arc'),
                       help='原始re-arc数据集路径（所有数据来源）')
    parser.add_argument('--output-dir',
                       default=str(DATA_DIR / 'dpo_cross_task'),
                       help='输出目录')
    parser.add_argument('--top-k', type=int, default=16, help='取相似度前k个')
    parser.add_argument('--no-augment', action='store_true', help='不应用16种增广')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--train-size', type=int, default=6, help='每组的fewshot数量')
    parser.add_argument('--num-epochs', type=int, default=3, help='每个任务的epoch数（组数）')
    parser.add_argument('--max-tasks', type=int, default=None, help='最大base任务数（测试用）')
    parser.add_argument('--mismatch-mode', choices=['normal', 'same_task', 'cross_task', 'mixed', 'mixed_normal', 'dual_random'], 
                       default='normal',
                       help='模式: normal=正常匹配, same_task=同任务不匹配, cross_task=跨任务不匹配, mixed=cross+same混合, mixed_normal=normal+same混合, dual_random=chosen和rejected来自不同随机任务')
    parser.add_argument('--mix-ratio', type=float, default=0.5,
                       help='mixed/mixed_normal模式中第一种模式的比例 (0.0-1.0，默认0.5，即50%%)')
    parser.add_argument('--save-every', type=int, default=10,
                       help='每处理N个任务保存一次checkpoint（默认10）')
    parser.add_argument('--resume', action='store_true',
                       help='从上次checkpoint恢复')
    parser.add_argument('--transform-category', 
                       choices=['all', 'atomic_all', 'grid_block', 'rigid_shift', 'morphology', 'random_perturb'],
                       default='all',
                       help='变换类别: all=全部494种, atomic_all=28种原子操作并集, 或单类消融(grid_block/rigid_shift/morphology/random_perturb)')
    
    args = parser.parse_args()
    
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"\n开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置随机种子（确保可复现）
    print(f"Setting random seed: {args.seed}")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 格式化选项
    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n</s>',
        lines_sep='\n',
    )
    
    # 生成变换（支持单类消融）
    transforms = generate_all_transforms(category_filter=args.transform_category)
    print(f"Transform category: {args.transform_category}")
    print(f"Total transforms: {len(transforms)}")
    
    # 16种增广配置
    augmentation_list = generate_16_augmentations()
    print(f"Augmentation variants: {len(augmentation_list)}")
    
    # 加载原始re-arc数据（所有数据来源）
    print(f"\nLoading original re-arc data from {args.rearc_path}...")
    rearc_data = {}
    rearc_tasks_path = os.path.join(args.rearc_path, 'tasks')
    for task_file in sorted(os.listdir(rearc_tasks_path)):
        if task_file.endswith('.json'):
            base_id = task_file.replace('.json', '')
            with open(os.path.join(rearc_tasks_path, task_file), 'r') as f:
                rearc_data[base_id] = json.load(f)
    print(f"Loaded {len(rearc_data)} base tasks from original re-arc")
    
    # 生成任务列表：每个 base task 生成 num_epochs 个子任务
    # 格式: (base_id, epoch) -> 对应 examples[epoch*7 : (epoch+1)*7]
    all_task_keys = []
    base_ids = sorted(rearc_data.keys())
    if args.max_tasks:
        base_ids = base_ids[:args.max_tasks]
    
    group_size = args.train_size + 1  # 6 fewshot + 1 test = 7
    for base_id in base_ids:
        examples = rearc_data[base_id]
        # 计算可以生成多少个epoch
        max_epochs = min(args.num_epochs, len(examples) // group_size)
        for epoch in range(max_epochs):
            all_task_keys.append((base_id, epoch))
    
    print(f"Generated {len(all_task_keys)} task keys ({len(base_ids)} base tasks × up to {args.num_epochs} epochs)")
    
    # 打印配置
    print("=" * 60)
    print("配置:")
    print(f"  模式: {args.mismatch_mode}")
    print(f"  数据源: {args.rearc_path} (原始re-arc)")
    print(f"  每组: {args.train_size} fewshot + 1 test = {group_size} examples")
    print(f"  每个任务 {args.num_epochs} 个 epoch")
    transform_desc = {
        'all': '494种（28单类 + 466跨类组合）',
        'atomic_all': '28种四类原子操作并集（不含跨类组合）',
        'grid_block': '3种 grid_block 单类操作',
        'rigid_shift': '16种 rigid_shift 单类操作',
        'morphology': '7种 morphology 单类操作',
        'random_perturb': '2种 random_perturb 单类操作',
    }
    print(f"  变换集合: {transform_desc.get(args.transform_category, args.transform_category)}")
    rejected_desc = (
        "chosen 的 494 变换后取"
        if args.transform_category == "all"
        else "chosen 的当前变换集合后取"
    )
    if args.mismatch_mode == 'normal':
        print(f"  数据流 (baseline):")
        print(f"    - fewshot/test_input: 任务 A 的当前组")
        print(f"    - chosen: 当前组第7个 example 的 output（匹配 test_input）")
        print(f"    - rejected: {rejected_desc} top-{args.top_k}")
    elif args.mismatch_mode == 'same_task':
        print(f"  数据流 (同任务不匹配):")
        print(f"    - fewshot/test_input: 任务 A 的当前组")
        print(f"    - chosen: 任务 A 的额外 example（不匹配）")
        print(f"    - rejected: {rejected_desc} top-{args.top_k}")
    elif args.mismatch_mode == 'cross_task':
        print(f"  数据流 (跨任务不匹配):")
        print(f"    - fewshot/test_input: 任务 A 的当前组")
        print(f"    - chosen: 随机选择的任务 B 的 example（不匹配）")
        print(f"    - rejected: {rejected_desc} top-{args.top_k}")
    elif args.mismatch_mode == 'mixed':
        print(f"  数据流 (混合模式 cross+same):")
        print(f"    - fewshot/test_input: 任务 A 的当前组")
        print(f"    - chosen: {args.mix_ratio*100:.0f}% cross_task + {(1-args.mix_ratio)*100:.0f}% same_task")
        print(f"    - rejected: {rejected_desc} top-{args.top_k}")
    elif args.mismatch_mode == 'mixed_normal':
        print(f"  数据流 (混合模式 normal+same):")
        print(f"    - fewshot/test_input: 任务 A 的当前组")
        print(f"    - chosen: {args.mix_ratio*100:.0f}% normal + {(1-args.mix_ratio)*100:.0f}% same_task")
        print(f"    - rejected: {rejected_desc} top-{args.top_k}")
    elif args.mismatch_mode == 'dual_random':
        print(f"  数据流 (双随机模式):")
        print(f"    - fewshot/test_input: 任务 A 的当前组")
        print(f"    - chosen: 随机任务 B 的 example")
        print(f"    - rejected: 另一个随机任务 C 的 example（无输出变换）")
    print(f"  CLIP: ViT-L/14@336px")
    print(f"  增广: 16种 (8 D8 × 2 示例顺序 + 颜色置换)")
    print(f"  Seed: {args.seed}")
    print("=" * 60)
    
    # 创建CLIP计算器
    clip_calc = CLIPCalculator(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # checkpoint 文件路径
    checkpoint_file = os.path.join(args.output_dir, 'checkpoint.json')
    
    # 尝试恢复 checkpoint
    all_samples = []
    processed_keys = set()
    stats = {'total_tasks': 0, 'skipped_tasks': 0, 'tasks_with_less_than_k': 0}
    
    if args.resume and os.path.exists(checkpoint_file):
        print(f"\n从checkpoint恢复: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
        # checkpoint 中的 key 是字符串格式 "base_id,epoch"
        processed_keys = set(tuple(k.split(',')) if isinstance(k, str) else tuple(k) 
                            for k in ckpt.get('processed_keys', []))
        stats = ckpt.get('stats', stats)
        samples_file = os.path.join(args.output_dir, 'samples_checkpoint.jsonl')
        if os.path.exists(samples_file):
            with open(samples_file, 'r') as f:
                for line in f:
                    all_samples.append(json.loads(line))
        print(f"已恢复 {len(processed_keys)} 个任务, {len(all_samples)} 个样本")
    
    # 过滤出未处理的任务
    remaining_keys = [k for k in all_task_keys if k not in processed_keys]
    print(f"待处理任务: {len(remaining_keys)} / {len(all_task_keys)}")
    
    # 处理所有任务
    for i, (base_id, epoch) in enumerate(tqdm(remaining_keys, desc="Processing tasks")):
        samples = process_task_mismatch_v2(
            base_id, epoch, rearc_data, transforms, clip_calc, fmt_opts,
            augmentation_list, args.top_k, args.no_augment, args.train_size,
            mismatch_mode=args.mismatch_mode, num_epochs=args.num_epochs,
            mix_ratio=args.mix_ratio
        )
        
        if samples:
            stats['total_tasks'] += 1
            if len(samples) < args.top_k:
                stats['tasks_with_less_than_k'] += 1
            all_samples.extend(samples)
        else:
            stats['skipped_tasks'] += 1
        
        processed_keys.add((base_id, epoch))
        
        # 每 save_every 个任务保存一次 checkpoint
        if (i + 1) % args.save_every == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'processed_keys': [f"{b},{e}" for b, e in processed_keys],
                    'stats': stats,
                    'last_save': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }, f)
            samples_file = os.path.join(args.output_dir, 'samples_checkpoint.jsonl')
            with open(samples_file, 'w') as f:
                for s in all_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')
            print(f"\n[Checkpoint] 已保存 {len(processed_keys)}/{len(all_task_keys)} 任务, {len(all_samples)} 样本")
    
    stats['total_samples'] = len(all_samples)
    
    # 按rank分组保存
    num_stages = args.top_k // 4
    print(f"\nGenerating {num_stages} stage files...")
    for stage in range(1, num_stages + 1):
        stage_ranks = [(stage - 1) * 4 + i for i in range(4)]
        stage_samples = [s for s in all_samples if s['rank'] in stage_ranks]
        
        output_file = os.path.join(args.output_dir, f'arc_dpo_data_stage_{stage}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in stage_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  Stage {stage} (rank {stage_ranks}): {len(stage_samples)} samples")
    
    # 保存完整文件
    output_file = os.path.join(args.output_dir, 'arc_dpo_data_all.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 计算总时间
    end_time = time.time()
    end_datetime = datetime.now()
    elapsed_seconds = end_time - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
    
    print(f"\n{'='*60}")
    print(f"数据生成完成!")
    print(f"{'='*60}")
    print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总用时: {elapsed_str}")
    print(f"{'='*60}")
    print(f"统计:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Skipped tasks: {stats['skipped_tasks']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Tasks with < {args.top_k} samples: {stats['tasks_with_less_than_k']}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")
    
    # 保存生成信息
    mode_descriptions = {
        'normal': "Baseline：chosen/rejected匹配test_input",
        'same_task': "同任务不匹配：chosen来自任务A的额外example",
        'cross_task': "跨任务不匹配：chosen来自随机选择的任务B",
        'mixed': f"混合模式：{args.mix_ratio*100:.0f}% cross_task + {(1-args.mix_ratio)*100:.0f}% same_task",
        'mixed_normal': f"混合模式：{args.mix_ratio*100:.0f}% normal + {(1-args.mix_ratio)*100:.0f}% same_task",
        'dual_random': "双随机模式：chosen来自任务B，rejected来自任务C（无输出变换）",
    }
    
    generation_info = {
        "mode": args.mismatch_mode,
        "mix_ratio": args.mix_ratio if args.mismatch_mode in ['mixed', 'mixed_normal'] else None,
        "description": mode_descriptions.get(args.mismatch_mode, ""),
        "data_source": args.rearc_path,
        "data_flow": {
            "fewshot": "任务A当前组的前N个example",
            "test_input": "任务A当前组的第N+1个example的input",
            "chosen": "根据mode决定",
            "rejected": "chosen的当前变换集合后取top-k",
        },
        "train_size": args.train_size,
        "group_size": args.train_size + 1,
        "num_epochs": args.num_epochs,
        "clip_model": "ViT-L/14@336px",
        "augmentation": "16种 (8 D8 × 2 示例顺序 + 颜色置换)",
        "transform_category": args.transform_category,
        "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        "elapsed_seconds": elapsed_seconds,
        "seed": args.seed,
        "total_tasks": stats['total_tasks'],
        "total_samples": stats['total_samples'],
        "top_k": args.top_k,
    }
    with open(os.path.join(args.output_dir, 'generation_info.json'), 'w') as f:
        json.dump(generation_info, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
