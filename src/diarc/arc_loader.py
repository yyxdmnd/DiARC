# Copyright 2024-2025 Daniel Franzen, Jan Disselhoff and David Hartmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import itertools
import json
import hashlib
import numpy as np
from numpy.random import randint
from glob import glob
from tqdm import tqdm
from collections import OrderedDict


class ArcDataset(object):
    def __init__(self, challenge, solutions={}, keys=None, is_fake=False, is_orig=False):
        if keys is None:
            self.keys = []
            for k, v in challenge.items():
                reply_num = len(v['test'])
                self.keys.extend([f'{k}_{i}' for i in range(reply_num)] if reply_num else [k])
            self.keys = sorted(self.keys)
        else:
            self.keys = [k for k in keys]
        base_keys = set(map(self.get_base_key, self.keys))
        self.challenge = {k: challenge[k] for k in base_keys}
        self.solutions = {k: solutions[k] for k in base_keys if k in solutions}
        self.is_orig = is_fake
        self.is_orig = is_orig

    @classmethod
    def load_from_json(cls, challenges_file):  # for loading challenges in kaggle json arc dataset format
        with open(challenges_file) as f:
            challenge = f.read()
        return cls(
            challenge=json.loads(challenge),
            is_fake=hashlib.md5(challenge.encode('utf-8')).hexdigest().lower() == 'a6b7dac3cab03abf2eb333e16610d6dc',
            is_orig=True,
        )

    def load_solutions(self, solutions_file):  # for loading solutions in kaggle json arc dataset format
        with open(solutions_file) as f: solutions = f.read()
        data = json.loads(solutions)
        solutions = {k: data[k] for k in self.challenge}
        return self.__class__(keys=self.keys, challenge=self.challenge, solutions=solutions, is_orig=self.is_orig)

    # loader for Michael Hodel's ReArc https://github.com/neoneye/arc-dataset-collection
    @classmethod
    def load_from_rearc(cls, path, n, sizes, seed, shuffle=True, remove_easiest=0):  # loader for ReArc
        np.random.seed(seed)
        keys = [[] for _ in range(n)]
        challenge = {}
        solutions = {}
        sizes = list(sizes)

        with open(os.path.join(path, 'metadata.json')) as f:
            metadata = json.load(f)

        for key in tqdm(sorted(metadata.keys()), desc="load dataset 're-arc'"):
            with open(os.path.join(path, 'tasks', f'{key}.json')) as f:
                tasks = json.load(f)

            if remove_easiest:
                tasks = zip(tasks, metadata[key]['pso_difficulties'])
                tasks = sorted(tasks, key=lambda x: x[1])
                tasks = tasks[int(len(tasks) * remove_easiest):]
                tasks = [x[0] for x in tasks]

            tasks = np.random.permutation(tasks).tolist()

            next_sizes = []
            for epoch in range(n):
                if not len(next_sizes):
                    next_sizes = np.random.permutation(sizes).tolist()
                next_size_with_test = 1 + next_sizes.pop()
                base_key = f'rearc-{key}{epoch:02x}'
                keys[epoch].append(f'{base_key}_0')
                challenge[base_key] = {'train': [], 'test': []}
                solutions[base_key] = reply = []
                for _ in range(next_size_with_test):
                    if not len(tasks):
                        raise RuntimeError('Not enough examples - generate more re-arc examples or reduce epochs.')
                    challenge[base_key]['train'].append({k: v for k, v in tasks.pop().items()})
                challenge[base_key]['test'].append(challenge[base_key]['train'].pop())
                solutions[base_key].append(challenge[base_key]['test'][-1].pop('output'))

        if shuffle:
            keys = [np.random.permutation(epoch) for epoch in keys]
        keys = [k for epoch in keys for k in epoch]
        return cls(keys=keys, challenge=challenge, solutions=solutions, is_orig=True)

    # loader for neoneye's format, as used in https://github.com/neoneye/arc-dataset-collection
    # also supports the flat `data/*.json` layout used by some ARC-like benchmarks
    @classmethod
    def load_from_neoneye(cls, path):
        patterns = [
            os.path.join(path, 'data', '*', '*.json'),
            os.path.join(path, 'data', '*.json'),
        ]
        files = set()
        for pattern in patterns:
            files.update(glob(pattern))
        for i in itertools.count():
            updated = [fn for fn in files if fn.endswith(f'_v{i + 1}.json')]
            if not updated: break
            for fn in updated:
                files.remove(fn.replace(f'_v{i + 1}.json', ('.json' if i == 1 else f'_v{i}.json')))
        assert len(files), f"No files found for pattern '{pattern}'."
        challenge = {}
        solutions = {}
        assert len(files), 'no files found'
        for fn in tqdm(files, desc=f"load dataset '{os.path.split(path)[-1]}'"):
            with open(fn) as f:
                key = cls.base_key_replace_invalid_chars(os.path.split(fn)[-1].replace('.json', ''))
                challenge[key] = json.load(f)
                solutions[key] = [test_case.pop('output') for test_case in challenge[key]['test']]
        return cls(challenge=challenge, solutions=solutions, is_orig=True)

    @classmethod
    def load_sudoku_csv(cls, path, examples_per_challenge=3, num_challenges=10, diff_lim=None, diff_lim_lower=None, min_indx=None, max_indx=None):
        import pandas as pd
        df = pd.read_csv(path)

        if max_indx is not None:
            df = df[:max_indx]

        if min_indx is not None:
            df = df[min_indx:]

        if diff_lim is not None:
            df = df[df['difficulty'] < diff_lim]

        if diff_lim_lower is not None:
            df = df[df['difficulty'] > diff_lim_lower]

        print(f"Loaded {len(df)} sudoku puzzles.")

        df['puzzle'] = df['puzzle'].apply(lambda x: x.replace(".", "0"))

        challenges = {}
        solutions = {}
        difficulty = {}

        for j in range(num_challenges):
            examples = []
            for i in range(j * (examples_per_challenge + 1), (j + 1) * (examples_per_challenge + 1)):
                row = df.iloc[i]
                puzzle = np.array([int(x) for x in row['puzzle']]).reshape(9, 9)
                solution = np.array([int(x) for x in row['solution']]).reshape(9, 9)
                examples.append({"input": puzzle, 'output': solution})

            id = f"sudoku{df.iloc[i]['id']}"
            last = examples.pop()
            solutions[id] = [last.pop('output')]
            challenges[id] = {'train': examples, 'test': [last]}
            difficulty[id] = row['difficulty']

        return_class = cls(challenge=challenges, solutions=solutions, is_orig=True)
        return_class.difficulty = difficulty

        return return_class

    def change_keys(self, keys):
        return self.__class__(challenge=self.challenge, solutions=self.solutions, keys=keys, is_orig=self.is_orig)

    def split(self, n, split_seed, **kwargs):
        assert self.is_orig, 'Must be run on original dataset.'
        keys = sorted(self.challenge.keys())
        if split_seed == 'len':
            keys = self.sort_keys_by_len(keys=keys, **kwargs)
        else:
            assert isinstance(split_seed, int)
            assert not kwargs
            np.random.seed(split_seed)
            keys = np.random.permutation(keys)
        split_datasets = []
        for new_keys in np.array_split(keys, n):
            new_challenge = {k: self.challenge[k] for k in new_keys}
            split_datasets.append(self.__class__(challenge=new_challenge, solutions=self.solutions, is_orig=True))
        return split_datasets

    def remove_test_data(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {k: {'train': v['train'], 'test': []} for k, v in self.challenge.items()}
        return self.__class__(challenge=new_challenge)

    @staticmethod
    def base_key_replace_invalid_chars(base_key):
        return base_key.replace('_', '-').replace('.', '-')

    @staticmethod
    def get_base_key_and_reply_num(key):
        key_num = key.split('.', 1)[0]
        base_key, reply_num = key_num.split('_') if '_' in key_num else (key_num, -1)
        return base_key, int(reply_num)

    @classmethod
    def get_base_key(cls, key):
        return cls.get_base_key_and_reply_num(key)[0]

    def grouped_keys(self):
        grouped_keys = OrderedDict()
        for key in self.keys:
            base_key, reply_num = self.get_base_key_and_reply_num(key)
            if base_key not in grouped_keys:
                grouped_keys[base_key] = []
            while len(grouped_keys[base_key])<=reply_num:
                grouped_keys[base_key].append([])
            grouped_keys[base_key][reply_num].append(key)
        return grouped_keys

    def move_test_to_train(self):
        assert self.is_orig, 'Must be run on original dataset.'
        new_challenge = {}
        for k, v in self.challenge.items():
            new_challenge[k] = {
                'train': v['train'] + [{**t, 'output': self.solutions[k][i]} for i, t in enumerate(v['test'])],
                'test': []
            }
        return self.__class__(challenge=new_challenge, is_orig=self.is_orig)

    @staticmethod
    def permute_array(a, descriptor, invert=False):
        permutation = [int(i) for i in descriptor if str(i).isdigit()]
        assert sorted(permutation) == list(range(10))
        a = np.asarray(a)
        assert a.ndim == 2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
        return a

    @classmethod
    def transform_array(cls, array, transforms, apply_perm=True, apply_shift=False, invert=False, color_candidates=None):
        """应用变换到数组
        
        Args:
            array: 输入数组
            transforms: 变换列表
            apply_perm: 是否应用颜色置换
            apply_shift: 是否应用位移
            invert: 是否反转变换
            color_candidates: 颜色候选集（用于noise操作，默认从array获取）
        """
        if array is None: return None
        array = np.asarray(array)
        if invert: transforms = transforms[::-1]
        for tf in transforms:
            # geometric transpose
            if tf == 'tp':
                array = np.swapaxes(array, 0, 1)
            # 90-degree rotation (invert applies opposite rotation)
            if tf == 'rt':
                array = np.rot90(np.rot90(np.rot90(array)) if invert else array)
            if isinstance(tf, str) and tf == 'modecenter3':
                array = cls.mode_center_pool(array, kernel_size=3)
            # 马赛克化
            if isinstance(tf, str) and tf == 'pooldownup2':
                if invert:
                    continue
                array = cls.pool_down_up(array, kernel_size=2)
            # 刚体平移
            if apply_shift and isinstance(tf, str) and tf.startswith('shf'):
                array = cls.shift_array(array, tf, invert=invert)
            # 颜色置换
            if apply_perm and tf.startswith('perm'):
                array = cls.permute_array(array, tf, invert=invert)
            # 形态变换
            if isinstance(tf, str) and tf == 'erosion':
                if not invert:
                    array = cls.erosion(array)
            if isinstance(tf, str) and tf == 'dilation':
                if not invert:
                    array = cls.dilation(array)
            # 结构抽离
            if isinstance(tf, str) and tf == 'edge':
                if not invert:
                    array = cls.edge_detection(array)
            if isinstance(tf, str) and tf == 'boundary':
                if not invert:
                    array = cls.boundary_trace(array)
            if isinstance(tf, str) and tf == 'skeleton':
                if not invert:
                    array = cls.skeletonize(array)
            # 区域拓扑
            if isinstance(tf, str) and tf == 'removesmall':
                if not invert:
                    array = cls.remove_small_objects(array, min_size=2)
            if isinstance(tf, str) and tf == 'fillholes':
                if not invert:
                    array = cls.fill_holes(array)
            if isinstance(tf, str) and tf == 'convexhull':
                if not invert:
                    array = cls.convex_hull(array)
            # 噪点处理
            if isinstance(tf, str) and tf.startswith('noise'):
                if not invert:
                    ratio = int(tf[5:]) / 100.0 if len(tf) > 5 else 0.05
                    array = cls.salt_pepper_noise(array, ratio=ratio, seed=42, color_candidates=color_candidates)
            if isinstance(tf, str) and tf.startswith('swap1d'):
                if not invert:
                    ratio = int(tf[6:]) / 100.0 if len(tf) > 6 else 0.10
                    array = cls.local_swap_1d(array, ratio=ratio, seed=42)
            elif isinstance(tf, str) and tf.startswith('swap'):
                if not invert:
                    ratio = int(tf[4:]) / 100.0 if len(tf) > 4 else 0.05
                    array = cls.local_swap(array, ratio=ratio, seed=42)

        return array

    @classmethod
    def fmt_array(cls, array, lines_sep, tf=None):
        if tf is not None:
            array = cls.transform_array(array, tf, apply_shift=True)
        return lines_sep.join(''.join(map(str, row)) for row in array)

    @classmethod
    def fmt_input(cls, array, query_beg, reply_beg, **kwargs):
        return query_beg + cls.fmt_array(array, **kwargs) + reply_beg

    @classmethod
    def fmt_output(cls, array, reply_end, **kwargs):
        return cls.fmt_array(array, **kwargs) + reply_end

    @classmethod
    def fmt_train(cls, train_ex, preprompt, query_beg, reply_beg, reply_end, **kwargs):
        examples = [cls.fmt_input(x['input'], query_beg, reply_beg, **kwargs) +
                    cls.fmt_output(x['output'], reply_end, **kwargs) for x in train_ex]
        return preprompt + ''.join(examples)

    def fmt_task(self, key, preprompt, query_beg, reply_beg, reply_end, reply=True, **kwargs):
        key_num, *tf = key.split('.')
        base_key, reply_num = self.get_base_key_and_reply_num(key_num)
        data_train = self.challenge[base_key]['train']
        data_query = self.challenge[base_key]['test']
        if reply is True:
            reply = self.solutions[base_key][reply_num] if base_key in self.solutions and reply_num >= 0 else None
        elif reply is not None:
            assert reply_num >= 0
        for t in tf:
            if t.startswith('ex'):
                data_train = [data_train[int(i)] for i in t[2:].split('-')]
        ret = dict(key=key)
        ret['train'] = self.fmt_train(data_train, preprompt, query_beg, reply_beg, reply_end, tf=tf, **kwargs)
        ret['query'] = self.fmt_input(data_query[reply_num]['input'], query_beg, reply_beg, tf=tf, **kwargs) if reply_num >= 0 else ''
        ret['input'] = ret['train'] + ret['query'] if reply_num >= 0 else ''
        if reply is not None:
            ret['reply'] = self.fmt_output(reply, reply_end, tf=tf, **kwargs)
        ret['text'] = ret['train'] + (ret['query'] + ret['reply'] if reply is not None else '')
        return ret

    def get_task(self, key, max_tokens=None, len_name=None, **kwargs):
        while True:
            fmt = self.fmt_task(key, **kwargs)
            if max_tokens is None or self.count_tokens(fmt[len_name]) <= max_tokens:
                break
            if not key.split('.')[-1].startswith('ex'):
                base_key = self.get_base_key(key)
                key = f"{key}.ex{'-'.join(map(str, range(len(self.challenge[base_key]['train']))))}"
            key_split = key.split('.')
            key_split[-1] = '-'.join(key_split[-1].split('-')[:-1])
            assert len(key_split[-1]) > 2 and key_split[-1].startswith('ex')
            key = '.'.join(key_split)
        return key, fmt

    @staticmethod
    def count_tokens(data, replace_special=re.compile('<[^<]*>')):
        replaced = replace_special.sub('x', data)  # replace '<...>' by a single char to count special tokens only once
        return len(replaced)

    @classmethod
    def max_new_tokens(cls, reply_end, lines_sep, max_size=30, safety_margin=1, **_):
        max_sized_reply = np.zeros([max_size, max_size], dtype=int)
        fmt = cls.fmt_output(max_sized_reply, reply_end=reply_end, lines_sep=lines_sep)
        return cls.count_tokens(fmt) + safety_margin

    def get_length(self, key, len_name, max_of_transposed=False, max_tokens=None, **fmt_opts):
        if not fmt_opts:
            fmt_opts = dict(preprompt='', query_beg='', reply_beg='', reply_end='', lines_sep='')
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name])
        else:
            length = self.count_tokens(self.fmt_task(key, **fmt_opts)[len_name])
            if max_of_transposed:
                length = max(length, self.count_tokens(self.fmt_task(f'{key}.tp', fmt_opts)[len_name]))
            length += 1  # for bos token
        return length

    def sort_keys_by_len(self, keys, reverse=False, **kwargs):
        lengths = [(key, self.get_length(key, **kwargs)) for key in keys]
        return [x[0] for x in sorted(lengths, reverse=reverse, key=lambda x: x[1])]

    def sorted_by_len(self,**kwargs):
        return self.change_keys(self.sort_keys_by_len(self.keys, **kwargs))

    def convert_with_token_limit(self, **kwargs):
        out_list = []
        new_keys = []
        for key in tqdm(self.keys, desc='convert dataset'):
            key, fmt = self.get_task(key, **kwargs)
            new_keys.append(key)
            out_list.append(fmt)
        return out_list, self.change_keys(new_keys)

    def as_list(self, **kwargs):
        return self.convert_with_token_limit(**kwargs)[0]

    @staticmethod
    def rand_perm(n, sep=None, keep_zero=False):
        permutation = np.random.permutation(n).tolist()
        if keep_zero:
            permutation = [0] + [x for x in permutation if x != 0]
        return permutation if sep is None else sep.join(map(str, permutation))

    def augment_keys(self, keys, tp=False, rt=False, n=1, shfl_keys=False, perm=False, keep_bg=False, shfl_ex=False, shf=False):
        keys = [k + n * '.tp' for n in range(2) for k in keys] if tp == 'all' else keys
        keys = [k + n * '.rt' for n in range(4) for k in keys] if rt == 'all' else keys
        keys = sum([list(np.random.permutation(keys) if shfl_keys else keys) for _ in range(n)], [])
        keys = [k + bool(tp) * randint(0, 2) * '.tp' for k in keys] if tp != 'all' else keys
        keys = [k + bool(rt) * randint(0, 4) * '.rt' for k in keys] if rt != 'all' else keys
        keys = [k + bool(perm) * ('.perm' + self.rand_perm(10, '', keep_bg)) for k in keys]
        n_ex = lambda k: len(self.challenge[self.get_base_key(k)]['train'])
        keys = [k + bool(shfl_ex) * ('.ex' + self.rand_perm(n_ex(k), '-')) for k in keys]
        # grid shift augmentation: shf='all' enumerates all 8 dirs x 5 pcts, True picks one randomly
        if shf == 'all':
            dirs = ['U', 'D', 'L', 'R', 'UR', 'UL', 'DR', 'DL']
            pcts = [10, 20, 30, 40, 50]
            keys = [f"{k}.shf{d}{p}" for k in keys for d in dirs for p in pcts]
        elif shf:
            dirs = ['U', 'D', 'L', 'R', 'UR', 'UL', 'DR', 'DL']
            pcts = [10, 20, 30, 40, 50]
            keys = [f"{k}.shf{dirs[randint(0, len(dirs))]}{pcts[randint(0, len(pcts))]}" for k in keys]
        return keys




    @staticmethod
    def _parse_shift_token(token):
        # token like 'shfUR30' -> ('UR', 30)
        m = re.match(r'^shf([UDLR]{1,2})(10|20|30|40|50)$', token)
        if not m:
            return None, None
        direction = m.group(1)
        # validate allowed composite directions
        if direction not in ('U', 'D', 'L', 'R', 'UR', 'UL', 'DR', 'DL'):
            return None, None
        pct = int(m.group(2))
        return direction, pct

    @classmethod
    def shift_array(cls, array, token, invert=False):
        # 刚体平移：空缺填充背景色
        h, w = array.shape
        direction, pct = cls._parse_shift_token(token)
        if direction is None:
            return array
        # 检测背景色
        bg_color = cls.get_background_color(array)
        # determine signed shifts (dy, dx). Positive dy moves content down, positive dx moves right
        def dir_to_vec(d):
            dy = (1 if 'D' in d else (-1 if 'U' in d else 0))
            dx = (1 if 'R' in d else (-1 if 'L' in d else 0))
            return dy, dx
        dy_sign, dx_sign = dir_to_vec(direction)
        # compute pixel offsets by percentage with ceil; axis with zero sign has zero shift
        dy = int(np.ceil(h * (pct / 100.0))) if dy_sign != 0 else 0
        dx = int(np.ceil(w * (pct / 100.0))) if dx_sign != 0 else 0
        if invert:
            dy_sign, dx_sign = -dy_sign, -dx_sign
        dy *= dy_sign
        dx *= dx_sign
        if dy == 0 and dx == 0:
            return array
        out = np.full_like(array, bg_color)
        # compute source and destination slices respecting bounds
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy) if dy <= 0 else min(h, h)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy) if dy >= 0 else min(h, h)
        # Correct y1 based on overlap length
        y_len = min(h - src_y0, h - dst_y0)
        src_y1 = src_y0 + max(0, y_len)
        dst_y1 = dst_y0 + max(0, y_len)

        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx) if dx <= 0 else min(w, w)
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx) if dx >= 0 else min(w, w)
        x_len = min(w - src_x0, w - dst_x0)
        src_x1 = src_x0 + max(0, x_len)
        dst_x1 = dst_x0 + max(0, x_len)

        if y_len > 0 and x_len > 0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
        return out

    # ==============================================================================
    # 视觉操作 (Visual Operations)
    # ==============================================================================
    # [网格分块]
    #   - pooldownup2: 马赛克化 (降采样后升采样)
    #   - modecenter3: 众数中心池化
    #   - removesmall: 移除小对象
    #
    # [刚体位移]
    #   - shf*: 刚体平移 (如 shfU10)
    #
    # [形态变换]
    #   - erosion: 腐蚀 (前景收缩)
    #   - dilation: 膨胀 (前景扩张)
    #   - skeleton: 骨架化 (拓扑骨架)
    #   - edge: 边缘检测 (像素级边缘)
    #   - boundary: 边界追踪 (对象外轮廓)
    #   - convexhull: 凸包填充
    #   - fillholes: 填充小洞 
    #  
    # [随机扰动]
    #   - noise*: 椒盐噪声
    #   - swap*: 局部像素交换
    # ==============================================================================

    @staticmethod
    def pool_down_up(array, kernel_size=2):
        """马赛克化：降采样后升采样，处理奇数行列
        """
        if kernel_size <= 1:
            return array
        
        h, w = array.shape
        h_trim = (h // kernel_size) * kernel_size
        w_trim = (w // kernel_size) * kernel_size
        
        # 如果网格太小无法进行池化，直接返回
        if h_trim == 0 or w_trim == 0:
            return array
        
        # 保存奇数行和列（原始数据）
        remainder_rows = h - h_trim  # 0 或 1
        remainder_cols = w - w_trim  # 0 或 1
        
        # 1. 下采样：对可整除部分进行众数池化
        trimmed = array[:h_trim, :w_trim]
        out_h = h_trim // kernel_size
        out_w = w_trim // kernel_size
        pooled = np.zeros((out_h, out_w), dtype=array.dtype)
        
        for i in range(out_h):
            for j in range(out_w):
                y0 = i * kernel_size
                x0 = j * kernel_size
                block = trimmed[y0:y0 + kernel_size, x0:x0 + kernel_size]
                values, counts = np.unique(block.flatten(), return_counts=True)
                pooled[i, j] = values[np.argmax(counts)]
        
        # 2. 上采样：最邻近插值
        upsampled = np.repeat(np.repeat(pooled, kernel_size, axis=0), kernel_size, axis=1)
        
        # 3. 构建完整结果，拼接保存的奇数行和列
        result = np.zeros_like(array)
        result[:h_trim, :w_trim] = upsampled
        
        # 拼接奇数列（保持原样）
        if remainder_cols > 0:
            result[:, w_trim:] = array[:, w_trim:]
        
        # 拼接奇数行（保持原样）
        if remainder_rows > 0:
            result[h_trim:, :] = array[h_trim:, :]
        
        return result

    @staticmethod
    def mode_center_pool(array, kernel_size=3):
        if kernel_size % 2 == 0 or kernel_size < 3:
            return array
        h, w = array.shape
        out = array.copy()
        blocks_y = h // kernel_size
        blocks_x = w // kernel_size
        center_offset = kernel_size // 2
        for by in range(blocks_y):
            for bx in range(blocks_x):
                y0 = by * kernel_size
                x0 = bx * kernel_size
                block = array[y0:y0 + kernel_size, x0:x0 + kernel_size]
                values, counts = np.unique(block, return_counts=True)
                mode_val = values[np.argmax(counts)]
                out[y0 + center_offset, x0 + center_offset] = mode_val
        return out

    @staticmethod
    def get_background_color(grid):
        """自动检测背景色（出现最多的颜色）"""
        h, w = grid.shape
        if h == 0 or w == 0:
            return 0
        border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        values, counts = np.unique(border, return_counts=True)
        max_count = counts.max()
        candidates = values[counts == max_count]
        if len(candidates) == 1:
            return int(candidates[0])
        corners = np.array([grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]])
        values, counts = np.unique(corners, return_counts=True)
        max_count = counts.max()
        candidates = values[counts == max_count]
        if len(candidates) == 1:
            return int(candidates[0])
        return int(grid[0, 0])

    @classmethod
    def erosion(cls, grid, bg_color=None):
        """腐蚀：前景像素如果4邻域有背景，则变为背景"""
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        h, w = grid.shape
        result = grid.copy()
        for i in range(h):
            for j in range(w):
                if grid[i, j] != bg_color:
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < h and 0 <= nj < w) or grid[ni, nj] == bg_color:
                            result[i, j] = bg_color
                            break
        return result

    @classmethod
    def dilation(cls, grid, bg_color=None):
        """膨胀：背景像素如果4邻域有前景，则变为该前景色"""
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        h, w = grid.shape
        result = grid.copy()
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(h):
            for j in range(w):
                if grid[i, j] != bg_color:
                    continue
                neighbor_colors = []
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        c = grid[ni, nj]
                        if c != bg_color:
                            neighbor_colors.append(c)
                if neighbor_colors:
                    values, counts = np.unique(neighbor_colors, return_counts=True)
                    max_count = counts.max()
                    candidates = values[counts == max_count]
                    result[i, j] = candidates.min()
        return result

    @classmethod
    def edge_detection(cls, grid, bg_color=None):
        """边缘检测：只保留边缘像素"""
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        h, w = grid.shape
        result = np.full_like(grid, bg_color)
        for i in range(h):
            for j in range(w):
                if grid[i, j] != bg_color:
                    is_edge = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < h and 0 <= nj < w) or grid[ni, nj] == bg_color:
                            is_edge = True
                            break
                    if is_edge:
                        result[i, j] = grid[i, j]
        return result

    @classmethod
    def salt_pepper_noise(cls, grid, ratio=0.05, seed=None, bg_color=None, color_candidates=None):
        """椒盐噪声：随机翻转部分像素
        
        Args:
            grid: 输入网格
            ratio: 噪声比例
            seed: 随机种子
            bg_color: 背景色
            color_candidates: 颜色候选集（可选，默认从grid获取）
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        result = grid.copy()
        h, w = grid.shape
        if ratio <= 0:
            return result
        n_noise = max(1, int(h * w * ratio))
        # 使用外部提供的颜色候选集，或从grid获取
        if color_candidates is not None:
            colors = list(color_candidates)
        else:
            colors = np.unique(grid).tolist()
        fg_colors = [c for c in colors if c != bg_color]
        # 如果没有前景色，则噪声操作无效
        if not fg_colors:
            return result
        for _ in range(n_noise):
            i, j = rng.integers(h), rng.integers(w)
            if grid[i, j] == bg_color:
                result[i, j] = rng.choice(fg_colors)
            else:
                result[i, j] = bg_color
        return result

    @classmethod
    def flood_fill_label(cls, mask):
        """简单的连通区域标记（4连通）"""
        h, w = mask.shape
        labeled = np.zeros_like(mask, dtype=int)
        current_label = 0
        for i in range(h):
            for j in range(w):
                if mask[i, j] and labeled[i, j] == 0:
                    current_label += 1
                    stack = [(i, j)]
                    while stack:
                        ci, cj = stack.pop()
                        if 0 <= ci < h and 0 <= cj < w and mask[ci, cj] and labeled[ci, cj] == 0:
                            labeled[ci, cj] = current_label
                            stack.extend([(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)])
        return labeled, current_label

    @classmethod
    def remove_small_objects(cls, grid, min_size=2, bg_color=None):
        """移除小于min_size的连通区域"""
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        result = grid.copy()
        for color in set(grid.flatten()) - {bg_color}:
            mask = (grid == color)
            labeled, n = cls.flood_fill_label(mask)
            for i in range(1, n + 1):
                if np.sum(labeled == i) < min_size:
                    result[labeled == i] = bg_color
        return result

    @classmethod
    def fill_holes(cls, grid, bg_color=None):
        """填充被前景完全包围的背景区域"""
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        result = grid.copy()
        h, w = grid.shape
        bg_mask = (grid == bg_color)
        labeled, n = cls.flood_fill_label(bg_mask)
        border_labels = set()
        border_labels.update(labeled[0, :].tolist())
        border_labels.update(labeled[-1, :].tolist())
        border_labels.update(labeled[:, 0].tolist())
        border_labels.update(labeled[:, -1].tolist())
        for label_id in range(1, n + 1):
            if label_id not in border_labels:
                mask = (labeled == label_id)
                neighbor_colors = []
                for i in range(h):
                    for j in range(w):
                        if mask[i, j]:
                            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w and not mask[ni, nj] and grid[ni, nj] != bg_color:
                                    neighbor_colors.append(grid[ni, nj])
                if neighbor_colors:
                    values, counts = np.unique(neighbor_colors, return_counts=True)
                    max_count = counts.max()
                    fill_color = values[counts == max_count].min()
                    result[mask] = fill_color
        return result

    @classmethod
    def local_swap_1d(cls, grid, ratio=0.1, seed=None):
        """一维相邻边界交换：用于1 x N或N x 1网格。"""
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        result = grid.copy()
        h, w = grid.shape
        if h == 1 and w > 1:
            line = result[0, :]
        elif w == 1 and h > 1:
            line = result[:, 0]
        else:
            return cls.local_swap(grid, ratio=ratio, seed=seed)

        boundary_indices = []
        for idx in range(line.shape[0]):
            if idx > 0 and line[idx - 1] != line[idx]:
                boundary_indices.append(idx)
                continue
            if idx + 1 < line.shape[0] and line[idx + 1] != line[idx]:
                boundary_indices.append(idx)

        if not boundary_indices:
            return result

        n_swaps = max(1, int(len(boundary_indices) * ratio))
        selected = rng.choice(len(boundary_indices), size=min(n_swaps, len(boundary_indices)), replace=False)
        for selected_idx in selected:
            idx = boundary_indices[selected_idx]
            diff_neighbors = []
            if idx > 0 and line[idx - 1] != line[idx]:
                diff_neighbors.append(idx - 1)
            if idx + 1 < line.shape[0] and line[idx + 1] != line[idx]:
                diff_neighbors.append(idx + 1)
            if diff_neighbors:
                other = diff_neighbors[rng.integers(len(diff_neighbors))]
                line[idx], line[other] = line[other], line[idx]

        return result

    @classmethod
    def local_swap(cls, grid, ratio=0.1, seed=None):
        """
        边界像素交换：随机交换颜色边界处的像素与其不同色邻居
        
        边界像素：任何与至少一个不同颜色邻居相邻的像素
        交换比例：边界像素总数的 ratio（默认 10%）
        """
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        result = grid.copy()
        h, w = grid.shape
        if h < 2 or w < 2:
            return result
        
        # 找到所有边界像素（与至少一个不同颜色邻居相邻）
        boundary_pixels = []
        for i in range(h):
            for j in range(w):
                color = grid[i, j]
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and grid[ni, nj] != color:
                        boundary_pixels.append((i, j))
                        break
        
        if not boundary_pixels:
            return result
        
        # 随机选择 ratio 比例的边界像素进行交换
        n_swaps = max(1, int(len(boundary_pixels) * ratio))
        selected = rng.choice(len(boundary_pixels), size=min(n_swaps, len(boundary_pixels)), replace=False)
        
        for idx in selected:
            i, j = boundary_pixels[idx]
            # 找该像素的不同色邻居并随机选一个交换
            diff_neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and result[ni, nj] != result[i, j]:
                    diff_neighbors.append((ni, nj))
            if diff_neighbors:
                ni, nj = diff_neighbors[rng.integers(len(diff_neighbors))]
                result[i, j], result[ni, nj] = result[ni, nj], result[i, j]
        
        return result

    @classmethod
    def skeletonize(cls, grid, bg_color=None):
        """骨架化：提取对象骨架，保持拓扑连通性"""
        h, w = grid.shape
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        result = grid.copy()
        
        # 对每种前景色分别进行骨架化
        for color in set(grid.flatten()) - {bg_color}:
            mask = (grid == color).astype(np.uint8)
            skeleton = np.zeros_like(mask)
            
            # Zhang-Suen 细化算法简化版
            changed = True
            while changed:
                changed = False
                # 第一次遍历
                to_delete = []
                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        if mask[i, j] == 0:
                            continue
                        p2, p3, p4 = mask[i-1, j], mask[i-1, j+1], mask[i, j+1]
                        p5, p6, p7 = mask[i+1, j+1], mask[i+1, j], mask[i+1, j-1]
                        p8, p9 = mask[i, j-1], mask[i-1, j-1]
                        
                        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                        B = sum(neighbors)  # 非零邻居数
                        
                        # 0->1 转换数
                        A = 0
                        seq = neighbors + [neighbors[0]]
                        for k in range(8):
                            if seq[k] == 0 and seq[k+1] == 1:
                                A += 1
                        
                        if 2 <= B <= 6 and A == 1:
                            if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                                to_delete.append((i, j))
                                changed = True
                
                for i, j in to_delete:
                    mask[i, j] = 0
                
                # 第二次遍历
                to_delete = []
                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        if mask[i, j] == 0:
                            continue
                        p2, p3, p4 = mask[i-1, j], mask[i-1, j+1], mask[i, j+1]
                        p5, p6, p7 = mask[i+1, j+1], mask[i+1, j], mask[i+1, j-1]
                        p8, p9 = mask[i, j-1], mask[i-1, j-1]
                        
                        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                        B = sum(neighbors)
                        
                        A = 0
                        seq = neighbors + [neighbors[0]]
                        for k in range(8):
                            if seq[k] == 0 and seq[k+1] == 1:
                                A += 1
                        
                        if 2 <= B <= 6 and A == 1:
                            if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                                to_delete.append((i, j))
                                changed = True
                
                for i, j in to_delete:
                    mask[i, j] = 0
            
            # 将骨架写入结果：先清空该颜色区域，再写入骨架
            result[grid == color] = bg_color
            result[mask == 1] = color
            
        return result

    @classmethod
    def boundary_trace(cls, grid, bg_color=None):
        """边界追踪：只保留对象的外边界"""
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        
        h, w = grid.shape
        result = np.full_like(grid, bg_color)
        
        for color in set(grid.flatten()) - {bg_color}:
            mask = (grid == color)
            labeled, n = cls.flood_fill_label(mask)
            
            for label_id in range(1, n + 1):
                region = (labeled == label_id)
                for i in range(h):
                    for j in range(w):
                        if region[i, j]:
                            is_boundary = False
                            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ni, nj = i + di, j + dj
                                if not (0 <= ni < h and 0 <= nj < w) or not region[ni, nj]:
                                    is_boundary = True
                                    break
                            if is_boundary:
                                result[i, j] = color
        
        return result

    @classmethod
    def convex_hull(cls, grid, bg_color=None):
        """凸包：对每个连通区域计算几何凸包并填充"""
        if bg_color is None:
            bg_color = cls.get_background_color(grid)
        h, w = grid.shape
        result = grid.copy()
        
        def cross(o, a, b):
            """叉积：判断 OA 到 OB 的转向"""
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        def convex_hull_points(points):
            """Andrew's monotone chain 算法计算凸包顶点"""
            points = sorted(set(map(tuple, points)))
            if len(points) <= 2:
                return points
            # 下凸包
            lower = []
            for p in points:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            # 上凸包
            upper = []
            for p in reversed(points):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            return lower[:-1] + upper[:-1]
        
        def point_in_polygon(x, y, polygon):
            """射线法判断点是否在多边形内"""
            n = len(polygon)
            if n < 3:
                return False
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside
        
        for color in set(grid.flatten()) - {bg_color}:
            mask = (grid == color)
            labeled, n = cls.flood_fill_label(mask)
            
            # 对每个连通域分别求凸包
            for label_id in range(1, n + 1):
                region = (labeled == label_id)
                points = np.argwhere(region)  # (row, col) 格式
                if len(points) < 3:
                    continue
                # 转换为 (col, row) 即 (x, y) 格式计算凸包
                xy_points = [(p[1], p[0]) for p in points]
                hull = convex_hull_points(xy_points)
                if len(hull) < 3:
                    continue
                # 填充凸包内部
                for r in range(h):
                    for c in range(w):
                        if result[r, c] == bg_color and point_in_polygon(c, r, hull):
                            result[r, c] = color
        
        return result

    def augment(self, seed, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return self.change_keys(self.augment_keys(self.keys, **kwargs))

    def decode(self, text, lines_sep, key=None):
        correct, info = None, 'unknown'
        try:
            data = [[int(x) for x in row if x.isdigit()] for row in text.split(lines_sep)]
            data = [row for row in data if len(row)]
            data = np.array(data, dtype=int)
            assert data.ndim == 2 and all(0 < x <= 30 for x in data.shape)
        except:
            data = None
            correct, info = False, 'cant_decode'
        if key is not None and data is not None:
            key_num, *transforms = key.split('.')
            base_key, reply_num = self.get_base_key_and_reply_num(key_num)
            data = self.transform_array(data, transforms, apply_shift=False, invert=True)
            correct_solution = self.solutions.get(base_key)
            if correct_solution is None:
                info = 'sol_unknown'
            else:
                correct_solution = np.asarray(correct_solution[reply_num])
                if np.array_equal(correct_solution, data):
                    correct, info = True, 'ALL_CORRECT'
                else:
                    correct, info = False, ('bad_content' if correct_solution.shape == data.shape else 'bad_xy_size')
        return data, correct, info

    def get_submission(self, results=None):
        assert self.is_orig, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(v['test']))] for k, v in self.challenge.items()}
        if results is not None:
            self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        for base_key, data in results.items():
            for reply_num, guesses in enumerate(data):
                target_dict = submission[base_key][reply_num]
                for i, g in enumerate(guesses[:len(target_dict)]):
                    target_dict[f'attempt_{i + 1}'] = g['output'].tolist()

    def validate_submission(self, submission):
        assert self.is_orig, 'Must be run on original dataset.'
        assert self.solutions, 'Solutions must be loaded for submission verification.'
        score = 0
        for k, v in self.solutions.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score
