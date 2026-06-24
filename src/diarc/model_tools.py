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
import json
import shutil
import warnings
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import peft
from tokenizers import Tokenizer
from transformers import DataCollatorForLanguageModeling

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
        def __init__(
            self,
            response_template: Union[str, List[int]],
            instruction_template: Optional[Union[str, List[int]]] = None,
            *args,
            mlm: bool = False,
            ignore_index: int = -100,
            **kwargs,
        ):
            super().__init__(*args, mlm=mlm, **kwargs)

            self.instruction_template = instruction_template
            if isinstance(instruction_template, str):
                self.instruction_token_ids = self.tokenizer.encode(instruction_template, add_special_tokens=False)
            else:
                self.instruction_token_ids = instruction_template

            self.response_template = response_template
            if isinstance(response_template, str):
                self.response_token_ids = self.tokenizer.encode(response_template, add_special_tokens=False)
            else:
                self.response_token_ids = response_template

            if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                warnings.warn(
                    "pad_token_id and eos_token_id are identical; multi-turn masking may behave poorly."
                )

            self.ignore_index = ignore_index

        def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
            batch = super().torch_call(examples)

            if self.instruction_template is None:
                for i in range(len(examples)):
                    response_start_idx = None

                    for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                        if self.response_token_ids == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist():
                            response_start_idx = idx

                    if response_start_idx is None:
                        warnings.warn(
                            "Could not find response marker in an instance; this sample will be ignored in loss."
                        )
                        batch["labels"][i, :] = self.ignore_index
                    else:
                        response_end_idx = response_start_idx + len(self.response_token_ids)
                        batch["labels"][i, :response_end_idx] = self.ignore_index

            else:
                for i in range(len(examples)):
                    response_idxs = []
                    instruction_idxs = []

                    for response_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                        if self.response_token_ids == batch["labels"][i][response_idx : response_idx + len(self.response_token_ids)].tolist():
                            response_idxs.append(response_idx + len(self.response_token_ids))

                    if len(response_idxs) == 0:
                        warnings.warn(
                            "Could not find response marker in an instance; this sample will be ignored in loss."
                        )
                        batch["labels"][i, :] = self.ignore_index
                        continue

                    instruction_token_ids = self.instruction_token_ids
                    for instruction_idx in np.where(batch["labels"][i] == instruction_token_ids[0])[0]:
                        if instruction_token_ids == batch["labels"][i][instruction_idx : instruction_idx + len(instruction_token_ids)].tolist():
                            instruction_idxs.append(instruction_idx)

                    if len(instruction_idxs) == 0:
                        warnings.warn(
                            "Could not find instruction marker in an instance; this sample will be ignored in loss."
                        )
                        batch["labels"][i, :] = self.ignore_index
                        continue

                    if instruction_idxs[0] > response_idxs[0]:
                        instruction_idxs = [0] + instruction_idxs

                    for idx, (start, end) in enumerate(zip(instruction_idxs, response_idxs)):
                        if idx != 0:
                            batch["labels"][i, start:end] = self.ignore_index
                        else:
                            batch["labels"][i, :end] = self.ignore_index

                    if len(response_idxs) < len(instruction_idxs):
                        batch["labels"][i, instruction_idxs[-1] :] = self.ignore_index

            return batch


# trl version warning
import trl
assert not trl.__version__.startswith('0.15'), """
WARNING: Do not use this code with trl version 0.15.x!
In combination with unsloth, this will shorten all training inputs
to 1024 tokens, speeding up training, but severely degrading accuracy. 
"""


class InputMaskingDataCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, mask_first_n_examples=0, **kwargs):
        super().__init__(**kwargs)
        self.mask_first_n_examples = mask_first_n_examples

    def torch_call(self, examples):
        batch = super().torch_call(examples)  # call super, masking all inputs
        for i in range(len(batch['labels'])):
            for _ in range(self.mask_first_n_examples):
                # mask first still unmasked output block
                beg_pos = ((batch['labels'][i] != -100).nonzero().min()).item()
                if not (batch['labels'][i][beg_pos:] == -100).any(): break
                mid_pos = ((batch['labels'][i][beg_pos:] == -100).nonzero().min()).item() + beg_pos
                end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                if mid_pos < end_pos:
                    batch['labels'][i][beg_pos:mid_pos] = -100
        return batch


def load_tf_tokenizer(model_path):
    from transformers import AutoTokenizer
    _patch_auto_tokenizer_from_pretrained()
    return AutoTokenizer.from_pretrained(_sanitize_local_model_path(model_path))


def load_tf_model(model_path, bits, dtype=torch.bfloat16, attn_implementation='flash_attention_2', **kw):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    model_path = _sanitize_local_model_path(model_path)
    if bits is not None:
        kw['quantization_config'] = {
            8: BitsAndBytesConfig(load_in_8bit=True),
            4: BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=dtype),
        }[bits]

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, attn_implementation=attn_implementation, **kw)
    return model, load_tf_tokenizer(model_path)


def _ensure_symlink(dst: Path, target: Path):
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and os.path.realpath(dst) == os.path.realpath(target):
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(target, dst, target_is_directory=target.is_dir())


def _sanitize_local_model_path(model_path):
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return model_path

    broken_cfg_path = model_dir / 'configuration.json'
    config_path = model_dir / 'config.json'
    if not (broken_cfg_path.exists() and config_path.exists()):
        return model_path

    try:
        broken_cfg = json.loads(broken_cfg_path.read_text())
        config = json.loads(config_path.read_text())
    except Exception:
        return model_path

    if isinstance(broken_cfg, dict) and broken_cfg.get('model_type'):
        return model_path
    if not (isinstance(config, dict) and config.get('model_type')):
        return model_path

    alias_root = Path('/tmp/codex_model_aliases')
    alias_root.mkdir(parents=True, exist_ok=True)
    alias_dir = alias_root / model_dir.name
    alias_dir.mkdir(parents=True, exist_ok=True)

    for child in model_dir.iterdir():
        alias_child = alias_dir / child.name
        if child.name == 'configuration.json':
            _ensure_symlink(alias_child, config_path)
        else:
            _ensure_symlink(alias_child, child)

    return str(alias_dir)


def _build_fast_tokenizer_from_local_files(model_path):
    from transformers import PreTrainedTokenizerFast

    model_dir = Path(model_path)
    tokenizer_file = model_dir / 'tokenizer.json'
    if not tokenizer_file.exists():
        raise FileNotFoundError(f'missing tokenizer file: {tokenizer_file}')

    tokenizer_config = {}
    special_tokens = {}
    tokenizer_config_path = model_dir / 'tokenizer_config.json'
    special_tokens_path = model_dir / 'special_tokens_map.json'
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
    if special_tokens_path.exists():
        special_tokens = json.loads(special_tokens_path.read_text())

    kwargs = {}
    for key in ['bos_token', 'eos_token', 'unk_token', 'pad_token', 'cls_token', 'sep_token', 'mask_token']:
        value = special_tokens.get(key, tokenizer_config.get(key))
        if isinstance(value, dict) and 'content' in value:
            value = value['content']
        if value is not None:
            kwargs[key] = value

    for key in ['model_max_length', 'padding_side', 'truncation_side', 'clean_up_tokenization_spaces']:
        value = tokenizer_config.get(key)
        if value is not None:
            kwargs[key] = value

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **kwargs)
    chat_template = tokenizer_config.get('chat_template')
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def _patch_auto_tokenizer_from_pretrained():
    from transformers import AutoTokenizer

    if getattr(AutoTokenizer, '_codex_tokenizer_patch', False):
        return

    original_from_pretrained = AutoTokenizer.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        try:
            return original_from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        except AttributeError as exc:
            if "'dict' object has no attribute 'model_type'" not in str(exc):
                raise
            local_path = _sanitize_local_model_path(pretrained_model_name_or_path)
            return _build_fast_tokenizer_from_local_files(local_path)

    AutoTokenizer.from_pretrained = patched_from_pretrained
    AutoTokenizer._codex_tokenizer_patch = True


def load_unsloth_model(model_name, bits, **kw):
    assert bits in [None, 8, 4]
    from unsloth import FastLanguageModel
    _patch_auto_tokenizer_from_pretrained()
    model_name = _sanitize_local_model_path(model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name, load_in_4bit=bits==4, load_in_8bit=bits==8, **kw)
    if model.max_seq_length == 2048 < model.generation_config.max_length:
        print(f'CHANGING MAX_SEQ_LENGTH {model.max_seq_length} -> {model.generation_config.max_length} (unsloth bug?)')
        to_fix = model
        while to_fix is not None:
            to_fix.max_seq_length = model.generation_config.max_length
            to_fix = getattr(to_fix, 'model', None)
    return model, tokenizer


def save_model_and_tokenizer(store_path, model, tokenizer):
    model.save_pretrained(store_path)
    tokenizer.save_pretrained(store_path)
    to_delete = os.path.join(store_path, 'tokenizer.model')  # delete file, as it interferes with token removal
    if os.path.isfile(to_delete):
        os.remove(to_delete)


def fix_dtypes(model, fix_weights=True, fix_quant_states=True):
    # fix some data types (workaround for unsloth)
    for module in model.modules():
        weight = getattr(module, 'weight', None)
        if weight is not None:
            if torch.is_floating_point(weight):
                if fix_weights and weight.dtype != model.dtype:
                    module.to(model.dtype)
            else:
                qs = getattr(weight, 'quant_state', None)
                if qs is not None:
                    if fix_quant_states and qs.dtype != model.dtype:
                        qs.dtype = model.dtype
    return model


def is_peft_model(model):
    return hasattr(model, 'peft_type')


def merge_peft_into_base(model):
    assert is_peft_model(model)
    return fix_dtypes(model.merge_and_unload())


def get_and_fix_peft_weights(store):
    # change some keys (workaround for added 'modules_to_save' and 'base_layer')
    # Check if store is a local path or a HuggingFace Hub repo
    if os.path.exists(store):
        # Load from local path
        from safetensors.torch import load_file
        adapter_path = os.path.join(store, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            state_dict = load_file(adapter_path)
        else:
            # Try loading with peft for backward compatibility
            state_dict = peft.load_peft_weights(store)
    else:
        # Load from HuggingFace Hub
        state_dict = peft.load_peft_weights(store)
    
    # Create a new state dict with fixed keys
    fixed_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'modules_to_save' keys
        if 'modules_to_save' in k:
            # Skip modules_to_save keys, use the original key instead
            original_key = k.replace('.modules_to_save.', '.')
            if original_key in state_dict:
                continue  # Skip, will use original key
            original_module_key = k.replace('.modules_to_save.', '.original_module.')
            if original_module_key in state_dict:
                continue  # Skip, will use original key
            # If we can't find original, skip this key
            continue
        # Remove 'base_layer' prefix if present (PEFT expects keys without base_layer)
        if '.base_layer.' in k:
            fixed_key = k.replace('.base_layer.', '.')
            fixed_state_dict[fixed_key] = v
        else:
            fixed_state_dict[k] = v
    return fixed_state_dict


def set_peft_weights(model, state_dict):
    try:
        res = peft.set_peft_model_state_dict(model, state_dict)
        # Check if res has unexpected_keys attribute (older PEFT versions)
        if res is not None and hasattr(res, 'unexpected_keys'):
            if res.unexpected_keys:
                print(f"Warning: Unexpected keys when loading PEFT weights: {res.unexpected_keys[:5]}...")
    except KeyError as e:
        # Handle key errors - this might happen if key names don't match
        error_msg = str(e)
        print(f"Error loading PEFT weights: {error_msg}")
        raise


def load_peft_state(model, store):
    # convenience method to load peft weights from file and set them for model
    set_peft_weights(model, get_and_fix_peft_weights(store))


def get_or_map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:  # find and/or update special token mappings
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
        for v in data.values():  # recursively process dict values
            tokens.update(get_or_map_special_tokens(v, mapping))
    if isinstance(data, list):
        for v in data:  # recursively process lists
            tokens.update(get_or_map_special_tokens(v, mapping))
    return tokens


def remove_tokenizer_normalizer(tokenizer):
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special=True, remove_unk=False):
    assert tokenizer.is_fast
    tok_json = json.loads(tokenizer._tokenizer.to_str())
    assert tok_json['model']['type'] == "BPE"

    if keep_special:  # get special tokens to keep
        keep_indices.update(tokenizer.all_special_ids)
        keep_indices.update(get_or_map_special_tokens(tok_json.get('post_processor')))

    if remove_unk:  # remove unknown token
        keep_indices -= {tokenizer.unk_token_id}

    # build mapping from old to new id
    mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

    # update tokenizer info
    tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
    tok_json['model']['merges'] = []
    tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
    tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])
    get_or_map_special_tokens(tok_json.get('post_processor'), mapping)

    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))  # reload json, modifying tokenizer in-place

    if remove_unk:
        tokenizer.unk_token = None

    return mapping  # token mapping to be used later


def shrink_model_embeddings(model, mapping):
    with torch.no_grad():
        # copy embeddings to keep
        row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
        row_select = row_select.to(model.get_input_embeddings().weight.data.device)
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select)
        row_select = row_select.to(model.get_output_embeddings().weight.data.device)
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select)

        # resize model embeddings
        model.resize_token_embeddings(len(row_select))

        # set to copied values
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head

        # map model tokens to new id
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))


def keep_single_char_tokens(model, tokenizer, keep=None, keep_norm=False, keep_model_tok=True, **kwargs):
    if not keep_norm:
        remove_tokenizer_normalizer(tokenizer)  # required for some models
    if keep is None:  # keep all single_length tokens
        keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
    else:  # keep tokens that were passed
        keep_indices = set(tokenizer.vocab[t] for t in keep)
    if keep_model_tok:  # keep tokens used by model
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update(v if isinstance(v, list) else [v])
    keep_indices -= {None}
    mapping = shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
    shrink_model_embeddings(model, mapping)
    return mapping
