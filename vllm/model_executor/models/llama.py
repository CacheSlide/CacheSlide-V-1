# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig
import torch.nn.functional as F

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_cope #CCPE
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

KVCACHE = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]


def _split_kv_cache(KV_cache: KVCACHE, kv_size: int):
    """Support:
    1) (K_cache, V_cache)
    2) {"k": K_cache, "v": V_cache}
    3) Tensor lastdim == 2*kv_size  -> split
    4) Tensor lastdim == kv_size    -> only K_cache (V_cache=None)
    """
    if isinstance(KV_cache, tuple):
        return KV_cache[0], KV_cache[1]
    if isinstance(KV_cache, dict):
        return KV_cache.get("k", None), KV_cache.get("v", None)
    # tensor
    if KV_cache.dim() >= 2 and KV_cache.shape[-1] == 2 * kv_size:
        k_cache, v_cache = KV_cache.split([kv_size, kv_size], dim=-1)
        return k_cache, v_cache
    if KV_cache.dim() >= 2 and KV_cache.shape[-1] == kv_size:
        return KV_cache, None
    raise ValueError(f"Unsupported KV_cache shape: {tuple(KV_cache.shape)}")


def _gather_cache_full_flat(cache: torch.Tensor,
                            cache_idx: torch.Tensor,
                            D: int) -> torch.Tensor:
    """Return gathered cache as [N, D] for all tokens (invalid positions will be garbage if idx=-1).
    Supports:
      cache: [S, D] and cache_idx: [B, T] or [N]
      cache: [B, S, D] and cache_idx: [B, T]
    """
    if cache.dim() == 2:
        # [S, D], use flat index_select for reuse positions only, caller should mask
        idx_flat = cache_idx.reshape(-1).clamp_min(0).long()
        return cache.index_select(0, idx_flat)  # [N, D]
    elif cache.dim() == 3:
        # [B, S, D], gather per batch
        assert cache_idx.dim() == 2, "For [B,S,D] cache, cache_idx must be [B,T]"
        idx = cache_idx.clamp_min(0).long().unsqueeze(-1).expand(-1, -1, D)  # [B,T,D]
        gathered = torch.gather(cache, dim=1, index=idx)  # [B,T,D]
        return gathered.reshape(-1, D)
    else:
        raise ValueError(f"Unsupported cache dim: {cache.dim()}")


def _token_cksim(k_rec: torch.Tensor, k_reuse: torch.Tensor,
                 num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """k_*: [M, D] where D=num_kv_heads*head_dim, return [M]"""
    M, D = k_rec.shape
    assert D == num_kv_heads * head_dim
    a = k_rec.view(M, num_kv_heads, head_dim).float()
    b = k_reuse.view(M, num_kv_heads, head_dim).float()
    cos = F.cosine_similarity(a, b, dim=-1)  # [M, H]
    return cos.mean(dim=-1)  # [M]
    # [N]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):

    def __init__(self,
                 config: LlamaConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 bias_o_proj: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        # Phi models introduced a partial_rotary_factor parameter in the config
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor",
                                             1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.npos_max = max_position_embeddings/2 # set the maximum number of CoPE position buckets as needed.

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )


        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        # self.rotary_emb = get_rope(
        #     self.head_dim,
        #     rotary_dim=self.head_dim,
        #     max_position=max_position_embeddings,
        #     base=rope_theta,
        #     rope_scaling=rope_scaling,
        #     is_neox_style=is_neox_style,
        #     partial_rotary_factor=self.partial_rotary_factor,
        # )

        self.rotary_emb = get_cope(
            npos_max=self.npos_max,
            head_dim=self.head_dim,
            # device=device,
            # dtype=dtype,
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        layernums: int,
        KV_cache: Optional[KVCACHE],
        cache_idx: Optional[torch.Tensor] = None,
        wca_ctx: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Requirements / assumptions:
        - self.qkv_proj returns qkv (and maybe bias) and you already split it
        - self.rotary_emb(positions, q, k) works
        - self.attn(q,k,v) works
        - self has attributes: self.kv_size, self.num_kv_heads, self.head_dim, self.q_size
        - cache_idx: reused token positions have >=0, miss tokens are -1
        - KV_cache provides K (+ optional V) in one of KVCACHE formats

        IMPORTANT:
        - wca_ctx must be the SAME dict passed through all layers of ONE request.
          Create it outside the layer loop and pass it into every layer forward.
        """

        # ---- 0) QKV + rotary ----
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # ---- 1)WCA defaults ----
        if wca_ctx is None:
            wca_ctx = {}
        topk_ratio = float(wca_ctx.get("topk_ratio", 0.26))
        tau = float(wca_ctx.get("tau", 0.12))
        eps = float(wca_ctx.get("eps", 1e-6))
        reselection_period = int(wca_ctx.get("reselect_period", 4))
        drop_if_low = bool(wca_ctx.get("drop_if_low", True))  # True: drop cksim < tau

        # ---- 2) flatten views ----
        D = k.shape[-1]  # should == self.kv_size
        k_flat = k.reshape(-1, D)
        v_flat = v.reshape(-1, D)

        # Save recomputed K/V BEFORE overwriting with cache.
        # Use clone() to avoid view/in-place aliasing issues.
        k_rec_flat = k_flat.clone()
        v_rec_flat = v_flat.clone()

        # ---- 3) If no reuse info / no cache, skip WCA ----
        if cache_idx is None or KV_cache is None:
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

        reuse_mask_flat = (cache_idx.reshape(-1) >= 0)
        if not reuse_mask_flat.any():
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

        # ---- 4) Gather cached K/V for all tokens ----
        reuse_pos = reuse_mask_flat.nonzero(as_tuple=False).squeeze(-1)  # [Nr]
        k_cache, v_cache = _split_kv_cache(KV_cache, self.kv_size)

        k_reuse_full = _gather_cache_full_flat(k_cache, cache_idx, D).to(k_flat.dtype)  # [N,D]
        v_reuse_full = None
        if v_cache is not None:
            v_reuse_full = _gather_cache_full_flat(v_cache, cache_idx, D).to(v_flat.dtype)

        # ---- 5) Paper semantics: for reused tokens, baseline uses cached KV ----
        k_flat.index_copy_(0, reuse_pos, k_reuse_full.index_select(0, reuse_pos))
        if v_reuse_full is not None:
            v_flat.index_copy_(0, reuse_pos, v_reuse_full.index_select(0, reuse_pos))

        # ---- 6) Layer 1: build S_sorted / Sk / ptr ----
        if layernums == 1:
            # d_i = ||Krec - Kreuse||^2 on reused tokens
            k_rec_reuse = k_rec_flat.index_select(0, reuse_pos).float()       # [Nr,D]
            k_reuse_reuse = k_reuse_full.index_select(0, reuse_pos).float()   # [Nr,D]
            diff = (k_rec_reuse - k_reuse_reuse).pow(2).sum(dim=-1)           # [Nr]

            nr = diff.numel()
            k_count = max(1, int(topk_ratio * nr))

            sorted_local = torch.argsort(diff, descending=True)               # [Nr]
            S_sorted = reuse_pos.index_select(0, sorted_local).contiguous()   # [Nr] (flat token indices)
            Sk = S_sorted[:k_count].contiguous()
            ptr = int(k_count)

            wca_ctx["S_sorted"] = S_sorted
            wca_ctx["Sk"] = Sk
            wca_ctx["ptr"] = ptr

        # ---- 7) Layer >=2: fuse only for Sk (K and V) ----
        Sk = wca_ctx.get("Sk", None)
        if isinstance(Sk, torch.Tensor) and Sk.numel() > 0 and layernums >= 2:
            Sk = Sk.to(k_flat.device).long()

            k_reuse_Sk = k_reuse_full.index_select(0, Sk).float()
            k_rec_Sk = k_rec_flat.index_select(0, Sk).float()

            # alpha = ||Krec - Kreuse||^2 / (||Kreuse||^2 + eps), clamp [0,1]
            num = (k_rec_Sk - k_reuse_Sk).pow(2).sum(dim=-1)                       # [k]
            den = k_reuse_Sk.pow(2).sum(dim=-1).clamp_min(eps)                     # [k]
            alpha = (num / den).clamp(0.0, 1.0).to(k_flat.dtype)                   # [k]

            alpha_col = alpha.unsqueeze(-1)                                        # [k,1]
            k_fused = alpha_col * k_rec_Sk.to(k_flat.dtype) + (1 - alpha_col) * k_reuse_Sk.to(k_flat.dtype)
            k_flat.index_copy_(0, Sk, k_fused)

            if v_reuse_full is not None:
                v_reuse_Sk = v_reuse_full.index_select(0, Sk).float()
                v_rec_Sk = v_rec_flat.index_select(0, Sk).float()
                v_fused = alpha_col * v_rec_Sk.to(v_flat.dtype) + (1 - alpha_col) * v_reuse_Sk.to(v_flat.dtype)
                v_flat.index_copy_(0, Sk, v_fused)

            # ---- 8) gated reselection every N layers ----
            if reselection_period > 0 and (layernums % reselection_period == 0):
                S_sorted = wca_ctx.get("S_sorted", None)
                ptr = int(wca_ctx.get("ptr", 0))
                if isinstance(S_sorted, torch.Tensor) and ptr < S_sorted.numel():
                    # CKSim on Sk using Krec vs Kreuse (token-level)
                    cksim = _token_cksim(
                        k_rec_Sk.to(k_flat.dtype),
                        k_reuse_Sk.to(k_flat.dtype),
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                    )
                    drop_mask = (cksim < tau) if drop_if_low else (cksim > tau)
                    n_drop = int(drop_mask.sum().item())

                    if n_drop > 0:
                        kept = Sk[~drop_mask].contiguous()
                        end = min(ptr + n_drop, S_sorted.numel())
                        new = S_sorted[ptr:end].to(Sk.device).long()
                        Sk_new = torch.cat([kept, new], dim=0)

                        wca_ctx["Sk"] = Sk_new
                        wca_ctx["ptr"] = end

        # ---- 9) reshape back and run attention ----
        k = k_flat.view_as(k)
        v = v_flat.view_as(v)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
     # ---- End WCA defaults ----

class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, 'qkv_bias'):
            attention_bias = config.qkv_bias

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int] = tuple()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        self.cope = CoPEPositionalEncoder(fixed_ranges=config.fixed_position_ranges)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        KV_cache: Optional[KVCACHE] = None,
        cache_idx: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, IntermediateTensors, tuple[torch.Tensor,
                                                        list[torch.Tensor]]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        # for idx, layer in enumerate(
        #         self.layers[self.start_layer:self.end_layer]):
        #     if idx in self.aux_hidden_state_layers:
        #         aux_hidden_states.append(hidden_states + residual)
        #     hidden_states, residual = layer(positions, hidden_states, residual)
        # 1) 在 layer loop 之前创建 wca_ctx（一次 forward 只创建一次）
        wca_ctx = {
            "topk_ratio": 0.26,
            "tau": 0.12,
            "reselect_period": 4,
            "eps": 1e-6,
            "drop_if_low": True,
        }

        # 2) 进入 layer loop，每层传同一个 wca_ctx
        for local_idx, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            global_layernum = self.start_layer + local_idx + 1  # 全局层号：1..L

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                layernums=global_layernum,
                KV_cache=KV_cache,  # 你得在 LlamaModel.forward() 里能拿到它
                cache_idx=cache_idx,  # 同上
                wca_ctx=wca_ctx,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[nn.Module] = LlamaDecoderLayer):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"),
                                      layer_type=layer_type)

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def set_aux_hidden_state_layers(self, layers: tuple[int]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = "",
                    layer_type: type[nn.Module] = LlamaDecoderLayer):
        return LlamaModel(vllm_config=vllm_config,
                          prefix=prefix,
                          layer_type=layer_type)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules and modules[-1] == "weight":
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        num_modules = len(modules)
        for i in range(num_modules):
            item = modules[i]
            next_item = modules[i + 1] if i < num_modules - 1 else None

            combined_item = (f"{item}.{next_item}"
                             if next_item is not None else None)

            if combined_item in mapping:
                name = name.replace(combined_item, mapping[combined_item])
            elif item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
