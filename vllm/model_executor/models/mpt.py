# SPDX-License-Identifier: Apache-2.0

# Adapted from https://huggingface.co/mosaicml/mpt-7b/tree/main
import math
from typing import Iterable, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.mpt import MPTConfig

from .interfaces import SupportsPP
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union
import torch.nn.functional as F


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
    if KV_cache.dim() >= 2 and KV_cache.shape[-1] == 2 * kv_size:
        k_cache, v_cache = KV_cache.split([kv_size, kv_size], dim=-1)
        return k_cache, v_cache
    if KV_cache.dim() >= 2 and KV_cache.shape[-1] == kv_size:
        return KV_cache, None
    raise ValueError(f"Unsupported KV_cache shape: {tuple(KV_cache.shape)}")

def _gather_cache_full_flat(cache: torch.Tensor, cache_idx: torch.Tensor, D: int) -> torch.Tensor:
    """Return gathered cache as [N, D] for all tokens (idx=-1 will be clamped to 0)."""
    if cache.dim() == 2:
        idx_flat = cache_idx.reshape(-1).clamp_min(0).long()
        return cache.index_select(0, idx_flat)  # [N,D]
    if cache.dim() == 3:
        assert cache_idx.dim() == 2, "For [B,S,D] cache, cache_idx must be [B,T]"
        idx = cache_idx.clamp_min(0).long().unsqueeze(-1).expand(-1, -1, D)  # [B,T,D]
        gathered = torch.gather(cache, dim=1, index=idx)  # [B,T,D]
        return gathered.reshape(-1, D)
    raise ValueError(f"Unsupported cache dim: {cache.dim()}")

def _token_cksim(k_rec: torch.Tensor, k_reuse: torch.Tensor,
                 num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """k_*: [M,D], D=num_kv_heads*head_dim, return [M]"""
    M, D = k_rec.shape
    assert D == num_kv_heads * head_dim
    a = k_rec.view(M, num_kv_heads, head_dim).float()
    b = k_reuse.view(M, num_kv_heads, head_dim).float()
    cos = F.cosine_similarity(a, b, dim=-1)  # [M,H]
    return cos.mean(dim=-1)                  # [M]


def _get_alibi_slopes(
    total_num_heads: int,
    alibi_bias_max: int,
) -> torch.Tensor:
    next_power_of_2 = 2**math.ceil(math.log2(total_num_heads))
    m = torch.arange(1, next_power_of_2 + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / next_power_of_2)
    slopes = 1.0 / torch.pow(2, m)
    if next_power_of_2 != total_num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:total_num_heads]
    return slopes


class MPTAttention(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.clip_qkv = config.attn_config["clip_qkv"]
        self.qk_ln = config.attn_config["qk_ln"]
        self.alibi_bias_max = config.attn_config["alibi_bias_max"]
        if "kv_n_heads" in config.attn_config:
            self.total_num_kv_heads = config.attn_config['kv_n_heads']
        else:
            self.total_num_kv_heads = self.total_num_heads
        assert not config.attn_config["prefix_lm"]
        assert config.attn_config["alibi"]

        # pylint: disable=invalid-name
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=not config.no_bias,
            quant_config=quant_config,
        )
        if self.qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model)
            self.k_ln = nn.LayerNorm(self.d_model)
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=not config.no_bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Create the alibi slopes and slice them.
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.total_num_heads,
                                         self.alibi_bias_max)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()

        self.head_dim = self.d_model // self.total_num_heads
        scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scaling,
                              alibi_slopes=alibi_slopes,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
            self,
            position_ids: torch.Tensor,
            hidden_states: torch.Tensor,
            layernums: int,
            KV_cache: Optional[KVCACHE],
            cache_idx: Optional[torch.Tensor] = None,
            wca_ctx: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        del position_ids  # unused.

        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)

        # ---- WCA defaults ----
        if wca_ctx is None:
            wca_ctx = {}
        topk_ratio = float(wca_ctx.get("topk_ratio", 0.26))
        tau = float(wca_ctx.get("tau", 0.12))
        eps = float(wca_ctx.get("eps", 1e-6))
        reselection_period = int(wca_ctx.get("reselect_period", 4))
        drop_if_low = bool(wca_ctx.get("drop_if_low", True))

        # flatten
        D = k.shape[-1]  # == kv_size
        k_flat = k.reshape(-1, D)
        v_flat = v.reshape(-1, D)

        # save recomputed
        k_rec_flat = k_flat.clone()
        v_rec_flat = v_flat.clone()

        # ---- no cache -> normal attention ----
        if cache_idx is None or KV_cache is None:
            attn_output = self.attn(q, k, v)
            output, _ = self.out_proj(attn_output)
            return output

        reuse_mask_flat = (cache_idx.reshape(-1) >= 0)
        if not reuse_mask_flat.any():
            attn_output = self.attn(q, k, v)
            output, _ = self.out_proj(attn_output)
            return output

        reuse_pos = reuse_mask_flat.nonzero(as_tuple=False).squeeze(-1)  # [Nr]

        # ---- gather cache ----
        k_cache, v_cache = _split_kv_cache(KV_cache, self.kv_size)
        k_reuse_full = _gather_cache_full_flat(k_cache, cache_idx, D).to(k_flat.dtype)
        v_reuse_full = None
        if v_cache is not None:
            v_reuse_full = _gather_cache_full_flat(v_cache, cache_idx, D).to(v_flat.dtype)

        # baseline semantics: overwrite reused tokens with cached kv
        k_flat.index_copy_(0, reuse_pos, k_reuse_full.index_select(0, reuse_pos))
        if v_reuse_full is not None:
            v_flat.index_copy_(0, reuse_pos, v_reuse_full.index_select(0, reuse_pos))

        # ---- layer 1: build S_sorted/Sk/ptr ----
        if layernums == 1:
            k_rec_reuse = k_rec_flat.index_select(0, reuse_pos).float()
            k_reuse_reuse = k_reuse_full.index_select(0, reuse_pos).float()
            diff = (k_rec_reuse - k_reuse_reuse).pow(2).sum(dim=-1)  # [Nr]

            nr = diff.numel()
            k_count = max(1, int(topk_ratio * nr))

            sorted_local = torch.argsort(diff, descending=True)
            S_sorted = reuse_pos.index_select(0, sorted_local).contiguous()
            Sk = S_sorted[:k_count].contiguous()
            ptr = int(k_count)

            wca_ctx["S_sorted"] = S_sorted
            wca_ctx["Sk"] = Sk
            wca_ctx["ptr"] = ptr

        # ---- layer >=2: fuse on Sk ----
        Sk = wca_ctx.get("Sk", None)
        if isinstance(Sk, torch.Tensor) and Sk.numel() > 0 and layernums >= 2:
            Sk = Sk.to(k_flat.device).long()

            k_reuse_Sk = k_reuse_full.index_select(0, Sk).float()
            k_rec_Sk = k_rec_flat.index_select(0, Sk).float()

            num = (k_rec_Sk - k_reuse_Sk).pow(2).sum(dim=-1)
            den = k_reuse_Sk.pow(2).sum(dim=-1).clamp_min(eps)
            alpha = (num / den).clamp(0.0, 1.0).to(k_flat.dtype)

            alpha_col = alpha.unsqueeze(-1)
            k_fused = alpha_col * k_rec_Sk.to(k_flat.dtype) + (1 - alpha_col) * k_reuse_Sk.to(k_flat.dtype)
            k_flat.index_copy_(0, Sk, k_fused)

            if v_reuse_full is not None:
                v_reuse_Sk = v_reuse_full.index_select(0, Sk).float()
                v_rec_Sk = v_rec_flat.index_select(0, Sk).float()
                v_fused = alpha_col * v_rec_Sk.to(v_flat.dtype) + (1 - alpha_col) * v_reuse_Sk.to(v_flat.dtype)
                v_flat.index_copy_(0, Sk, v_fused)

            # ---- reselection ----
            if reselection_period > 0 and (layernums % reselection_period == 0):
                S_sorted = wca_ctx.get("S_sorted", None)
                ptr = int(wca_ctx.get("ptr", 0))
                if isinstance(S_sorted, torch.Tensor) and ptr < S_sorted.numel():
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

        # reshape back
        k = k_flat.view_as(k)
        v = v_flat.view_as(v)

        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class MPTMLP(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.d_model
        expansion_ratio = config.expansion_ratio
        intermediate_size = expansion_ratio * hidden_size
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=not config.no_bias,
            quant_config=quant_config,
        )
        self.act = get_act_fn("gelu")
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=not config.no_bias,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act(x)
        x, _ = self.down_proj(x)
        return x


class MPTBlock(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.d_model
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn = MPTAttention(config,
                                 cache_config,
                                 quant_config,
                                 prefix=f"{prefix}.attn")
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ffn = MPTMLP(config, quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm_1(hidden_states)
        x = self.attn(
            position_ids=position_ids,
            hidden_states=x,
        )
        hidden_states = hidden_states + x
        x = self.norm_2(hidden_states)
        x = self.ffn(x)
        hidden_states = hidden_states + x
        return hidden_states


@support_torch_compile
class MPTModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        assert config.embedding_fraction == 1.0
        assert config.norm_type == "low_precision_layernorm"

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.start_layer, self.end_layer, self.blocks = make_layers(
            config.n_layers,
            lambda prefix: MPTBlock(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.blocks")
        self.norm_f = nn.LayerNorm(config.d_model)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, "bias") and isinstance(
                        module.bias, nn.Parameter):
                    # Remove the bias term in Linear and LayerNorm.
                    module.register_parameter("bias", None)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.d_model))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor] = None,
            KV_cache: Optional[KVCACHE] = None,
            cache_idx: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # for block in self.blocks[self.start_layer:self.end_layer]:
        #     hidden_states = block(position_ids, hidden_states)
        # 1) one ctx per forward
        wca_ctx = {
            "topk_ratio": 0.26,
            "tau": 0.12,
            "reselect_period": 4,
            "eps": 1e-6,
            "drop_if_low": True,
        }

        for local_idx, block in enumerate(self.blocks[self.start_layer:self.end_layer]):
            global_layernum = self.start_layer + local_idx + 1  # 1..L (global)
            hidden_states = block(
                position_ids,
                hidden_states,
                layernums=global_layernum,
                KV_cache=KV_cache,
                cache_idx=cache_idx,
                wca_ctx=wca_ctx,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.norm_f(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
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


class MPTForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        assert config.tie_word_embeddings
        self.quant_config = quant_config

        self.transformer = MPTModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "transformer"))
        self.lm_head = self.transformer.wte
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.get_input_embeddings(input_ids)

    # def forward(
    #     self,
    #     input_ids: torch.Tensor,
    #     positions: torch.Tensor,
    #     intermediate_tensors: Optional[IntermediateTensors] = None,
    #     inputs_embeds: Optional[torch.Tensor] = None,
    # ) -> Union[torch.Tensor, IntermediateTensors]:
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            KV_cache: Optional[KVCACHE] = None,
            cache_idx: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        # hidden_states = self.transformer(input_ids, positions,
        #                                  intermediate_tensors, inputs_embeds)
        hidden_states = self.transformer(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            KV_cache=KV_cache,
            cache_idx=cache_idx,
        )
        return hidden_states

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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
