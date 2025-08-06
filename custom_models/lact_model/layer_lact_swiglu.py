# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.utils import logging

from fla.models.utils import Cache
from fla.modules import RMSNorm, RotaryEmbedding
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

from .ttt_operation import block_causal_lact_swiglu, prenorm_block_causal_lact_swiglu, l2_norm, zeropower_via_newtonschulz5_2d

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class LowRankFastWeight(nn.Module):
    """
    Low rank fast weight. This is a compromise to keep the number of parameters low when comparing against baselines. 
    Idealy, low-rank parameterization always hurts the performance. 
    Args:
        num_heads: number of heads
        out_features: output features
        in_features: input features
        rank: rank of the low rank fast weight
        init_gain: initialization gain
        add_identity: whether to add identity matrix to the fast weight
    Returns:
        W: [num_heads, out_features, in_features]
    W = W_left @ W_right + I * 0.5
        where I is the identity matrix if add_identity is True.
    """
    def __init__(self, num_heads, out_features, in_features, rank=32, 
                 init_gain=0.5, add_identity=False):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.add_identity = add_identity
        
        self.w_left = nn.Parameter(torch.randn(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.randn(num_heads, rank, in_features))
        self.init_gain = init_gain

        print("init low rank fast weight", num_heads, out_features, in_features, rank)

    def _init_weights(self):
        
        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(self.w_right, std=1.0 / math.sqrt(self.in_features) * self.init_gain)

    def forward(self,):
        """
        Returns:
            W: [num_heads, out_features, in_features]
            W = W_left @ W_right + I * 0.5
            where I is the identity matrix if add_identity is True.
        """

        W = self.w_left @ self.w_right

        if self.add_identity:
            W += torch.eye(self.out_features, self.in_features, device=W.device, dtype=W.dtype).unsqueeze(0) * 0.5

        return W


class SwiGLUFastWeightBlock(nn.Module):
    """
    A single SwiGLU fast weight block for the stacked TTT architecture.
    This encapsulates the w0, w1, w2 parameters for a single MLP in the stack.
    """
    def __init__(
        self,
        num_heads: int,
        d_h: int,
        d_in: int,
        d_out: int,
        w0_w2_low_rank: int = -1,
        fw_init_gain: float = 0.5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_h = d_h
        self.d_in = d_in
        self.d_out = d_out
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        
        # Initialize fast weights
        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(num_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=self.fw_init_gain, add_identity=True)
            self.w2 = LowRankFastWeight(num_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=self.fw_init_gain, add_identity=True)
        else:
            self.w0 = nn.Parameter(
                torch.randn(num_heads, d_h, d_in) / math.sqrt(d_in)
            )
            self.w2 = nn.Parameter(
                torch.randn(num_heads, d_h, d_in) / math.sqrt(d_in)
            )
        
        self.w1 = nn.Parameter(
            torch.randn(num_heads, d_out, d_h) / math.sqrt(d_h)
        )


class LaCTSWIGLULayer(nn.Module):

    def __init__(
        self,
        config=None,  # Full config object
        hidden_size: int = None,
        num_attn_heads: int = None,
        num_lact_heads: int = None,
        inter_multi: float = None,
        window_size: int = None,
        lact_chunk_size: int = None,
        qkv_bias: bool = None,
        attn_qk_norm: bool = None,
        qkv_silu: bool = None,
        lr_dim: int = None,
        use_muon: bool = None,
        lr_parameterization: str = None,
        learnable_ttt_scale: bool = None,
        ttt_prenorm: bool = None,
        ttt_nope: bool = None,
        rope_theta: float = None,
        layer_idx: int = None,
        max_position_embeddings: int = None,
        w0_w2_low_rank: int = None,
        use_momentum: bool = None,
        ttt_loss_type: str = None,
        fw_init_gain: float = None, # init the fast weights
        r: int = None,  # Number of stacked SwiGLU MLPs
        residual_ttt: bool = None,  # Whether to use residual connections between stacked MLPs
    ):
        # Use config values if provided, otherwise use individual parameters
        if config is not None:
            self.config = config
            hidden_size = config.hidden_size
            num_attn_heads = config.num_attn_heads
            num_lact_heads = config.num_lact_heads
            inter_multi = config.inter_multi
            window_size = config.window_size
            lact_chunk_size = config.lact_chunk_size
            qkv_bias = config.qkv_bias
            attn_qk_norm = config.attn_qk_norm
            qkv_silu = config.qkv_silu
            lr_dim = config.lr_dim
            use_muon = config.use_muon
            lr_parameterization = config.lr_parameterization
            learnable_ttt_scale = config.learnable_ttt_scale
            ttt_prenorm = config.ttt_prenorm
            ttt_nope = config.ttt_nope
            rope_theta = config.rope_theta
            max_position_embeddings = config.max_position_embeddings
            w0_w2_low_rank = config.w0_w2_low_rank
            use_momentum = config.use_momentum
            ttt_loss_type = config.ttt_loss_type
            fw_init_gain = config.fw_init_gain
            r = config.r
            residual_ttt = config.residual_ttt
        else:
            self.config = None
            # Set default values for backward compatibility
            if hidden_size is None: hidden_size = 2048
            if num_attn_heads is None: num_attn_heads = 32
            if num_lact_heads is None: num_lact_heads = 4
            if inter_multi is None: inter_multi = 1
            if window_size is None: window_size = 2048
            if lact_chunk_size is None: lact_chunk_size = 2048
            if qkv_bias is None: qkv_bias = False
            if attn_qk_norm is None: attn_qk_norm = True
            if qkv_silu is None: qkv_silu = True
            if lr_dim is None: lr_dim = 1
            if use_muon is None: use_muon = False
            if lr_parameterization is None: lr_parameterization = "mamba"
            if learnable_ttt_scale is None: learnable_ttt_scale = False
            if ttt_prenorm is None: ttt_prenorm = False
            if ttt_nope is None: ttt_nope = False
            if rope_theta is None: rope_theta = 500000.0
            if max_position_embeddings is None: max_position_embeddings = 2048
            if w0_w2_low_rank is None: w0_w2_low_rank = -1
            if use_momentum is None: use_momentum = False
            if ttt_loss_type is None: ttt_loss_type = "dot_product"
            if fw_init_gain is None: fw_init_gain = 0.5
            if r is None: r = 1
            if residual_ttt is None: residual_ttt = False
        
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attn_heads # num of heads for attention
        self.inter_multi = inter_multi
        self.window_size = window_size
        # head dim for attention
        self.head_dim = hidden_size // num_attn_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        
        self.attn_qk_norm = attn_qk_norm
        if self.attn_qk_norm:
            self.q_norm = RMSNorm(self.hidden_size)
            self.k_norm = RMSNorm(self.hidden_size)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rope_theta = rope_theta
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings

        ### Fast Weight init
        self.use_muon = use_muon
        self.lact_chunk_size = lact_chunk_size
        self.num_fw_heads = num_lact_heads
        self.fw_head_dim = self.hidden_size // self.num_fw_heads
        self.qkv_silu = qkv_silu
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope
        
        d_in, d_out = self.fw_head_dim, self.fw_head_dim
        d_h = int(d_in * inter_multi)

        self.d_h = d_h
        self.d_in = d_in
        self.d_out = d_out
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        self.r = r
        self.residual_ttt = residual_ttt

        # Initialize stacked TTT architecture
        if r == 1:
            # Original single MLP behavior for backward compatibility
            if self.w0_w2_low_rank > 0:
                self.w0 = LowRankFastWeight(self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=self.fw_init_gain, add_identity=True)
                self.w2 = LowRankFastWeight(self.num_fw_heads, d_h, d_in, self.w0_w2_low_rank, init_gain=self.fw_init_gain, add_identity=True)
            else:
                self.w0 = nn.Parameter(
                    torch.randn(self.num_fw_heads, int(d_h), d_in)
                    / math.sqrt(d_in)
                )  # [num_fw_heads, d_h, d_in]
                self.w2 = nn.Parameter(
                    torch.randn(self.num_fw_heads, int(d_h), d_in)
                    / math.sqrt(d_in)
                )  # [num_fw_heads, d_h, d_in]
            self.w1 = nn.Parameter(
                torch.randn(self.num_fw_heads, int(d_out), d_h)
                / math.sqrt(d_h)
            )  # [num_fw_heads, d_out, d_h]
        else:
            # Stacked MLP architecture
            self.stacked_mlps = nn.ModuleList([
                SwiGLUFastWeightBlock(
                    num_heads=self.num_fw_heads,
                    d_h=d_h,
                    d_in=d_in,
                    d_out=d_out,
                    w0_w2_low_rank=self.w0_w2_low_rank,
                    fw_init_gain=self.fw_init_gain,
                )
                for _ in range(r)
            ])
        
        #### Per-Token LR parameterization. 
        self.lr_dim = int(lr_dim * 3 * self.num_fw_heads)
        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim)
        base_lr = 0.001
        # Lr parameterization and initialization
        if lr_parameterization.lower() == "mamba":
            self.base_lr_inv = inv_softplus(base_lr)
        self.lr_parameterization = lr_parameterization
        
        #### per-channel scaling and offset for Q, and K. 
        self.qk_scale = nn.Parameter(torch.ones(hidden_size, 2))
        self.qk_offset = nn.Parameter(torch.zeros(hidden_size, 2))
        self.learnable_ttt_scale = learnable_ttt_scale
        if self.learnable_ttt_scale:
            # per-head scaling. 
            self.ttt_scale_proj = nn.Linear(hidden_size, self.num_fw_heads)
        
        # ttt output norm per head. 
        self.ttt_norm = RMSNorm(self.fw_head_dim, elementwise_affine=True)
        
        self.use_momentum = use_momentum
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(hidden_size, self.num_fw_heads),
                nn.Sigmoid(),
            )

        self.ttt_loss_type = ttt_loss_type
        
        assert self.ttt_loss_type in ["dot_product"], f"Loss type {self.ttt_loss_type} not supported"

        
    def _rescale_qk(self, q, k):
        """
        Args:
            q: [b, s, d]
            k: [b, s, d]
        Returns:
            q: [b, s, d]
            k: [b, s, d]
        """
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        return q, k
    
    def forward(
        self,
        hidden_states: torch.Tensor, # [b, s, d]
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        #### compute window attention first, then do ttt. ####

        if self.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # rescale and reshift the q, k for test-time training layer.
        fast_q, fast_k = self._rescale_qk(q, k)
        fast_v = v
        
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # WARNING: current implementation ignores cu_seqlens for ttt-layer. 
        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        # [b, s, n_h, d]
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(q, k, v, attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
            )
        o = o.reshape(batch_size, q_len, -1)

        ##### TTT starts here. 
        # Split heads then merge it to batch dimension
        fast_q = rearrange(fast_q, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
        fast_k = rearrange(fast_k, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
        fast_v = rearrange(fast_v, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
        

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            fast_v = F.silu(fast_v)

        # per head l2 norm for fast_q, fast_k. 
        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)
        
        if not self.ttt_nope:
            #### Apply rotary embedding.  Here we use the same rope as the attention layer. 
            # I observed that using NoPE for ttt (No positional encoding) here also works. 
            fast_q = rearrange(fast_q, '(b n_h) s d -> b s (n_h d)', n_h=self.num_fw_heads)
            fast_k = rearrange(fast_k, '(b n_h) s d -> b s (n_h d)', n_h=self.num_fw_heads)

            fast_q = rearrange(fast_q, 'b s (n_h d) -> b s n_h d', n_h=self.num_heads)
            fast_k = rearrange(fast_k, 'b s (n_h d) -> b s n_h d', n_h=self.num_heads)

            fast_q, fast_k = self.rotary(fast_q, fast_k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

            fast_q = rearrange(fast_q, 'b s n_h d -> b s (n_h d)', n_h=self.num_heads)
            fast_k = rearrange(fast_k, 'b s n_h d -> b s (n_h d)', n_h=self.num_heads)

            fast_q = rearrange(fast_q, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
            fast_k = rearrange(fast_k, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
            #### RoPE done. ####

        # Prepare learning rates and momentum
        lr = self.lr_proj(hidden_states) # [b, s, num_heads * lr_dim_per_head]
        if self.lr_parameterization == "mamba":
            lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        else:
            raise NotImplementedError(f"LR parameterization {self.lr_parameterization} not implemented")
        fw_lr = rearrange(lr, 'b s (n_h lr_dim) -> (b n_h) s lr_dim', n_h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states) # [b, s, nh]
            momentum = rearrange(momentum, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
        else:
            momentum = None
        
        # Stacked TTT computation
        if hasattr(self, 'config') and hasattr(self.config, 'greedy_ttt') and self.config.greedy_ttt:
            # Legacy greedy TTT approach
            if self.r == 1:
                # Original single MLP behavior
                if self.w0_w2_low_rank > 0:
                    fw_w0 = self.w0().repeat(batch_size, 1, 1)
                    fw_w2 = self.w2().repeat(batch_size, 1, 1)
                else:
                    fw_w0 = self.w0.repeat(batch_size, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
                    fw_w2 = self.w2.repeat(batch_size, 1, 1) # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
                
                fw_w1 = self.w1.repeat(batch_size, 1, 1) # [nh, d_out, d_h] -> [b*nh, d_out, d_h]
                
                # [b * nh, s, d_ttt_head]
                if self.ttt_prenorm:
                    # pre-norm version of ttt.   state = state + f(norm(state))
                    fw_x = prenorm_block_causal_lact_swiglu(
                        fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v,
                        fw_lr1, fw_lr2, fw_lr3,
                        chunk_size=self.lact_chunk_size,
                        use_muon=self.use_muon,
                        momentum=momentum)
                else:
                    # post-norm version of ttt.   state = norm(state + f(state))
                    fw_x = block_causal_lact_swiglu(
                        fw_w0, fw_w1, fw_w2, fast_q, fast_k, fast_v,
                        fw_lr1, fw_lr2, fw_lr3,
                        chunk_size=self.lact_chunk_size,
                        use_muon=self.use_muon,
                        momentum=momentum)
            else:
                # Stacked MLP architecture
                fw_x = fast_q  # Start with the input
                
                for i, mlp in enumerate(self.stacked_mlps):
                    # Prepare weights for this MLP
                    if self.w0_w2_low_rank > 0:
                        fw_w0 = mlp.w0().repeat(batch_size, 1, 1)
                        fw_w2 = mlp.w2().repeat(batch_size, 1, 1)
                    else:
                        fw_w0 = mlp.w0.repeat(batch_size, 1, 1)
                        fw_w2 = mlp.w2.repeat(batch_size, 1, 1)
                    
                    fw_w1 = mlp.w1.repeat(batch_size, 1, 1)
                    
                    # Apply TTT for this MLP
                    if self.ttt_prenorm:
                        mlp_output = prenorm_block_causal_lact_swiglu(
                            fw_w0, fw_w1, fw_w2, fw_x, fast_k, fast_v,
                            fw_lr1, fw_lr2, fw_lr3,
                            chunk_size=self.lact_chunk_size,
                            use_muon=self.use_muon,
                            momentum=momentum)
                    else:
                        mlp_output = block_causal_lact_swiglu(
                            fw_w0, fw_w1, fw_w2, fw_x, fast_k, fast_v,
                            fw_lr1, fw_lr2, fw_lr3,
                            chunk_size=self.lact_chunk_size,
                            use_muon=self.use_muon,
                            momentum=momentum)
                    
                    # Apply residual connection if enabled and not the last MLP
                    if self.residual_ttt and i < len(self.stacked_mlps) - 1:
                        fw_x = fw_x + mlp_output
                    else:
                        fw_x = mlp_output
        else:
            # New end-to-end TTT approach
            fw_x = self._end_to_end_ttt_update(
                fast_q, fast_k, fast_v, fw_lr1, fw_lr2, fw_lr3, momentum
            )
        
        # per-head output norm for ttt layer.
        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale: 
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(ttt_scale, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_fw_heads)
            ttt_x_normed = ttt_x_normed * ttt_scale

        ttt_x_normed = rearrange(ttt_x_normed, '(b n_h) s d -> b s (n_h d)', n_h=self.num_fw_heads)

        o = o + ttt_x_normed
        o = self.o_proj(o)
        
        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values

    def _end_to_end_ttt_update(
        self,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        fw_lr1: torch.Tensor,
        fw_lr2: torch.Tensor,
        fw_lr3: torch.Tensor,
        momentum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        End-to-end Test-Time Training (TTT) update using PyTorch autograd for mathematically correct gradients.
        
        This function implements the core TTT mechanism where fast weights (MLP parameters) are updated
        on-the-fly during the forward pass. Unlike the greedy approach, this uses autograd to compute
        true gradients across the entire computation graph, enabling proper end-to-end training.
        
        Key concepts:
        1. TTT processes sequences in chunks to manage memory
        2. For each chunk, we compute a forward pass, calculate a loss, and update weights
        3. The loss is negative cosine similarity between MLP output and target values
        4. Weight updates use computed gradients with learning rates, momentum, and Muon optimization
        5. Supports both single MLP (r=1) and stacked MLP (r>1) architectures
        
        Args:
            fast_q: [b*nh, s, d] - Query tokens (input to MLP)
            fast_k: [b*nh, s, d] - Key tokens (used as targets for loss computation)
            fast_v: [b*nh, s, d] - Value tokens (used as targets for loss computation)
            fw_lr1, fw_lr2, fw_lr3: [b*nh, s, lr_dim] - Per-token learning rates for w1, w2, w0
            momentum: [b*nh, s, 1] - Momentum values (optional)
            
        Returns:
            output: [b*nh, s, d] - TTT output after processing all chunks
        """
        # Extract dimensions
        batch_size = fast_q.shape[0] // self.num_fw_heads
        seq_len = fast_q.shape[1]
        chunk_size = self.lact_chunk_size
        
        # Initialize output tensor to store results
        output = torch.zeros_like(fast_q)
        
        # Process sequence in chunks (except the last chunk which gets no weight updates)
        e_index = 0
        for i in range(0, seq_len - chunk_size, chunk_size):
            s_index = i
            e_index = s_index + chunk_size
            
            # Extract current chunk data
            ki = fast_k[:, s_index:e_index, :]  # Target keys for this chunk
            vi = fast_v[:, s_index:e_index, :]  # Target values for this chunk
            qi = fast_q[:, s_index:e_index, :]  # Input queries for this chunk
            lr1i = fw_lr1[:, s_index:e_index, :]  # Learning rates for w1
            lr2i = fw_lr2[:, s_index:e_index, :]  # Learning rates for w2
            lr0i = fw_lr3[:, s_index:e_index, :]  # Learning rates for w0
            
            # === FORWARD PASS: Apply current fast weights to queries ===
            f_k = self._apply_fast_weights_to_chunk(qi, batch_size)
            
            # === TTT LOSS COMPUTATION ===
            # Use negative cosine similarity as the TTT loss
            # This encourages the MLP output to be similar to the target values
            loss = -torch.nn.functional.cosine_similarity(f_k, vi.detach(), dim=-1).mean()
            
            # === GRADIENT COMPUTATION AND WEIGHT UPDATES (only in training mode) ===
            if self.training:
                # Collect all fast weight parameters for gradient computation
                all_fast_weights = self._collect_fast_weights()
                
                # Compute gradients using PyTorch autograd
                # create_graph=True enables higher-order gradients for outer-loop training
                grads = torch.autograd.grad(loss, all_fast_weights, create_graph=True)
                
                # Apply weight updates with learning rates, momentum, and Muon
                self._apply_weight_updates(grads, lr0i, lr1i, lr2i, momentum, s_index, e_index)
            
            # Store the forward pass output for this chunk
            output[:, s_index:e_index, :] = f_k
        
        # === HANDLE FINAL CHUNK (no weight updates, just forward pass) ===
        s_index = e_index
        e_index = seq_len
        if s_index < seq_len:
            qi = fast_q[:, s_index:e_index, :]
            f_k = self._apply_fast_weights_to_chunk(qi, batch_size)
            output[:, s_index:e_index, :] = f_k
        
        return output
    
    def _apply_fast_weights_to_chunk(self, qi: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Apply current fast weights to a chunk of queries.
        
        Supports both single MLP (r=1) and stacked MLP (r>1) architectures.
        For stacked MLPs, applies each MLP in sequence with optional residual connections.
        
        Args:
            qi: [b*nh, chunk_size, d] - Input queries for this chunk
            batch_size: Batch size (needed for weight expansion)
            
        Returns:
            f_k: [b*nh, chunk_size, d] - MLP output for this chunk
        """
        if self.r == 1:
            # === SINGLE MLP CASE ===
            # Prepare weights (expand to batch dimension)
            if self.w0_w2_low_rank > 0:
                fw_w0 = self.w0().repeat(batch_size, 1, 1)  # Low-rank w0
                fw_w2 = self.w2().repeat(batch_size, 1, 1)  # Low-rank w2
            else:
                fw_w0 = self.w0.repeat(batch_size, 1, 1)    # Full-rank w0
                fw_w2 = self.w2.repeat(batch_size, 1, 1)    # Full-rank w2
            fw_w1 = self.w1.repeat(batch_size, 1, 1)        # w1 (always full-rank)
            
            # SwiGLU forward pass: w1 @ (silu(w0 @ x) * (w2 @ x))
            h = torch.bmm(fw_w2, qi.transpose(1, 2))        # w2 @ x
            gate = F.silu(torch.bmm(fw_w0, qi.transpose(1, 2)))  # silu(w0 @ x)
            f_k = torch.bmm(fw_w1, gate * h).transpose(1, 2)     # w1 @ (gate * h)
            
        else:
            # === STACKED MLP CASE ===
            f_k = qi  # Start with input
            
            # Apply each MLP in the stack
            for j, mlp in enumerate(self.stacked_mlps):
                # Prepare weights for this MLP
                if self.w0_w2_low_rank > 0:
                    fw_w0 = mlp.w0().repeat(batch_size, 1, 1)
                    fw_w2 = mlp.w2().repeat(batch_size, 1, 1)
                else:
                    fw_w0 = mlp.w0.repeat(batch_size, 1, 1)
                    fw_w2 = mlp.w2.repeat(batch_size, 1, 1)
                fw_w1 = mlp.w1.repeat(batch_size, 1, 1)
                
                # SwiGLU forward pass for this MLP
                h = torch.bmm(fw_w2, f_k.transpose(1, 2))
                gate = F.silu(torch.bmm(fw_w0, f_k.transpose(1, 2)))
                mlp_output = torch.bmm(fw_w1, gate * h).transpose(1, 2)
                
                # Apply residual connection if enabled and not the last MLP
                if self.residual_ttt and j < len(self.stacked_mlps) - 1:
                    f_k = f_k + mlp_output  # Residual connection
                else:
                    f_k = mlp_output        # No residual connection
        
        return f_k
    
    def _collect_fast_weights(self) -> List[torch.Tensor]:
        """
        Collect all fast weight parameters for gradient computation.
        
        Returns:
            all_fast_weights: List of parameter tensors to compute gradients for
        """
        all_fast_weights = []
        
        if self.r == 1:
            # Single MLP case
            if self.w0_w2_low_rank > 0:
                # Low-rank parameterization: collect left and right matrices
                all_fast_weights.extend([
                    self.w0.w_left, self.w0.w_right,  # w0 = w_left @ w_right
                    self.w2.w_left, self.w2.w_right,  # w2 = w_left @ w_right
                    self.w1                           # w1 (always full-rank)
                ])
            else:
                # Full-rank parameterization
                all_fast_weights.extend([self.w0, self.w1, self.w2])
        else:
            # Stacked MLP case: collect weights from all MLPs in the stack
            for mlp in self.stacked_mlps:
                if self.w0_w2_low_rank > 0:
                    all_fast_weights.extend([
                        mlp.w0.w_left, mlp.w0.w_right,
                        mlp.w2.w_left, mlp.w2.w_right,
                        mlp.w1
                    ])
                else:
                    all_fast_weights.extend([mlp.w0, mlp.w1, mlp.w2])
        
        return all_fast_weights
    
    def _apply_weight_updates(
        self,
        grads: List[torch.Tensor],
        lr0i: torch.Tensor,
        lr1i: torch.Tensor,
        lr2i: torch.Tensor,
        momentum: Optional[torch.Tensor],
        s_index: int,
        e_index: int
    ):
        """
        Apply weight updates using computed gradients, learning rates, momentum, and Muon.
        
        Args:
            grads: List of gradient tensors from autograd
            lr0i, lr1i, lr2i: Learning rates for w0, w1, w2
            momentum: Momentum values (optional)
            s_index, e_index: Chunk indices (for momentum computation)
        """
        grad_idx = 0
        
        if self.r == 1:
            # === SINGLE MLP WEIGHT UPDATES ===
            if self.w0_w2_low_rank > 0:
                # Low-rank weight updates
                dw0_left, dw0_right = grads[grad_idx], grads[grad_idx + 1]
                dw2_left, dw2_right = grads[grad_idx + 2], grads[grad_idx + 3]
                dw1 = grads[grad_idx + 4]
                grad_idx += 5
                
                # Apply momentum if enabled
                if momentum is not None:
                    m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)
                    momentum_factor = m_i.mean()
                    dw0_left = dw0_left * momentum_factor
                    dw0_right = dw0_right * momentum_factor
                    dw2_left = dw2_left * momentum_factor
                    dw2_right = dw2_right * momentum_factor
                    dw1 = dw1 * momentum_factor
                
                # Apply Muon orthogonalization if enabled
                if self.use_muon:
                    dw0_left = zeropower_via_newtonschulz5_2d(dw0_left)
                    dw0_right = zeropower_via_newtonschulz5_2d(dw0_right)
                    dw2_left = zeropower_via_newtonschulz5_2d(dw2_left)
                    dw2_right = zeropower_via_newtonschulz5_2d(dw2_right)
                    dw1 = zeropower_via_newtonschulz5_2d(dw1)
                
                # Update weights with learning rates
                self.w0.w_left.data.add_(dw0_left, alpha=-lr0i.mean().item())
                self.w0.w_right.data.add_(dw0_right, alpha=-lr0i.mean().item())
                self.w2.w_left.data.add_(dw2_left, alpha=-lr2i.mean().item())
                self.w2.w_right.data.add_(dw2_right, alpha=-lr2i.mean().item())
                self.w1.data.add_(dw1, alpha=-lr1i.mean().item())
                
            else:
                # Full-rank weight updates
                dw0, dw1, dw2 = grads[grad_idx], grads[grad_idx + 1], grads[grad_idx + 2]
                grad_idx += 3
                
                # Apply momentum if enabled
                if momentum is not None:
                    momentum_factor = momentum[:, s_index:e_index, :].mean()
                    dw0 = dw0 * momentum_factor
                    dw1 = dw1 * momentum_factor
                    dw2 = dw2 * momentum_factor
                
                # Apply Muon orthogonalization if enabled
                if self.use_muon:
                    dw0 = zeropower_via_newtonschulz5_2d(dw0)
                    dw1 = zeropower_via_newtonschulz5_2d(dw1)
                    dw2 = zeropower_via_newtonschulz5_2d(dw2)
                
                # Update weights with learning rates
                self.w0.data.add_(dw0, alpha=-lr0i.mean().item())
                self.w1.data.add_(dw1, alpha=-lr1i.mean().item())
                self.w2.data.add_(dw2, alpha=-lr2i.mean().item())
        
        else:
            # === STACKED MLP WEIGHT UPDATES ===
            for mlp in self.stacked_mlps:
                if self.w0_w2_low_rank > 0:
                    # Low-rank weight updates for this MLP
                    dw0_left, dw0_right = grads[grad_idx], grads[grad_idx + 1]
                    dw2_left, dw2_right = grads[grad_idx + 2], grads[grad_idx + 3]
                    dw1 = grads[grad_idx + 4]
                    grad_idx += 5
                    
                    # Apply momentum if enabled
                    if momentum is not None:
                        momentum_factor = momentum[:, s_index:e_index, :].mean()
                        dw0_left = dw0_left * momentum_factor
                        dw0_right = dw0_right * momentum_factor
                        dw2_left = dw2_left * momentum_factor
                        dw2_right = dw2_right * momentum_factor
                        dw1 = dw1 * momentum_factor
                    
                    # Apply Muon orthogonalization if enabled
                    if self.use_muon:
                        dw0_left = zeropower_via_newtonschulz5_2d(dw0_left)
                        dw0_right = zeropower_via_newtonschulz5_2d(dw0_right)
                        dw2_left = zeropower_via_newtonschulz5_2d(dw2_left)
                        dw2_right = zeropower_via_newtonschulz5_2d(dw2_right)
                        dw1 = zeropower_via_newtonschulz5_2d(dw1)
                    
                    # Update weights with learning rates
                    mlp.w0.w_left.data.add_(dw0_left, alpha=-lr0i.mean().item())
                    mlp.w0.w_right.data.add_(dw0_right, alpha=-lr0i.mean().item())
                    mlp.w2.w_left.data.add_(dw2_left, alpha=-lr2i.mean().item())
                    mlp.w2.w_right.data.add_(dw2_right, alpha=-lr2i.mean().item())
                    mlp.w1.data.add_(dw1, alpha=-lr1i.mean().item())
                    
                else:
                    # Full-rank weight updates for this MLP
                    dw0, dw1, dw2 = grads[grad_idx], grads[grad_idx + 1], grads[grad_idx + 2]
                    grad_idx += 3
                    
                    # Apply momentum if enabled
                    if momentum is not None:
                        momentum_factor = momentum[:, s_index:e_index, :].mean()
                        dw0 = dw0 * momentum_factor
                        dw1 = dw1 * momentum_factor
                        dw2 = dw2 * momentum_factor
                    
                    # Apply Muon orthogonalization if enabled
                    if self.use_muon:
                        dw0 = zeropower_via_newtonschulz5_2d(dw0)
                        dw1 = zeropower_via_newtonschulz5_2d(dw1)
                        dw2 = zeropower_via_newtonschulz5_2d(dw2)
                    
                    # Update weights with learning rates
                    mlp.w0.data.add_(dw0, alpha=-lr0i.mean().item())
                    mlp.w1.data.add_(dw1, alpha=-lr1i.mean().item())
                    mlp.w2.data.add_(dw2, alpha=-lr2i.mean().item())

    def _upad_input(self, q, k, v, attention_mask, q_len):
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        cache_mask = attention_mask[:, -seq_len:]
        seqlens = cache_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)

        
        
        