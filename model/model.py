import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

from .config import RecursiveLoopConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, config: RecursiveLoopConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.wq = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose for attention: (bsz, n_head, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Efficient SDPA (Flash Attention where available)
        # mask is (1, 1, seqlen, seqlen) additive mask from forward()
        # SDPA expects distinct shapes depending on implementation, but additive mask works.
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=mask, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False # We handle causality via the mask passed in
        )
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, config: RecursiveLoopConfig):
        super().__init__()
        hidden_dim = 4 * config.d_model
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class AdaptiveRouter(nn.Module):
    """
    Adaptive Computation Time (ACT) Router.
    Output: Halting probability p_h for the current step.
    """
    def __init__(self, config: RecursiveLoopConfig):
        super().__init__()
        self.linear = nn.Linear(config.d_model, 1, bias=config.router_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (bsz, seqlen, d_model)
        logits = self.linear(x).squeeze(-1) # (bsz, seqlen)
        p_h = self.sigmoid(logits)
        return p_h

class RecursiveBlock(nn.Module):
    def __init__(self, config: RecursiveLoopConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model, eps=config.eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.eps)
        self.router = AdaptiveRouter(config)
        
    def forward(self, x, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class RecursiveLoopTransformer(nn.Module):
    def __init__(self, config: RecursiveLoopConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([RecursiveBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.freqs_cis = self.precompute_freqs_cis(
            config.d_model // config.n_head, config.max_seq_len * 2
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device='cpu')
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, return_ponder_loss=False):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.view(1, 1, seqlen, seqlen) # Broadcast over heads

        # ACT State
        halting_probability = torch.zeros((bsz, seqlen), device=tokens.device)
        recurrence_state = torch.zeros_like(h) # To accumulate weighted states
        accumulated_output = torch.zeros_like(h)
        n_updates = torch.zeros((bsz, seqlen), device=tokens.device)
        
        still_running_mask = torch.ones((bsz, seqlen), device=tokens.device)
        
        # We perform `loops` cycles
        # In each cycle, we pass through ALL `n_layers`
        # ACT Check is performed at the END of each cycle (as per prompt/plan)
        
        current_state = h
        
        for loop_idx in range(self.config.loops):
            # Early break if everything halted (inference optimization)
            if not self.training and (still_running_mask == 0).all():
                break

            # 1. Forward through the unique stack
            for layer in self.layers:
                current_state = layer(current_state, freqs_cis, mask)
            
            # 2. Router Check (using the last layer's router, or a dedicated one?)
            # I'll use the LAST layer's router for the loop exit decision.
            # This implies the last layer "knows" if it should exit.
            p_h = self.layers[-1].router(current_state) # (bsz, seqlen)
            
            # 3. ACT Accumulation Logic
            # If prompt says "adaptive gating at end of each loop", this is correct.
            
            # halting_probability accumulated so far
            # If already halted, new contributions are zeroed by mask.
            
            # Calculate contribution for this step:
            # effective_p = p_h if (cum_prob + p_h < 1) else (1 - cum_prob)
            
            new_halted = halting_probability + p_h
            
            # Weights for this step's contribution to the final output
            step_weight = p_h * still_running_mask
            
            # Clamp if overshooting 1.0
            overshoot = (new_halted > 1.0)
            step_weight = torch.where(overshoot, 1.0 - halting_probability, step_weight)
            
            # Update accumulated output
            # (In standard ACT, output is weighted sum of intermediate states)
            # Here: accumulated_output += current_state * step_weight
            accumulated_output = accumulated_output + current_state * step_weight.unsqueeze(-1)
            
            # Update halting prob
            halting_probability = halting_probability + step_weight
            
            # Update running mask
            # If halted (prob >= 1.0), mask becomes 0
            # Note: overshoot handles exactly hitting 1.0
            still_running_mask = (halting_probability < 1.0 - 1e-6).float()
            
            n_updates = n_updates + still_running_mask

            # If it's the LAST loop, forces remaining probability to use this state
            if loop_idx == self.config.loops - 1:
                remainder = 1.0 - halting_probability
                accumulated_output = accumulated_output + current_state * remainder.unsqueeze(-1)
                halting_probability = torch.ones_like(halting_probability) 

        # Final Norm
        final_state = self.norm(accumulated_output)
        logits = self.output(final_state)
        
        if return_ponder_loss:
            ponder_loss = n_updates.mean()
            return logits, ponder_loss
            
        return logits, None
