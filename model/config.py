from dataclasses import dataclass
from typing import Optional

@dataclass
class RecursiveLoopConfig:
    d_model: int = 1536  # Base dimension for ~1.5B 
    n_head: int = 24     # Number of attention heads
    n_layers: int = 8    # Number of UNIQUE layers (weight-shared)
    loops: int = 4       # Number of times to loop through the unique layers 
                         # Total effective depth = n_layers * loops = 32
    
    vocab_size: int = 50257 # Using GPT2 tokenizer for ease
    max_seq_len: int = 4096 # Context window
    dropout: float = 0.1
    
    # Adaptive Gating
    router_hidden_dim: int = 256 # For the adaptive gating router
    router_bias: bool = True     # Whether to include bias in router
    
    # RoPE
    rope_theta: float = 10000.0  # RoPE base frequency
    
    # Implementation Specifics
    eps: float = 1e-6 # LayerNorm epsilon
    swiglu: bool = True # Use SwiGLU activation
    
    # Recursive interface
    self_call_token_id: Optional[int] = None # Will be set after tokenizer creation
    repl_timeout: int = 5 # Seconds for REPL execution
    
    def __post_init__(self):
        # Effective depth calculation for scaling laws if interested
        self.effective_depth = self.n_layers * self.loops
