import torch
import torch.nn as nn

class ACTLoss(nn.Module):
    """
    Combined Loss for Recursive Loop Transformer.
    L_total = L_xent + tau * L_ponder
    """
    def __init__(self, tau: float = 0.01, pad_token_id: int = -100):
        super().__init__()
        self.tau = tau
        self.xent = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        
    def forward(self, logits, targets, ponder_loss=None):
        vocab_size = logits.size(-1)
        l_xent = self.xent(logits.view(-1, vocab_size), targets.view(-1))
        
        l_ponder = 0.0
        if ponder_loss is not None:
            l_ponder = ponder_loss
            
        total_loss = l_xent + self.tau * l_ponder
        
        return total_loss, l_xent, l_ponder
