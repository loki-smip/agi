import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import argparse
import os
import sys
from tqdm import tqdm

# Make wandb optional
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import RecursiveLoopConfig
from model.model import RecursiveLoopTransformer
from training.loss import ACTLoss

# Real Dataset
class RealDataset(Dataset):
    def __init__(self, data_path, seq_len=1024):
        print(f"Loading data from {data_path}...")
        self.data = torch.load(data_path) # List of tensors
        self.seq_len = seq_len
        print(f"Loaded {len(self.data)} sequences.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already tokenized ids
        tokens = self.data[idx]
        if len(tokens) > self.seq_len + 1:
            tokens = tokens[:self.seq_len + 1]
        elif len(tokens) < self.seq_len + 1:
            # Pad if needed (simple padding for demo)
            pad_len = (self.seq_len + 1) - len(tokens)
            tokens = torch.cat([tokens, torch.zeros(pad_len, dtype=torch.long)])
            
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

# Mock Dataset Fallback
class MockDataset(Dataset):
    def __init__(self, tokenizer, length=1000, seq_len=128):
        self.tokenizer = tokenizer
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

def train(args):
    # Config
    config = RecursiveLoopConfig(
        d_model=args.d_model, 
        n_layers=args.n_layers, 
        loops=args.loops,
        max_seq_len=args.seq_len
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Model
    model = RecursiveLoopTransformer(config).to(device)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Data Loading Logic
    data_path = os.path.join(args.data_dir, "pretrain_data.pt")
    if os.path.exists(data_path):
        print(f"Found processed data at {data_path}. Using RealDataset.")
        # Ensure config matches data
        dataset = RealDataset(data_path, seq_len=config.max_seq_len)
    else:
        print(f"No data found at {data_path}. Using MockDataset (Synthetic Random Data).")
        dataset = MockDataset(tokenizer, length=1000, seq_len=config.max_seq_len)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loss
    criterion = ACTLoss(tau=args.tau)

    # Logging
    if args.wandb and HAS_WANDB:
        wandb.init(project="recursive-loop-transformer", config=args)
    elif args.wandb and not HAS_WANDB:
        print("Warning: --wandb requested but wandb module not found. Skipping logging.")

    model.train()
    
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            logits, ponder_cost = model(x, return_ponder_loss=True)
            loss, l_xent, l_ponder = criterion(logits, y, ponder_cost)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            
            # Logging
            ponder_val = l_ponder.item() if isinstance(l_ponder, torch.Tensor) else l_ponder
            metrics = {
                "loss": loss.item(),
                "xent": l_xent.item(),
                "ponder_cost": ponder_val
            }
            pbar.set_postfix(metrics)
            
            if args.wandb and HAS_WANDB:
                wandb.log(metrics)
                
            if step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.01, help="Ponder cost weight")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for training")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
