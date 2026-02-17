import argparse
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import os
import sys
import torch

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import RecursiveLoopConfig

def prepare_data(args):
    print(f"Preparing data for Phase I in {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Generate Dummy Data locally to avoid network/dataset issues
    print("Generating local dummy data...")
    data_buffer = []
    
    base_text = "The quick brown fox jumps over the lazy dog. "
    code_text = "def hello_world():\n    print('Hello World')\n"
    
    for i in range(args.num_samples):
        if i % 2 == 0:
            data_buffer.append(base_text * 50)
        else:
            data_buffer.append(code_text * 20)
            
    print(f"Total samples generated: {len(data_buffer)}")
    
    # Tokenize
    print("Tokenizing...")
    tokenized_data = []
    for text in data_buffer:
        ids = tokenizer.encode(text, truncation=True, max_length=1024)
        tokenized_data.append(torch.tensor(ids))
        
    # Save as a simple torch pt file 
    output_path = os.path.join(args.output_dir, "pretrain_data.pt")
    torch.save(tokenized_data, output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    prepare_data(args)
