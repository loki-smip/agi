import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import re
from typing import List, Optional
import sys
import os

# Add parent dir to path to find model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import RecursiveLoopTransformer
from model.config import RecursiveLoopConfig
from inference.repl_sandbox import PythonREPL

class RecursiveInferenceEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.config = RecursiveLoopConfig()
        # Initialize model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RecursiveLoopTransformer(self.config).to(self.device)
        self.model.eval()
        
        # Initialize tokenizer (using GPT2 as base for simplicity)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': ['<SELF_CALL>', '<END_SELF_CALL>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.output = torch.nn.Linear(self.config.d_model, len(self.tokenizer), bias=False).to(self.device)
        self.model.tok_embeddings = torch.nn.Embedding(len(self.tokenizer), self.config.d_model).to(self.device)

        self.self_call_token_id = self.tokenizer.convert_tokens_to_ids('<SELF_CALL>')
        self.end_self_call_token_id = self.tokenizer.convert_tokens_to_ids('<END_SELF_CALL>')
        
        print(f"Model initialized on {self.device}. Special tokens added.")

    def generate(self, prompt: str, context_data: Optional[str] = None, max_new_tokens: int = 200):
        # Initialize REPL with context
        repl = PythonREPL(context_data)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        generated_ids = input_ids
        
        print(f"\n--- Generating for prompt: {prompt} ---\n")
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                logits = self.model(generated_ids)
                next_token_logits = logits[:, -1, :]
            
            # Simple greedy sampling
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Check for <SELF_CALL>
            if next_token.item() == self.self_call_token_id:
                print("\n[<SELF_CALL> Triggered]")
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                code_segment = ""
                # Generate code loop
                while True:
                    with torch.no_grad():
                         logits = self.model(generated_ids)
                         next_code_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                    
                    if next_code_token.item() == self.end_self_call_token_id:
                        generated_ids = torch.cat([generated_ids, next_code_token], dim=-1)
                        break
                    
                    generated_ids = torch.cat([generated_ids, next_code_token], dim=-1)
                    code_segment += self.tokenizer.decode(next_code_token[0])
                    
                    if len(code_segment) > 1000: # Safety break
                        break

                print(f"[Executing Code]:\n{code_segment}")
                
                # Execute code
                result = repl.execute(code_segment)
                print(f"[REPL Output]:\n{result}")
                
                # Append result to context
                result_text = f"\nObservation:\n{result}\n"
                result_ids = self.tokenizer.encode(result_text, return_tensors="pt").to(self.device)
                generated_ids = torch.cat([generated_ids, result_ids], dim=-1)
                
                continue
            
            # Normal token generation
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Print token (streaming effect)
            word = self.tokenizer.decode(next_token[0])
            print(word, end="", flush=True)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        return self.tokenizer.decode(generated_ids[0])

if __name__ == "__main__":
    engine = RecursiveInferenceEngine()
    
    # Mock context
    huge_context = "Alice has 5 apples. Bob has 3 apples. The secret code is 12345."
    
    try:
        output = engine.generate("How many apples does Alice have?", context_data=huge_context, max_new_tokens=20)
        print("\n\n[Final Output]:", output)
    except Exception as e:
        print(f"\nError: {e}")
