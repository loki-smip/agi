import torch
import argparse
import os
import sys
import json
import random

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inference import RecursiveInferenceEngine
from inference.repl_sandbox import PythonREPL

def generate_self_correction_data(args):
    print(f"Starting SEAL Data Generation (Self-Correction)...")
    
    # Initialize Engine (Mocked/Random if untrained)
    engine = RecursiveInferenceEngine()
    
    # We need a "Generator" prompt to ask the model to create a problem
    # In a real SEAL implementation, we'd have a specific "Problem Generator" prompt.
    
    problem_templates = [
        "Write a Python function to calculate the factorial of {n}.",
        "Solve the equation x^2 - {b}x + {c} = 0 using Python.",
        "Write a script to sort a list of {n} random numbers."
    ]
    
    valid_samples = []
    
    for i in range(args.num_samples):
        # 1. Generate a Problem
        template = random.choice(problem_templates)
        n = random.randint(5, 20)
        b = random.randint(2, 10)
        c = random.randint(1, 20)
        prompt = template.format(n=n, b=b, c=c)
        
        print(f"\n[Sample {i}] Prompt: {prompt}")
        
        # 2. Model Generates Solution (with potential Self-Call)
        # For this script to work with an untrained model, we can't expect valid code.
        # But this structure allows the USER to run it once they have a trained model.
        
        # We simulate the "Inference" step
        try:
            # We use a dummy context as placeholder
            response = engine.generate(prompt, context_data=None, max_new_tokens=200)
            
            # 3. Verify (SEAL / Verifiable Reward)
            # We extract code blocks from the response
            # In our specialized model, code is inside <SELF_CALL> ... <END_SELF_CALL> 
            # or just standard markdown blocks if trained that way.
            # Let's assume the Inference Engine handles the execution and leaves "Observation".
            
            # For "Self-Correction", we need to check if the Final Answer is correct.
            # This is hard for generic tasks without a ground truth.
            # SEAL often uses "Consistency" or "Unit Tests".
            
            # Let's verify by checking if the REPL output contains no errors.
            if "Error:" not in response and "Output:" in response:
                print(">> Verification Passed (No Runtime Errors)")
                valid_samples.append({
                    "prompt": prompt,
                    "response": response,
                    "reward": 1.0
                })
            else:
                print(">> Verification Failed (Runtime Error or No Output)")
                
        except Exception as e:
            print(f"generation failed: {e}")
            
    # Save Valid Samples
    output_path = os.path.join(args.output_dir, "seal_self_correction.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")
            
    print(f"Saved {len(valid_samples)} valid samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/seal")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    generate_self_correction_data(args)
