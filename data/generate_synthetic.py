import random
import json
import os
import argparse

def generate_arithmetic_chain(num_samples=100):
    """
    Generates A + B + C chains with reasoning.
    Format: "Compute 12 + 45 + 3. Thought: 12 + 45 is 57. 57 + 3 is 60. Answer: 60"
    """
    samples = []
    for _ in range(num_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        c = random.randint(1, 100)
        
        prompt = f"Compute {a} + {b} + {c}."
        thought1 = f"{a} + {b} is {a+b}."
        thought2 = f"{a+b} + {c} is {a+b+c}."
        answer = f"Answer: {a+b+c}"
        
        full_text = f"{prompt} Thought: {thought1} {thought2} {answer}"
        samples.append({"text": full_text})
        
    return samples

def generate_recursion_trigger(num_samples=100):
    """
    Generates samples that explicitly require <SELF_CALL> (mocked).
    """
    samples = []
    for _ in range(num_samples):
        filename = f"file_{random.randint(1,100)}.txt"
        prompt = f"Read the first 100 bytes of {filename}."
        # The model should generate:
        # <SELF_CALL> f = open('{filename}', 'rb'); output = f.read(100); print(output) <END_SELF_CALL>
        # Observation:...
        
        target_code = f"<SELF_CALL> print(open('{filename}').read(100)) <END_SELF_CALL>"
        full_text = f"{prompt} {target_code}"
        samples.append({"text": full_text})
        
    return samples

def main(args):
    print(f"Generating synthetic data to {args.output_file}")
    
    data = []
    data.extend(generate_arithmetic_chain(args.num_samples))
    data.extend(generate_recursion_trigger(args.num_samples // 2))
    
    random.shuffle(data)
    
    with open(args.output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="data/sft_synthetic.jsonl")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    main(args)
