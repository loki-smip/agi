import torch
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import RecursiveLoopConfig
from model.model import RecursiveLoopTransformer
from inference.repl_sandbox import PythonREPL

def test_model_forward():
    print("\n[Test 1] Model Forward Pass with Looping")
    config = RecursiveLoopConfig(
        d_model=128, n_layers=2, loops=3, n_head=4, vocab_size=1000
    )
    model = RecursiveLoopTransformer(config)
    
    # Fake tokens
    x = torch.randint(0, 1000, (2, 10)) # bsz=2, seq=10
    
    logits, ponder_cost = model(x, return_ponder_loss=True)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Ponder cost: {ponder_cost.item()}")
    
    assert logits.shape == (2, 10, 1000)
    print(">> Success: Forward pass complete.")

def test_repl_sandbox():
    print("\n[Test 2] REPL Sandbox")
    repl = PythonREPL(context_data="Secret Data")
    
    out1 = repl.execute("print(context)")
    print(f"Code: print(context) -> {out1}")
    assert "Secret Data" in out1
    
    repl.execute("x = 100")
    out2 = repl.execute("print(x + 50)")
    print(f"Code: print(x + 50) -> {out2}")
    assert "150" in out2
    
    print(">> Success: REPL works.")

if __name__ == "__main__":
    try:
        test_model_forward()
        test_repl_sandbox()
        print("\nAll Verification Tests Passed.")
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        import traceback
        traceback.print_exc()
