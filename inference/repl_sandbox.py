import sys
import io
import contextlib
import traceback
from typing import Dict, Any

class PythonREPL:
    """
    A persistent Python REPL loop that maintains state (variables) across calls.
    Used by the Recursive Loop Transformer to 'think' by executing code.
    """
    def __init__(self, context_data: Any = None):
        self.locals: Dict[str, Any] = {}
        self.buffer = io.StringIO()
        
        # Inject context if provided
        if context_data is not None:
            self.locals['context'] = context_data
            
    def execute(self, code: str) -> str:
        """
        Executes the provided Python code and returns the stdout + stderr.
        """
        # Capture stdout, stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(code, {}, self.locals)
            
            output = stdout_capture.getvalue()
            error = stderr_capture.getvalue()
            
            result_str = ""
            if output:
                result_str += f"Output:\n{output}\n"
            if error:
                result_str += f"Error:\n{error}\n"
                
            if not result_str:
                result_str = "Output: (No output)\n"
                
            return result_str.strip()
            
        except Exception:
            return f"Error Traceback:\n{traceback.format_exc()}"

    def add_variable(self, name: str, value: Any):
        self.locals[name] = value

    def clear(self):
        self.locals = {}

if __name__ == "__main__":
    repl = PythonREPL(context_data="The quick brown fox jumps over the lazy dog.")
    print(repl.execute("print(context[:10])"))
    print(repl.execute("x = 5"))
    print(repl.execute("print(x * 2)"))
