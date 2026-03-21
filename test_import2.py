import sys
import importlib

# First, let's try to manually execute the utils module code
print("Reading utils.py file...")
with open("src/utils.py", "r") as f:
    code = f.read()
    print(f"File contents:\n{code}\n")

print("Attempting to compile the file...")
try:
    compiled = compile(code, "src/utils.py", "exec")
    print("Compilation successful")
except Exception as e:
    print(f"Compilation error: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to execute the code in a namespace...")
namespace = {}
try:
    exec(compiled, namespace)
    print("Execution successful")
    print(f"Namespace contents: {[x for x in namespace.keys() if not x.startswith('_')]}")
except Exception as e:
    print(f"Execution error: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to actually import...")
try:
    # Force reload
    if 'src.utils' in sys.modules:
        del sys.modules['src.utils']
    import src.utils
    print(f"Import successful, module: {src.utils}")
    print(f"Module attributes: {[x for x in dir(src.utils) if not x.startswith('_')]}")
except Exception as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
