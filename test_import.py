import sys
print("Python paths:")
for p in sys.path[:5]:
    print(f"  {p}")

print("\nimporting src...")
import src
print(f"src module: {src}")

print("\nimporting src.exception...")
import src.exception
print(f"src.exception module: {src.exception}")
print(f"src.exception contents: {[x for x in dir(src.exception) if not x.startswith('_')]}")

print("\nimporting src.logger...")
import src.logger
print(f"src.logger loaded")

print("\nAttempting to import src.utils...")
try:
    import src.utils
    print(f"src.utils loaded successfully")
    print(f"Contents of src.utils: {[x for x in dir(src.utils) if not x.startswith('_')]}")
    
    print("\nAttempting to import save_object from src.utils...")
    from src.utils import save_object
    print(f"save_object imported: {save_object}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
