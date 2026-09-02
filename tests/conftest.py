import os
import sys

# Allow tests to import from scripts/ (e.g. `from generate_orders import ...`)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
