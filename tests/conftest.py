import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parents[1])
src_dir = str(Path(__file__).resolve().parents[1] / "src")

while root_dir in sys.path:
    sys.path.remove(root_dir)
while src_dir in sys.path:
    sys.path.remove(src_dir)

sys.path.insert(0, src_dir)
sys.path.append(root_dir)

if "pipeline" in sys.modules and not hasattr(sys.modules["pipeline"], "__path__"):
    del sys.modules["pipeline"]
