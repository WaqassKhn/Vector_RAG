import sys
import os
from pathlib import Path

# Add project root to sys.path for test discovery
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Prevent transformers from loading broken legacy tensorflow installation
# Setting sys.modules["tensorflow"] = None makes `import tensorflow` raise ModuleNotFoundError cleanly
sys.modules["tensorflow"] = None
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
