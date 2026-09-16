import os
import sys

# ComfyUI's legacy loader imports this package with a flat module name, so
# relative imports inside the modules would fail. Put this package dir on
# sys.path and import by unique names (never shadow ComfyUI's own `nodes`).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Register a folder for adapter weights so they show up in the loader and the
# `adapter_v14_bal.safetensors` name resolves via folder_paths.
try:
    import folder_paths
    folder_paths.add_model_folder_path(
        "klein_adapter", os.path.join(folder_paths.models_dir, "klein_adapter"))
except Exception:
    pass

from klein_adapter_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
