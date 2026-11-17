"""Keep optional model/runtime caches inside an explicitly chosen directory."""
import os
from pathlib import Path


def configure_model_caches(root):
    root=Path(root).resolve()
    for variable,folder in {
        'TEMP':'temporary','TMP':'temporary','HF_HOME':'huggingface','TORCH_HOME':'torch',
        'RF_HOME':'rfdetr','YOLO_CONFIG_DIR':'ultralytics','MPLCONFIGDIR':'matplotlib',
        'XDG_CACHE_HOME':'cache','NUMBA_CACHE_DIR':'numba',
    }.items():
        path=root/folder
        path.mkdir(parents=True,exist_ok=True)
        os.environ[variable]=str(path)
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'

