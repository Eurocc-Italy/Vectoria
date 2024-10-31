import os
from pathlib import Path
from vectoria_lib.common.constants import ALLOWED_EXTENSIONS
def get_files_in_folder(folder_path: Path, limit: int = -1) -> list[Path]:
        
    files = sorted([
        folder_path / f for f in os.listdir(folder_path) if Path(f).suffix in ALLOWED_EXTENSIONS
    ])

    if limit > 0:
        files = files[:limit]

    return files