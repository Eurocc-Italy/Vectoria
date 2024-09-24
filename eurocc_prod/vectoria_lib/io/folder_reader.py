import os
from pathlib import Path

def get_files_in_folder(folder_path: Path, limit: int = -1) -> list[Path]:
        
        files = sorted([
            folder_path / f for f in os.listdir(folder_path)
        ])

        if limit > 0:
            files = files[:limit]

        return files