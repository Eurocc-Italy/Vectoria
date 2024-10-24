from pathlib import Path
import time
import json
import yaml


class BaseIO():

    def write(
            self, 
            data: dict, 
            output_root_path: str | Path, 
            name: str, 
            add_time_stamp: bool = True
        ) -> Path:
        """
        Write data to a file.

        Args:
            data (dict): Data to write to the file.
            output_root_path (str | Path): Path to the root directory where the file will be saved.
            name (str): Name of the file.
            add_time_stamp (bool): Whether to add a timestamp to the file name.

        Returns:
            Path: Path to the saved file.
        """
        pass

class JsonIO(BaseIO):
    def read(self, file_path: str | Path) -> dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def write(self, data: dict, output_root_path: str | Path, name: str, add_time_stamp: bool = True) -> Path:
        output_path = Path(output_root_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if add_time_stamp:
            name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"

        name = f"{name}.json"
        with open(output_path / name, 'w', encoding='utf-8') as file:
            json.dump(data, file)

        return output_path / name

class YamlIO(BaseIO):
    def read(self, file_path: str | Path) -> dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def write(self, data: dict, output_root_path: str | Path, name: str, add_time_stamp: bool = True) -> Path:  
        output_path = Path(output_root_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if add_time_stamp:
            name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"

        name = f"{name}.yaml"
        with open(output_path / name, 'w', encoding='utf-8') as file:
            yaml.dump(data, file)

        return output_path / name

class TxtIO(BaseIO):
    def read(self, file_path: str | Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def write(self, data: dict, output_root_path: str | Path, name: str, add_time_stamp: bool = True) -> Path:
        output_path = Path(output_root_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if add_time_stamp:
            name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"

        name = f"{name}.txt"
        with open(output_path / name, 'w', encoding='utf-8') as file:
            file.write(str(data))

        return output_path / name   

def get_file_io(
    file_format: str
) -> BaseIO:
    if file_format == "json":
        return JsonIO()
    elif file_format == "yaml":
        return YamlIO()
    elif file_format == "txt":
        return TxtIO()
    else:
        raise ValueError(f"Invalid file format: {file_format}")