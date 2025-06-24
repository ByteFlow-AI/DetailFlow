import re
import json
from pathlib import Path
import ruamel.yaml


__all__ = ["sort_filenames", "json_dump", "json_load", "yaml_load", "get_file_from_folder"]

def extract_numbers(filename):
    return list(map(int, re.findall(r'\d+', filename)))

def sort_filenames(file_paths):
    return sorted(file_paths, key=lambda x: extract_numbers(x))


def json_dump(data, file_path):
    Path(file_path).parent.mkdir(exist_ok=True, parents=True)
    with open(str(file_path), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def json_load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def yaml_load(path):
    yaml = ruamel.yaml.YAML()
    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.load(file)
    return data


def get_file_from_folder(data_folder, file_suffix):
    if isinstance(file_suffix, str):
        file_suffix = [file_suffix]
    data_folder = Path(data_folder)
    assert data_folder.is_dir()
    img_list = [str(k) for k in sorted(data_folder.glob("*.*")) if k.suffix.lower() in file_suffix]

    return sorted(img_list)
