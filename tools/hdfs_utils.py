import os
import subprocess
import shutil
from typing import List

__all__ = ['list_files']


def is_hdfs_path(path):
    return path.startswith("hdfs://")


def listdir(path: str) -> List[str]:
    """
    List directory. Supports either hdfs or local path. Returns full path.

    Examples:
        - listdir("hdfs://dir") -> ["hdfs://dir/file1", "hdfs://dir/file2"]
        - listdir("/dir") -> ["/dir/file1", "/dir/file2"]
    """
    files = []

    if is_hdfs_path(path):
        pipe = subprocess.Popen(
            args=["hdfs", "dfs", "-ls", path],
            shell=False,
            stdout=subprocess.PIPE)

        for line in pipe.stdout:
            parts = line.strip().split()

            # drwxr-xr-x   - user group  4 file
            if len(parts) < 5:
                continue

            files.append(parts[-1].decode("utf8"))

        pipe.stdout.close()
        pipe.wait()

    else:
        files = [os.path.join(path, file) for file in os.listdir(path)]

    return files


def mkdir(path: str):
    """
    Create directory. Support either hdfs or local path.
    Create all parent directory if not present. No-op if directory already present.
    """
    if is_hdfs_path(path):
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", path])
    else:
        os.makedirs(path, exist_ok=True)


def copy(src: str, tgt: str):
    """
    Copy file. Source and destination supports either hdfs or local path.
    """
    src_hdfs = is_hdfs_path(src)
    tgt_hdfs = is_hdfs_path(tgt)

    if src_hdfs and tgt_hdfs:
        subprocess.run(["hdfs", "dfs", "-cp", "-f", src, tgt])
    elif src_hdfs and not tgt_hdfs:
        if os.path.exists(tgt):
            # hdfs no longer support "-f" option for local path, so we manually remove it
            os.remove(tgt)
        subprocess.run(["hdfs", "dfs", "-copyToLocal", src, tgt])
    elif not src_hdfs and tgt_hdfs:
        subprocess.run(["hdfs", "dfs", "-copyFromLocal", "-f", src, tgt])
    else:
        shutil.copy(src, tgt)


def delete(tgt):
    """
    Copy file. Source and destination supports either hdfs or local path.
    """
    tgt_hdfs = is_hdfs_path(tgt)
    if tgt_hdfs:
        subprocess.run(["hdfs", "dfs", "-rm", "-r", tgt])
    else:
        subprocess.run(["rm", "-r", tgt])


def exists(file_path: str, re=False) -> bool:
    """ 
    hdfs capable to check whether a file_path is exists 
    hdfs will return 0 if path exists
    """
    if is_hdfs_path(file_path):
        if re:
            return subprocess.run(["hdfs", "dfs", "-test", "-e", file_path+"*"], capture_output=True).returncode == 0
        return subprocess.run(["hdfs", "dfs", "-test", "-e", file_path], capture_output=True).returncode == 0
    return os.path.exists(file_path)


def list_files(folders: List[str]) -> List[str]:
    """
    Given a list of hdfs path, return all kvfiles under all the paths
    """
    files = []
    try:
        for folder in folders:
            files += listdir(folder)
    except:
        pass
    return files


def is_file(path):
    if is_hdfs_path(path):
        try:
            result = subprocess.run(["hdfs", "dfs", "-test", "-f", path], capture_output=True)
            return result.returncode == 0
        except Exception as e:
            return False
    else:
        return os.path.isfile(path)
    

def normalize_path(path):
    if is_hdfs_path(path):
        
        prefix = "hdfs://"
        path_without_prefix = path[len(prefix):]
        parts = path_without_prefix.split('/')
        normalized_parts = []
        for part in parts:
            if part == '' or part == '.':
                continue
            elif part == '..':
                if normalized_parts:
                    normalized_parts.pop()
            else:
                normalized_parts.append(part)
        normalized_path = '/'.join(normalized_parts)
        return prefix + normalized_path
    else:
        return os.path.normpath(path)