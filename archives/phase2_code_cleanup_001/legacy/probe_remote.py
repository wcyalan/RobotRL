"""只读远程环境探针；CUDA 小矩阵测试，不安装软件、不训练。"""
import importlib.metadata
import json
import os
import platform
import subprocess
import ctypes.util
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as error:
        return {"error": str(error)}


result = {"hostname": platform.node(), "python": platform.python_version(),
          "os_release": Path('/etc/os-release').read_text(),
          "gpu": command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv']),
          "driver_capabilities": os.environ.get('NVIDIA_DRIVER_CAPABILITIES'),
          "vulkan_library": ctypes.util.find_library('vulkan'),
          "egl_library": ctypes.util.find_library('EGL'),
          "vulkan_icds": [str(p) for base in ['/etc/vulkan/icd.d','/usr/share/vulkan/icd.d']
                          for p in Path(base).glob('*.json')],
          "shared_memory": command(['df', '-h', '/dev/shm']), "packages": {}}
for name in ['torch','isaacsim','isaaclab','rsl-rl-lib']:
    try:
        result['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result['packages'][name] = None
try:
    import torch
    result['cuda_available'] = torch.cuda.is_available()
    result['torch_cuda_version'] = torch.version.cuda
    if result['cuda_available']:
        x = torch.ones((128,128), device='cuda')
        y = x @ x
        torch.cuda.synchronize()
        result['cuda_matmul_passed'] = bool((y == 128).all().item())
        result['visible_gpus'] = torch.cuda.device_count()
except Exception as error:
    result['cuda_error'] = str(error)
print(json.dumps(result, indent=2), flush=True)
