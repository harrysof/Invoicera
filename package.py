"""
package.py

Packages gui.py into a single Windows .exe using PyInstaller, following
PaddleOCR's own documented packaging recipe:
https://www.paddleocr.ai/main/en/version3.x/inference_deployment/others/packaging.html

Usage:
    python package.py --file gui.py

This is their script, adapted only to default --file to gui.py so you don't
have to type it every time. Everything else matches their documented recipe
exactly -- including the fact that this is PyInstaller-only. Their docs
explicitly say Nuitka is NOT supported ("Nuitka's packaging principle is
incompatible with PaddleOCR"), so don't try to switch to it later expecting
a smaller/faster build.

Run this from the same environment where you installed requirements.txt --
it inspects your currently-installed packages to decide what metadata to
bundle, so it needs to see paddleocr/paddlex actually installed.
"""

import paddlex
import importlib.metadata
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='gui.py', help='Your file name, e.g. gui.py.')
parser.add_argument('--nvidia', action='store_true', help='Include NVIDIA CUDA and cuDNN dependencies.')

args = parser.parse_args()

main_file = args.file

user_deps = [dist.metadata["Name"] for dist in importlib.metadata.distributions()]
deps_all = list(paddlex.utils.deps.BASE_DEP_SPECS.keys())
deps_need = [dep for dep in user_deps if dep in deps_all]

cmd = [
    "pyinstaller", main_file,
    "--collect-data", "paddlex",
    "--collect-binaries", "paddle"
]

if args.nvidia:
    cmd += ["--collect-binaries", "nvidia"]

for dep in deps_need:
    cmd += ["--copy-metadata", dep]

print("PyInstaller command:", " ".join(cmd))

try:
    result = subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print("Packaging failed:", e)
    sys.exit(1)
