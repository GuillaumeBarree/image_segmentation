from __future__ import annotations

import os
from pathlib import Path

import git

# Define PROJECT ROOT path
try:
    PROJECT_ROOT = Path(git.Repo(Path.cwd(), search_parent_directories=True).working_dir)
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

os.environ['PROJECT_ROOT'] = str(PROJECT_ROOT)

# Fix SEED for reproductibility
DATA_LOADER_SEED = 0
