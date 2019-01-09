from pathlib import Path
from subprocess import check_output


# Absolute path to your base git repository
# REPO_PATH is a Path object, REPO_STR is a string object
REPO_PATH = Path(check_output('git rev-parse --show-toplevel', shell=True, universal_newlines=True).strip())
REPO_STR = str(REPO_PATH)