import os
import shutil
from ..cimport import get_platform_dir, wget_tpnd_headers

def test_wget_tpnd_headers():
    wget_tpnd_headers()
    assert 1 == 1
