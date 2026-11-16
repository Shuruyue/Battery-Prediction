"""The JavaScript model and the data block of web/ups-battery.html agree with the package."""
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed to run the page's JavaScript")
def test_page_agrees_with_package():
    sys.path.insert(0, str(SCRIPTS))           # verify_web_page imports scripts/_path.py
    spec = importlib.util.spec_from_file_location("verify_web_page", SCRIPTS / "verify_web_page.py")
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    for key, (n, bad) in verify.run(n_cases=12).items():
        assert bad == 0, f"{bad} of {n} {key} differ"
