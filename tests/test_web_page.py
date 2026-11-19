"""The data block of web/battery.html agrees with the results, and the page is built from web/src."""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def test_page_agrees_with_results():
    sys.path.insert(0, str(SCRIPTS))           # verify_web_page imports scripts/_path.py
    spec = importlib.util.spec_from_file_location("verify_web_page", SCRIPTS / "verify_web_page.py")
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    for key, (n, bad) in verify.run().items():
        assert bad == 0, f"{bad} of {n} {key} differ"
