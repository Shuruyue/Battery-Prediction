"""Print the main results (the numbers README.md quotes), computed from results/*.json."""
import _path  # noqa: F401

from ups_battery_prediction import summary

if __name__ == "__main__":
    print("\n".join(summary.main_lines(_path.RESULTS)))
