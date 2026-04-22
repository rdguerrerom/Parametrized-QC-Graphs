"""Regenerate all manuscript figures from the current results/*.json.

Run as:
    conda run -n pytorch-env python scripts/plot_all.py

Outputs figures/{fig1,fig2,fig3,fig4}_{name}.{pdf,png} at 300 dpi.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from importlib import import_module


def main() -> None:
    for mod in ("plot_fig1_framework",
                "plot_fig2_pareto",
                "plot_fig3_head_to_head",
                "plot_fig4_scaling"):
        m = import_module(mod)
        print(f"[plot_all] {mod} ...", flush=True)
        m.main()
    print(f"[plot_all] all figures written to {HERE.parent / 'figures'}")


if __name__ == "__main__":
    main()
