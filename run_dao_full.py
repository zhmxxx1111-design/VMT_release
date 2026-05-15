"""
完整计算流程：依次调用 VMT_release 内各数值模块（虚体介质论第18章 + 附录L）。

各步 try/except 隔离，单模块失败时继续后续汇总。
"""
from __future__ import annotations

import sys
from pathlib import Path

MODULES = [
    "nonlinear_bvp",
    "bvp_coherence_q",
    "wall_fractal_2d",
    "dao_physics",
    "sturm_liouville",
]


def run_module(name: str) -> tuple[int, str]:
    try:
        __import__(name)
        mod = sys.modules[name]
        if hasattr(mod, "main"):
            mod.main()
        return 0, "ok"
    except Exception as exc:
        return 1, str(exc)


def main() -> None:
    here = Path(__file__).resolve().parent
    rows: list[tuple[str, int, str]] = []
    print("run_dao_full: sequential module run")
    print(f"  cwd = {here}")
    for name in MODULES:
        try:
            if str(here) not in sys.path:
                sys.path.insert(0, str(here))
            code, msg = run_module(name)
        except Exception as exc:  # pragma: no cover
            code, msg = 1, str(exc)
        rows.append((name, code, msg))
        print(f"  [{name}] exit={code}  {msg}")

    print()
    print("--- summary ---")
    print(f"{'module':<22} {'exit':>6}  note")
    for name, code, msg in rows:
        note = msg if code else "ok"
        print(f"{name:<22} {code:>6}  {note[:60]}")


if __name__ == "__main__":
    main()
