from __future__ import annotations
import argparse, subprocess, sys, logging
from pathlib import Path
from backtest import run
from config_io import load_config

log = logging.getLogger("run_backtests")

def ensure_curve(project_root: Path) -> bool:
    out_dir = project_root / "data" / "output"
    candidates = [out_dir / "curve.parquet", out_dir / "curve.csv"]
    return any(p.exists() for p in candidates)

def main():
    parser = argparse.ArgumentParser(description="Run FX options backtests (pricing grid per model).")
    parser.add_argument("--recompute", action="store_true", help="Recompute all pricings (ignore existing results).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    cfg_path = project_root / "config" / "config.toml"
    cfg = load_config(cfg_path)
    models = cfg["models"]["use"]
    outdir = Path(cfg["reporting"]["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure curve files exist in data/output; if not, try to run a curve.py
    if not ensure_curve(project_root):
        # Prefer project-local src/curve.py; else fallback to /mnt/data/curve.py if present
        candidates = [project_root / "src" / "curve.py", Path("/mnt/data/curve.py")]
        curve_script = next((p for p in candidates if p.exists()), None)
        if curve_script is not None:
            log.info("No curve file in data/output/. Running %s ...", curve_script)
            subprocess.run([sys.executable, str(curve_script)], check=True)
        else:
            raise FileNotFoundError("Missing curve outputs (data/output/curve.parquet or curve.csv) and no curve.py found.")

    for m in models:
        target_csv = outdir / f"priced_grid_{m}.csv"
        if target_csv.exists() and not args.recompute:
            log.info("[%s] Reusing existing %s (use --recompute to rebuild).", m, target_csv.name)
            continue
        log.info("[%s] Pricing grid start ...", m)
        df = run(str(cfg_path), model_scope=[m], outdir=str(outdir))
        df.to_csv(target_csv, index=False)
        log.info("[%s] Done. Rows=%d -> %s", m, len(df), target_csv.name)

if __name__ == "__main__":
    main()
