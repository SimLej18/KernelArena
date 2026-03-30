"""Package benchmark results into a results/ submission directory."""
import argparse
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

OUT = Path("out")
RESULTS = Path("results")

LIB_IMPORTS = {
    "kernax":   "kernax",
    "sklearn":  "sklearn",
    "gpytorch": "gpytorch",
    "gpjax":    "gpjax",
    "gpflow":   "gpflow",
    "gpy":      "GPy",
}


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def get_machine_info() -> dict:
    json_files = [p for p in OUT.glob("*.json") if p.stem != "plots"]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {OUT}/. Run `make all` first.")
    with open(json_files[0]) as f:
        data = json.load(f)
    return data.get("machine_info", {})


def get_lib_version(lib: str, import_name: str) -> str | None:
    python = Path(f"env/.venv-{lib}/bin/python")
    if not python.exists():
        return None
    try:
        result = subprocess.run(
            [str(python), "-c", f"import {import_name}; print({import_name}.__version__)"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Package benchmark results for submission.")
    parser.add_argument("--dtype",  default="float32")
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()

    machine_info = get_machine_info()
    cpu_brand = machine_info.get("cpu", {}).get("brand_raw", "unknown-cpu")

    slug = slugify(cpu_brand)
    gpu_suffix = "_gpu" if args.gpu else ""
    submission_id = f"{slug}_{args.dtype}_{date.today().isoformat()}{gpu_suffix}"

    dest = RESULTS / submission_id
    dest.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in sorted(OUT.glob("*.json")):
        shutil.copy2(src, dest / src.name)
        copied.append(src.name)

    libraries = {}
    for lib, import_name in LIB_IMPORTS.items():
        version = get_lib_version(lib, import_name)
        if version:
            libraries[lib] = version

    machine_info.pop("node", None)

    run_info = {
        "id": submission_id,
        "date": date.today().isoformat(),
        "dtype": args.dtype,
        "rounds": args.rounds,
        "gpu": bool(args.gpu),
        "machine": machine_info,
        "libraries": libraries,
    }
    with open(dest / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Submission packaged: results/{submission_id}/")
    print(f"  Benchmarks : {', '.join(copied)}")
    if libraries:
        print(f"  Libraries  : {', '.join(f'{k}=={v}' for k, v in libraries.items())}")


if __name__ == "__main__":
    main()
