"""
Smoke-run the COMPILED Kubeflow pipeline (btc_kfp_pipeline.yaml) without a cluster.

For each step, in DAG order, this reads the executor's exact image, command and
args from the compiled spec, fills in the KFP placeholders (input parameters and
artifact paths) the way the Kubeflow launcher would, and runs it with `docker run`.
Artifacts are written to kubeflow/local_outputs/<step>/<artifact>, so they flow
from one container to the next just like on Kubeflow.

    python kubeflow/btc_kfp_pipeline.py                       # compile
    docker build -f kubeflow/Dockerfile -t ghcr.io/cwtausif/btc-pipeline:latest .
    python kubeflow/run_local.py
"""

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SPEC = yaml.safe_load((HERE / "btc_kfp_pipeline.yaml").read_text())
OUT = HERE / "local_outputs"
PARAMS = {
    k: v["defaultValue"]
    for k, v in SPEC["root"]["inputDefinitions"]["parameters"].items()
}
TASKS = SPEC["root"]["dag"]["tasks"]
EXECUTORS = SPEC["deploymentSpec"]["executors"]


def order(tasks):
    done, out = set(), []
    while len(out) < len(tasks):
        for name, t in tasks.items():
            deps = set(t.get("dependentTasks", []))
            if name not in done and deps <= done:
                done.add(name)
                out.append(name)
    return out


def run_task(name):
    task = TASKS[name]
    comp = SPEC["components"][task["componentRef"]["name"]]
    ex = EXECUTORS[comp["executorLabel"]]["container"]
    inputs = task.get("inputs", {})

    def resolve(arg):
        def param(m):
            key = m.group(1)
            src = inputs["parameters"][key]
            return str(PARAMS[src["componentInputParameter"]])

        def in_art(m):
            key = m.group(1)
            src = inputs["artifacts"][key]["taskOutputArtifact"]
            return f"/kfp/{src['producerTask']}/{src['outputArtifactKey']}"

        def out_art(m):
            return f"/kfp/{name}/{m.group(1)}"

        arg = re.sub(r"\{\{\$\.inputs\.parameters\['([^']+)'\]\}\}", param, arg)
        arg = re.sub(r"\{\{\$\.inputs\.artifacts\['([^']+)'\]\.path\}\}", in_art, arg)
        arg = re.sub(r"\{\{\$\.outputs\.artifacts\['([^']+)'\]\.path\}\}", out_art, arg)
        return arg

    args = [resolve(a) for a in ex.get("args", [])]
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name).chmod(0o777)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{OUT}:/kfp",
        ex["image"],
        *ex["command"],
        *args,
    ]
    print(f"\n> {name}  ({ex['image']})")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith("[") or "rror" in line:
            print("   ", line[:180])
    status = "SUCCESS" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"  {status} in {time.time() - t0:.1f}s")
    if proc.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    OUT.chmod(0o777)
    print(f"Pipeline: {SPEC['pipelineInfo']['name']}  params={PARAMS}")
    for t in order(TASKS):
        run_task(t)
    m = json.loads((OUT / "train-model" / "metrics").read_text())
    print("\nArtifacts:")
    for p in sorted(OUT.glob("*/*")):
        print(f"   {p.relative_to(HERE)}  ({p.stat().st_size:,} bytes)")
    print(
        f"\nModel metrics: MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} MAPE={m['mape']:.2f}%"
    )
