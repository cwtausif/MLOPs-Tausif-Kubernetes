"""
Build docs/MLOps_Assignment3_Report.pdf from the real evidence in docs/evidence and
docs/screenshots. Run:  python docs/build_report.py
"""

import base64
import csv
import html
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EV = DOCS / "evidence"
SHOTS = DOCS / "screenshots"
REPO = "https://github.com/cwtausif/MLOPs-Tausif-Kubernetes"
IMAGE = "ghcr.io/cwtausif/btc-forecast-api"


def ev(name):
    return (EV / name).read_text()


def pre(text, cls=""):
    return f'<pre class="term {cls}">{html.escape(text.strip())}</pre>'


def lines(text, *patterns, keep_cmd=True):
    """Keep lines matching any regex pattern (and '$ ' command lines)."""
    out = []
    for ln in text.splitlines():
        if (keep_cmd and ln.startswith("$ ")) or any(re.search(p, ln) for p in patterns):
            out.append(ln)
    return "\n".join(out)


def section(text, start, end=None):
    """Slice text from the line starting with `start` up to the next `end` line."""
    s = text.find(start)
    if s < 0:
        return ""
    e = text.find(end, s + len(start)) if end else -1
    return text[s : e if e > 0 else None].strip()


def img(name, cls=""):
    data = base64.b64encode((SHOTS / name).read_bytes()).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{data}">'


# ---------------------------------------------------------------- numbers
metrics = json.loads((ROOT / "models/metrics.json").read_text())
preds = json.loads((ROOT / "models/predictions.json").read_text())
hpa_rows = list(csv.DictReader(open(EV / "hpa_timeline.csv")))
max_rep = max(int(r["replicas"] or 0) for r in hpa_rows)
max_cpu = max(float(r["cpu_pct"]) for r in hpa_rows if r["cpu_pct"])
locust = ev("09_locust.txt")
agg = re.search(
    r"Aggregated\s+(\d+)\s+(\d+)\(([\d.]+)%\)\s+\|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\|\s+([\d.]+)",
    locust,
)
reqs, fails, fail_pct, avg_ms, _, max_ms, med_ms, rps = agg.groups()
pct = re.search(r"Aggregated\s+((?:\d+\s+){11})", locust.split("Response time percentiles")[1])
p = pct.group(1).split()
p95, p99 = p[5], p[7]
chaos = ev("10_chaos.txt")
heal = re.findall(r"--- t\+(\d+)s", chaos)[-1]
chaos_short = chaos.split("$ kubectl get events")[0]
chaos_short = re.sub(r"--- t\+(4|8|12)s.*?(?=--- t|\$ kubectl get deploy)", "", chaos_short, flags=re.S)
chaos_short += "\n" + section(chaos, "$ curl /health")

# ---------------------------------------------------------------- diagram
ARCH = """
<svg viewBox="0 0 760 250" class="arch" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica" font-size="10">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>
<g stroke="#475569" stroke-width="1.2" fill="none" marker-end="url(#a)">
<path d="M98,50 H128"/><path d="M402,50 H432"/><path d="M590,72 V112"/><path d="M98,190 H128"/>
<path d="M282,190 H312"/><path d="M468,190 H498"/><path d="M590,165 V150"/>
</g>
<rect x="8" y="28" width="90" height="44" rx="6" fill="#fff7ed" stroke="#ea580c"/>
<text x="53" y="47" text-anchor="middle" font-weight="bold">yfinance</text><text x="53" y="61" text-anchor="middle" fill="#6b7280">BTC-USD daily</text>
<rect x="130" y="8" width="272" height="86" rx="8" fill="#eff6ff" stroke="#2563eb"/>
<text x="140" y="24" font-weight="bold" fill="#1e40af">Apache Airflow DAG  @daily  (Kubeflow KFP mirror)</text>
<g font-size="9"><rect x="140" y="34" width="56" height="22" rx="4" fill="#fff" stroke="#93c5fd"/><text x="168" y="48" text-anchor="middle">fetch</text>
<rect x="203" y="34" width="62" height="22" rx="4" fill="#fff" stroke="#93c5fd"/><text x="234" y="48" text-anchor="middle">features</text>
<rect x="272" y="34" width="56" height="22" rx="4" fill="#fff" stroke="#93c5fd"/><text x="300" y="48" text-anchor="middle">train</text>
<rect x="335" y="34" width="58" height="22" rx="4" fill="#fff" stroke="#93c5fd"/><text x="364" y="48" text-anchor="middle">evaluate</text>
<text x="140" y="74" fill="#6b7280">XGBoost × 7 horizons · MAPE gate ≤ 10% · POST /reload</text>
<text x="140" y="87" fill="#6b7280">LocalExecutor + Postgres (docker-compose)</text></g>
<rect x="434" y="28" width="312" height="44" rx="6" fill="#f0fdf4" stroke="#16a34a"/>
<text x="590" y="47" text-anchor="middle" font-weight="bold">Model store — PVC btc-models-pvc (1Gi)</text>
<text x="590" y="61" text-anchor="middle" fill="#6b7280">model.pkl · metrics.json · predictions.json → /app/models</text>
<rect x="434" y="114" width="312" height="36" rx="6" fill="#fff" stroke="#16a34a" stroke-dasharray="3 2"/>
<text x="590" y="130" text-anchor="middle" font-weight="bold">Deployment btc-api (3 → HPA 2–8)</text>
<text x="590" y="143" text-anchor="middle" fill="#6b7280">Gunicorn · probes /health · 100m/128Mi → 500m/512Mi</text>
<rect x="8" y="168" width="90" height="44" rx="6" fill="#f8fafc" stroke="#334155"/>
<text x="53" y="187" text-anchor="middle" font-weight="bold">git push</text><text x="53" y="201" text-anchor="middle" fill="#6b7280">main / PR</text>
<rect x="130" y="160" width="152" height="60" rx="6" fill="#f8fafc" stroke="#334155"/>
<text x="206" y="178" text-anchor="middle" font-weight="bold">GitHub Actions</text>
<text x="206" y="193" text-anchor="middle" fill="#6b7280">flake8 · black · pytest</text><text x="206" y="206" text-anchor="middle" fill="#6b7280">KFP compile · buildx push</text>
<rect x="314" y="168" width="154" height="44" rx="6" fill="#f8fafc" stroke="#334155"/>
<text x="391" y="187" text-anchor="middle" font-weight="bold">GHCR / Docker Hub</text><text x="391" y="201" text-anchor="middle" fill="#6b7280">:sha  :latest</text>
<rect x="500" y="165" width="246" height="78" rx="8" fill="#faf5ff" stroke="#7c3aed"/>
<text x="623" y="182" text-anchor="middle" font-weight="bold" fill="#5b21b6">Minikube cluster</text>
<g font-size="9" fill="#374151"><text x="512" y="198">Ingress nginx  btc-forecast.local  →</text>
<text x="512" y="211">Service ClusterIP :80 → :5000 · NodePort 30080</text>
<text x="512" y="224">ConfigMap + Secret (envFrom) · HPA cpu 60%</text>
<text x="512" y="237">metrics-server · Locust Job (load)</text></g>
</svg>"""

# ---------------------------------------------------------------- evidence excerpts
lint = ev("01_lint_test.txt")
pytest_lines = lines(lint, r"PASSED|FAILED|passed|failed", keep_cmd=False)
pytest_short = "\n".join(
    [l for l in pytest_lines.splitlines() if "test_" in l][:25]
)
pytest_short = re.sub(r"\s+\[\s*\d+%\]", "", pytest_short)
pytest_two_col = "\n".join("✔ " + l.split("::")[1].replace(" PASSED", "") for l in pytest_short.splitlines())
build = ev("02_docker_build.txt")
build_ex = section(build, "#11 [runtime 4/8]", "$ docker history")
hist = "\n".join(section(build, "$ docker history", "$ docker run").splitlines()[:14])
nonroot = section(build, "$ docker run")
compose = ev("04_compose_api.txt")
compose_resp = lines(section(compose, "$ curl", "$ docker compose ps"), r"^\{", r"HTTP")
airflow = ev("03_airflow_run.txt")
af_states = section(airflow, "$ airflow tasks states-for-dag-run", "--- task log")
af_log = section(airflow, "--- task log highlights ---")
af_log = re.sub(r"\[notify\] ✅ (.{60}).*", r"[notify] ✅ \1…", af_log)
af_log = re.sub(r"(Saved candidate \S+)\. Next 7 days: .*", r"\1 (7-day forecast written)", af_log)
cicd = ev("08_cicd.txt")
k8s_apply = ev("05_k8s_apply.txt")
k8s_status = ev("06_k8s_status.txt")
e2e = ev("11_k8s_e2e.txt")
e2e = re.sub(r"\$ curl -s http://btc-forecast.local/metrics\n.*?\n\n", "", e2e, flags=re.S)
hpa_watch = ev("07_hpa_watch.txt")
hw = hpa_watch.splitlines()
hpa_tbl = "".join(
    f"<tr><td>{int(r['t'])//60}:{int(r['t'])%60:02d}</td><td>{r['cpu_pct']}%</td><td>{r['desired']}</td><td>{r['replicas']}</td><td>{r['ready']}</td></tr>"
    for i, r in enumerate(hpa_rows) if i in (0, len(hpa_rows) - 1) or any(r[k] != hpa_rows[i-1][k] for k in ('cpu_pct', 'desired', 'replicas', 'ready'))
)
loc_rows = re.findall(
    r"^(GET|POST)\s+(\S+(?: \(3 days\))?)\s+(\d+)\s+(\d+)\(([\d.]+)%\)\s+\|\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\|\s+([\d.]+)",
    locust.split("Response time percentiles")[0], re.M)
locust_tbl = "".join(
    f"<tr><td>{m[0]} {m[1]}</td><td>{int(m[2]):,}</td><td>{m[3]}</td><td>{m[5]}</td><td>{m[8]}</td><td>{m[7]}</td><td>{m[9]}</td></tr>"
    for m in loc_rows
) + f"<tr><th>Total</th><th>{int(reqs):,}</th><th>{fails}</th><th>{avg_ms}</th><th>{med_ms}</th><th>{max_ms}</th><th>{float(rps):.1f}</th></tr>"
kfp_local = ev("12_kfp_local.txt") if (EV / "12_kfp_local.txt").exists() else ""
kfp_local = re.sub(r"(Saved candidate \S+)\. Next 7 days: .*", r"\1 (7-day forecast written)", kfp_local)

forecast_rows = "".join(
    f"<tr><td>{x['date']}</td><td>${x['close']:,.2f}</td></tr>" for x in preds["predictions"]
)

CSS = """
@page { size: A4; }
* { box-sizing: border-box; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9.1pt; color: #1f2937; line-height: 1.38; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 2px; color: #0f172a; }
h2 { font-size: 11.5pt; margin: 12px 0 5px; padding-bottom: 3px; border-bottom: 1.5px solid #2563eb; color: #0f172a; break-after: avoid; }
h3 { font-size: 9.6pt; margin: 8px 0 3px; color: #1e40af; break-after: avoid; }
p { margin: 3px 0 5px; }
.sub { color: #475569; font-size: 9pt; }
.meta { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 18px; margin: 8px 0; padding: 7px 10px; background: #f1f5f9; border-radius: 6px; font-size: 8.6pt; }
.meta b { color: #334155; }
a { color: #1d4ed8; text-decoration: none; }
pre.term { background: #0f172a; color: #e2e8f0; font-family: Menlo, Consolas, monospace; font-size: 6.55pt; line-height: 1.33;
  padding: 6px 8px; border-radius: 5px; white-space: pre-wrap; word-break: break-all; margin: 3px 0 6px; break-inside: avoid; }
pre.small { font-size: 6.1pt; }
pre.cols { column-count: 2; column-gap: 14px; }
pre.brk { break-inside: auto; }
table.tight td, table.tight th { padding: 1.5px 4px; font-size: 7.4pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 3px 0 7px; break-inside: avoid; }
th, td { border: 1px solid #e2e8f0; padding: 2.5px 5px; text-align: left; vertical-align: top; }
th { background: #f1f5f9; color: #334155; }
td.ok { color: #15803d; font-weight: bold; white-space: nowrap; }
code { font-family: Menlo, monospace; font-size: 7.8pt; background: #f1f5f9; padding: 0 3px; border-radius: 3px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.two > div { min-width: 0; }
img { max-width: 100%; display: block; border: 1px solid #e2e8f0; border-radius: 4px; margin: 3px 0 2px; }
.cap { font-size: 7.6pt; color: #64748b; margin: 0 0 7px; }
.arch { width: 100%; margin: 2px 0; }
.kpi { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 5px 0 8px; }
.kpi div { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 5px 7px; }
.kpi b { display: block; font-size: 12.5pt; color: #0f172a; }
.kpi span { font-size: 7.5pt; color: #64748b; }
.note { background: #fffbeb; border-left: 3px solid #f59e0b; padding: 4px 8px; font-size: 8.4pt; margin: 5px 0; }
.avoid { break-inside: avoid; }
.pb { break-before: page; }
"""

BODY = f"""
<h1>Production MLOps Pipeline — BTC-USD 7-Day Forecast</h1>
<div class="sub">Assignment 3 · Microservices, ETL, Orchestration &amp; Kubernetes Deployment · Technical Report · 24 November 2026</div>
<div class="meta">
<div><b>Group:</b> Tausif (cwtausif) · __________________</div>
<div><b>GitHub:</b> <a href="{REPO}">github.com/cwtausif/MLOPs-Tausif-Kubernetes</a></div>
<div><b>Evidence:</b> logs = excerpts of <code>docs/evidence/*.txt</code> (secrets masked)</div>
<div><b>Registry:</b> <a href="https://{IMAGE}">{IMAGE}</a> (public)</div>
</div>

<div class="kpi">
<div><b>{metrics['mape']:.2f}%</b><span>MAPE on last 7 days (naive baseline {metrics['baseline_mape']:.2f}%)</span></div>
<div><b>25 / 25</b><span>pytest unit tests passing (local + CI)</span></div>
<div><b>2 → {max_rep} pods</b><span>HPA scale-out at {max_cpu:.0f}% CPU of request</span></div>
<div><b>{heal} s</b><span>self-healing: deleted pod replaced &amp; Ready</span></div>
</div>

<h2>1. System architecture</h2>
{ARCH}
<p>Every day Airflow pulls BTC-USD OHLCV data from Yahoo Finance (2017 → today, {metrics['train_rows']:,} usable rows),
builds 14 scale-free features (1-day return, 7/30-day moving-average ratios, 7/30-day volatility, 7 lagged close ratios,
high–low range, volume change) and trains a <b>direct multi-horizon XGBoost model</b>: seven regressors, one per horizon
<i>h</i> = 1…7, each predicting the log-return from today's close to day <i>t+h</i>. Predicting returns instead of raw prices
lets trees forecast prices they have never seen. The model is back-tested on the most recent 7 days (train on everything
before, forecast, compare) and is only promoted to <code>model.pkl</code> if MAPE ≤ 10%; Flask is then told to reload.
The same steps exist as a Kubeflow pipeline. Code is linted, tested and shipped by GitHub Actions to GHCR and deployed on Minikube.</p>

<table>
<tr><th>Part</th><th>Requirement</th><th>Implementation (file)</th><th>Status</th></tr>
<tr><td>A</td><td>/health, /predict, /metrics, validation, Gunicorn, pytest</td><td><code>app/app.py</code>, <code>tests/</code> (+ <code>/reload</code>)</td><td class="ok">✔ done</td></tr>
<tr><td>B</td><td>Multi-stage non-root Dockerfile, compose, CI lint/test, CD push SHA+latest</td><td><code>app/Dockerfile</code>, <code>docker-compose.yml</code>, <code>.github/workflows/ci_cd.yml</code></td><td class="ok">✔ done</td></tr>
<tr><td>C</td><td>Daily Airflow DAG (fetch, features, retrain+gate, reload); KFP pipeline → YAML</td><td><code>dags/btc_etl_pipeline.py</code>, <code>kubeflow/btc_kfp_pipeline.py/.yaml</code></td><td class="ok">✔ done</td></tr>
<tr><td>D</td><td>ConfigMap/Secret, PVC, 3-replica Deployment + probes + limits, Services, HPA 2–8 @60%, Ingress</td><td><code>k8s/*.yaml</code>, <code>loadtest/</code></td><td class="ok">✔ done</td></tr>
<tr><td>E</td><td>Chaos test, end-to-end verification, report</td><td>§E below, <code>docs/evidence/</code></td><td class="ok">✔ done</td></tr>
</table>

<h2>Part A — Flask REST API microservice</h2>
<table>
<tr><th>Endpoint</th><th>Behaviour</th></tr>
<tr><td><code>GET /health</code></td><td><code>status: ok</code>, uptime, loaded model version/stage, training date. Used by K8s liveness &amp; readiness probes.</td></tr>
<tr><td><code>POST /predict</code></td><td>Optional <code>forecast_days</code> (int 1–7, default 7) and optional <code>history</code> (≥ 40 OHLCV records → features recomputed on the fly). Returns dated predictions.</td></tr>
<tr><td><code>GET /metrics</code></td><td>MAE, RMSE, MAPE, naive-baseline MAPE and test window of the latest validation run.</td></tr>
<tr><td><code>POST /reload</code></td><td>Re-reads <code>model.pkl</code>; called by Airflow. Pods also hot-reload automatically when the file's mtime changes.</td></tr>
</table>
<p><b>Validation &amp; errors.</b> Every error is JSON: 400 for non-JSON bodies, invalid JSON, non-object payloads, wrong-type or
out-of-range <code>forecast_days</code> (incl. booleans/floats), short history, missing fields, negative prices or bad dates;
404/405 for wrong routes/methods; 503 when no model is loaded; a catch-all handler returns 500 without leaking a stack trace.
Model loading is thread-safe (lock) and promotion uses atomic file replace, so a request never sees a half-written model.
<b>Gunicorn</b> serves the app in production: <code>gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 60 app:app</code>.</p>
<h3>pytest — 25 passed (21 API + 4 ETL/training); live request/response samples are in §E2</h3>{pre(pytest_two_col, 'small cols')}

<h2>Part B — Containerisation &amp; CI/CD</h2>
<p><b>B1 Dockerfile.</b> Stage 1 (<code>python:3.12-slim</code>) installs requirements into <code>/install</code>; stage 2 copies only those
packages plus <code>app.py</code> and the shared feature code — no pip cache or build tools. Requirements are copied before code so the
dependency layer stays cached. Runs as non-root <code>app</code> (uid 1001), <code>OMP_NUM_THREADS=1</code>, Docker <code>HEALTHCHECK</code>,
<code>.dockerignore</code> keeps data/models/tests out of the context. Linux uses the CPU-only <code>xgboost-cpu</code> wheel (no CUDA libs).</p>
{pre(build_ex + chr(10) + chr(10) + nonroot, 'small cols')}
<p class="cap">Final image ≈ 529 MB; 383 MB of it is the scientific stack (pandas, NumPy, scikit-learn, XGBoost). Application layers are ~15 kB.</p>
<p><b>B2 docker-compose.</b> One command starts <code>flask-api</code>, <code>postgres:16</code>, <code>airflow-init</code>, <code>airflow-apiserver</code> (UI :8080),
<code>airflow-scheduler</code> and <code>airflow-dag-processor</code> (Airflow 3.3, LocalExecutor). Airflow and Flask share <code>./models</code>, and Airflow reaches
Flask at <code>http://flask-api:5000/reload</code> on the compose network. All five services reported <i>healthy</i> in <code>docker compose ps</code> (<code>docs/evidence/04_compose_api.txt</code>).</p>
<p><b>B3/B4 GitHub Actions.</b> <code>ci_cd.yml</code> runs on every push to <code>main</code> and every pull request. Job 1: flake8, black --check,
pytest (JUnit report uploaded), KFP compile. Job 2 (pushes only, needs job 1): Buildx builds <code>app/Dockerfile</code> and tags it with the
short commit SHA and <code>latest</code>; it pushes to GHCR with the built-in <code>GITHUB_TOKEN</code> and to Docker Hub when the
<code>DOCKER_USERNAME</code>/<code>DOCKER_PASSWORD</code> secrets are configured (that step is skipped otherwise). It also publishes the Kubeflow component image.</p>
<div class="two">
<div>{img('github_actions_run.png')}<p class="cap">Run #1 — both jobs green (3 min 24 s).</p></div>
<div>{pre(cicd, 'small')}</div>
</div>

<h2>Part C — Airflow ETL pipeline &amp; Kubeflow</h2>
<p><b>C1 DAG</b> <code>btc_etl_pipeline</code> (TaskFlow API, <code>@daily</code>, <code>catchup=False</code>, 2 retries): <b>fetch_data</b> → yfinance →
<code>data/raw.json</code>; <b>preprocess</b> → cleaning (dedupe, drop non-positive prices, forward-fill missing days) + indicators →
<code>clean.json</code>; <b>train_model</b> → back-test on the last 7 days + refit on all data → candidate model, metrics, forecast;
<b>evaluate_model</b> → promote only if MAPE ≤ <code>MAPE_THRESHOLD</code> (10%); <b>reload_api</b> → <code>POST /reload</code> to Flask.
A manual run was triggered with <code>airflow dags trigger btc_etl_pipeline</code>:</p>
{img('airflow_graph.png')}
<p class="cap">Airflow 3 graph view of manual run <code>manual__2026-11-24T11:25:47</code> — all five tasks succeeded (3 min 45 s).</p>
{pre(af_log, 'small')}
<p><b>C2 Kubeflow.</b> <code>kubeflow/btc_kfp_pipeline.py</code> (KFP SDK 2.18) defines four <i>container components</i> —
<code>ingest_data</code>, <code>generate_features</code>, <code>train_model</code>, <code>evaluate_and_register</code> — each running the
<code>btc-pipeline</code> image (<code>kubeflow/Dockerfile</code>). Kubeflow passes typed artifacts between them (<code>Dataset</code> → <code>Dataset</code> →
<code>Model</code> + metrics → registered <code>Model</code>), so lineage is tracked; caching is off so each run fetches today's data, and the gate step
fails the run if the model is rejected. The pipeline compiles to <code>kubeflow/btc_kfp_pipeline.yaml</code> (IR schema 2.1.0) in CI, and
<code>--schedule</code> uploads it as a daily recurring run on a KFP cluster.{(' To prove the compiled spec really runs, <code>kubeflow/run_local.py</code> reads each executor’s image, command and args from the YAML, fills the KFP placeholders and runs it with <code>docker run</code>, wiring artifacts between steps:') if kfp_local else ''}</p>
{pre(kfp_local, 'small brk') if kfp_local else ''}

<h2>Part D — Kubernetes deployment &amp; scaling (Minikube v1.37, Kubernetes 1.37)</h2>
<table>
<tr><th>Manifest</th><th>Key settings</th></tr>
<tr><td><code>configmap-secret.yaml</code></td><td>ConfigMap: MODEL_STAGE, LOG_LEVEL, PORT, MODEL_DIR, GUNICORN_WORKERS, OMP_NUM_THREADS · Secret (Opaque): API_KEY, COINGECKO_API_KEY · both injected with <code>envFrom</code></td></tr>
<tr><td><code>pvc.yaml</code></td><td>1 Gi RWO claim (standard StorageClass) mounted read-only at <code>/app/models</code>; <code>model-loader</code> pod + <code>load_model.sh</code> copy artifacts in atomically</td></tr>
<tr><td><code>deployment.yaml</code></td><td>3 replicas (rolled out with <code>kubectl rollout status</code>), RollingUpdate (maxUnavailable 1, maxSurge 1), liveness GET /health (15 s delay, 10 s period), readiness GET /health (5 s), requests 100m/128Mi, limits 500m/512Mi, runAsNonRoot, drop ALL capabilities</td></tr>
<tr><td><code>service.yaml</code></td><td><code>btc-api-service</code> ClusterIP 80→5000 · <code>btc-api-nodeport</code> NodePort 80→5000 (30080)</td></tr>
<tr><td><code>hpa.yaml</code></td><td>autoscaling/v2, min 2, max 8, CPU averageUtilization 60%, 60 s scale-down window</td></tr>
<tr><td><code>ingress.yaml</code></td><td>ingressClassName nginx, host <code>btc-forecast.local</code>, path / → btc-api-service:80</td></tr>
</table>
{pre(k8s_status, 'small brk')}

<h3>D5 — Autoscaling under load</h3>
<p>Load was generated by Locust running <i>inside</i> the cluster as a Job (<code>loadtest/locust-job.yaml</code>) against the ClusterIP
Service, so traffic is balanced across all pods (a <code>kubectl port-forward</code> would pin it to one pod). 150 users, spawn 15/s, 4 minutes,
mix of <code>/predict</code> (7 and 3 days), <code>/health</code> and <code>/metrics</code>.</p>
{img('hpa_chart.png')}
<div class="two">
<div><table class="tight"><tr><th>t (m:s)</th><th>CPU / 60%</th><th>desired</th><th>replicas</th><th>ready</th></tr>{hpa_tbl}</table>
<p class="cap">From <code>kubectl get hpa</code> / <code>get deploy</code>, sampled every 10 s (<code>docs/evidence/07_hpa_watch.txt</code>).</p></div>
<div><table class="tight"><tr><th>Locust endpoint</th><th>requests</th><th>fails</th><th>avg ms</th><th>median</th><th>max</th><th>req/s</th></tr>{locust_tbl}</table>
<p class="cap">{int(reqs):,} requests · {float(rps):.1f} req/s · median {med_ms} ms · p95 {p95} ms · p99 {p99} ms · <b>{fail_pct}% errors</b>.</p>
<p>The HPA reacts ~60 s after load starts because metrics-server reports a 60 s CPU average and the HPA syncs every 15 s; it then
jumped 2 → 6 → 8 (the maximum) within 30 s. Every new pod passed its readiness probe, so “ready” tracks “replicas”.
Scale-down waited for the 60 s stabilisation window, then stepped 8 → 4 → 2.</p></div>
</div>

<h2>Part E — Verification &amp; resilience</h2>
<h3>E1 — Chaos test: deleting a running pod</h3>
{pre(chaos_short, 'small brk')}
<p>The ReplicaSet controller noticed the missing replica immediately and created <code>…ljqlw</code> while the old pod was still terminating.
The new pod was scheduled, started, and became <b>Ready after {heal} s</b> (gated by the readiness probe: imports + model load). The Service only
routes to Ready endpoints, so the surviving pod kept answering through the Ingress the whole time — no request had to fail.</p>

<h3>E2 — End-to-end verification</h3>
<p>(1) Airflow run fetched today's data, retrained, passed the gate and promoted <code>{metrics['version']}</code>, then Flask confirmed the reload
(Part C). (2) GitHub Actions run #1 passed lint, tests and pushed the image (Part B). (3) The promoted model was copied into the PVC and the live
cluster was queried through the Ingress host <code>btc-forecast.local</code> and the NodePort:</p>
{pre(e2e, 'small brk')}
<div class="two">
<div>{img('forecast_chart.png')}</div>
<div><table><tr><th>Date</th><th>Predicted close (USD)</th></tr>{forecast_rows}</table>
<p class="cap">Forecast from the live K8s endpoint (last close {preds['last_date']}: ${preds['last_close']:,.2f}).</p></div>
</div>

<h3>Performance analysis &amp; issues found during deployment</h3>
<p><b>Model.</b> Back-test on {metrics['test_start']} → {metrics['test_end']}: MAE ${metrics['mae']:,.0f}, RMSE ${metrics['rmse']:,.0f},
MAPE {metrics['mape']:.2f}% vs {metrics['baseline_mape']:.2f}% for the naive “price stays flat” baseline. That week BTC fell sharply;
the model followed the level but under-predicted the drop — expected for a daily-frequency model with no news/sentiment inputs, and why the
promotion gate and daily retraining matter. One 7-day window is a noisy estimate; a rolling multi-window back-test would be the next improvement.</p>
<p><b>Serving &amp; scaling.</b> Inference is cheap (7 tree ensembles on one feature row): the median latency was {med_ms} ms with
{int(reqs):,} requests and <b>{fail_pct}% errors</b>. Under 150 users the 2 baseline pods hit {max_cpu:.0f}% of their CPU request; within one HPA
sync the controller asked for 6 and then 8 pods (max), all of which became Ready and absorbed the load, bringing CPU down to ~170%
(still above target, so the HPA correctly stayed at its ceiling). After the test it scaled down in steps (8 → 4 → 2) once the 60 s
stabilisation window passed. The p95/p99 tail ({p95}/{p99} ms) comes from the first ~60 s, before the new pods were Ready, while
the 2 original pods were throttled at their 500m limit. An earlier run done while the Airflow stack shared the same Docker VM
saturated the node (0.62% errors, see below) — a reminder that HPA can only add pods if the node has spare capacity; in production a
Cluster Autoscaler would add nodes.</p>
<table>
<tr><th>Problem observed</th><th>Root cause</th><th>Fix</th></tr>
<tr><td>Pods failed liveness probes and sat at 440m CPU while idle after the model loaded</td><td>XGBoost/OpenMP starts one thread per <i>host</i> CPU (10) inside a 0.5-CPU cgroup → heavy throttling</td><td><code>OMP_NUM_THREADS=1</code> (image + ConfigMap) and <code>n_jobs=1</code> on load → idle CPU ≈ 15m</td></tr>
<tr><td>First <code>evaluate_model</code> attempt raised FileNotFoundError on rename</td><td>Fixed temp file name on a macOS bind mount</td><td>Per-process temp name + fsync before <code>os.replace</code>; Airflow retry also recovered it</td></tr>
<tr><td>HPA showed <code>&lt;unknown&gt;</code> and pods flapped during the first load test</td><td>Minikube shared Docker Desktop’s VM with the Airflow stack: swap full, load avg 64, metrics-server starved</td><td>Stopped compose during the K8s test and restarted metrics-server; reran the test (results above)</td></tr>
<tr><td><code>docker pull</code> of the CI image failed on Apple Silicon</td><td>CI built linux/amd64 only</td><td>Added QEMU + <code>platforms: linux/amd64,linux/arm64</code> to the workflow</td></tr>
<tr><td>Gunicorn “Permission denied: /home/app”</td><td>Non-root user created without a home dir (gunicorn control socket)</td><td><code>useradd --create-home</code></td></tr>
</table>
"""

if __name__ == "__main__":
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{BODY}</body></html>"
    (DOCS / "report.html").write_text(html_doc)
    pdf = DOCS / "MLOps_Assignment3_Report.pdf"
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page()
        pg.set_content(html_doc, wait_until="load")
        pg.pdf(
            path=str(pdf),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template="<div style='font-size:7px;color:#94a3b8;width:100%;text-align:center'>BTC-USD MLOps · Assignment 3 · page <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
            margin={"top": "12mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
        )
        b.close()
    print("wrote", pdf)
