#!/usr/bin/env bash
# =============================================================================
# setup.sh - One-time environment setup for the BTC MLOps Assignment (macOS)
#
# Installs: Homebrew, Git, Python 3.11, Docker Desktop, Minikube, kubectl,
#           and a Python virtual environment with the libraries in requirements.txt.
# Then starts Minikube and enables the ingress + metrics-server addons.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Safe to run more than once - already-installed tools are skipped.
# Note: Airflow is NOT pip-installed here; it runs in Docker via docker-compose.
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[ok]${NC} $1"; }
step()  { echo -e "\n${YELLOW}==> $1${NC}"; }
fail()  { echo -e "${RED}[error]${NC} $1"; exit 1; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_VERSION="3.11"

# ----------------------------------------------------------------------------
# 0. Check OS
# ----------------------------------------------------------------------------
[[ "$(uname)" == "Darwin" ]] || fail "This script is for macOS only."
ARCH="$(uname -m)"
info "macOS detected ($ARCH)"

# ----------------------------------------------------------------------------
# 1. Homebrew
# ----------------------------------------------------------------------------
step "Checking Homebrew"
if ! command -v brew &>/dev/null; then
  echo "Installing Homebrew (you may be asked for your Mac password)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ "$ARCH" == "arm64" ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    grep -q 'brew shellenv' ~/.zprofile 2>/dev/null || \
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
  else
    eval "$(/usr/local/bin/brew shellenv)"
  fi
fi
info "Homebrew $(brew --version | head -1 | awk '{print $2}')"
brew update

# ----------------------------------------------------------------------------
# 2. CLI tools
# ----------------------------------------------------------------------------
step "Installing CLI tools (git, python, kubectl, minikube)"
FORMULAS=(git "python@${PYTHON_VERSION}" kubectl minikube)
for f in "${FORMULAS[@]}"; do
  if brew list --formula "$f" &>/dev/null; then
    info "$f already installed"
  else
    brew install "$f"
    info "$f installed"
  fi
done

# ----------------------------------------------------------------------------
# 3. Docker Desktop
# ----------------------------------------------------------------------------
step "Checking Docker Desktop"
if [[ -d "/Applications/Docker.app" ]]; then
  info "Docker Desktop already installed"
else
  brew install --cask docker
  info "Docker Desktop installed"
fi

if ! docker info &>/dev/null; then
  echo "Starting Docker Desktop (first launch may ask you to accept terms)..."
  open -a Docker
  echo -n "Waiting for Docker to start"
  for i in {1..60}; do
    if docker info &>/dev/null; then break; fi
    echo -n "."; sleep 3
  done
  echo
fi
docker info &>/dev/null || fail "Docker did not start. Open Docker Desktop manually, then re-run ./setup.sh"
info "Docker is running ($(docker --version))"

# ----------------------------------------------------------------------------
# 4. Python virtual environment + libraries
# ----------------------------------------------------------------------------
step "Creating Python virtual environment at .venv (libraries from requirements.txt)"
PYTHON_BIN="$(brew --prefix)/opt/python@${PYTHON_VERSION}/bin/python${PYTHON_VERSION}"
[[ -x "$PYTHON_BIN" ]] || fail "Python ${PYTHON_VERSION} not found at $PYTHON_BIN"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# XGBoost on macOS needs OpenMP
brew list --formula libomp &>/dev/null || brew install libomp

[[ -f "$PROJECT_DIR/requirements.txt" ]] || fail "requirements.txt not found next to setup.sh"
pip install -r "$PROJECT_DIR/requirements.txt"
info "Python libraries installed in .venv"

# ----------------------------------------------------------------------------
# 5. Minikube cluster + addons
# ----------------------------------------------------------------------------
step "Starting Minikube (Kubernetes) cluster"
if minikube status &>/dev/null; then
  info "Minikube already running"
else
  minikube start --driver=docker --cpus=4 --memory=6144
fi
minikube addons enable ingress
minikube addons enable metrics-server
info "Minikube running with ingress + metrics-server"

# ----------------------------------------------------------------------------
# 6. Summary
# ----------------------------------------------------------------------------
step "Installed versions"
echo "  Docker   : $(docker --version)"
echo "  Compose  : $(docker compose version)"
echo "  kubectl  : $(kubectl version --client 2>/dev/null | head -1)"
echo "  Minikube : $(minikube version --short)"
echo "  Python   : $(python --version)"
echo "  kfp      : $(pip show kfp 2>/dev/null | awk '/^Version/{print $2}')"
kubectl get nodes

cat <<EOF

$(echo -e "${GREEN}Setup complete!${NC}")

Next steps:
  1. Activate the Python environment in every new terminal:
       source .venv/bin/activate

  2. For the Ingress (btc-forecast.local) to work, run once:
       echo "127.0.0.1 btc-forecast.local" | sudo tee -a /etc/hosts
     and keep this running in a separate terminal while testing:
       minikube tunnel

  3. Log in to Docker Hub:
       docker login

  4. Stop Minikube when you're done (saves RAM):
       minikube stop
EOF
