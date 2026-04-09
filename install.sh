#!/usr/bin/env bash
#
# install.sh — automated build pipeline for DGX_Spark Qwen3.5-122B v2 (Steps 0-4).
#
# Walks through the Quick Start of README.md from a fresh clone:
#   0. Download Intel/Qwen3.5-122B-A10B-int4-AutoRound (~75 GB if not cached)
#   1. Build hybrid INT4+FP8 checkpoint               (~20 min, +9% perf, optional)
#   2. Add MTP speculative decoding weights
#   3. Build base vLLM image for SM121                (~30-60 min, runs Docker)
#   4. Build vllm-qwen35-v2 final image
#
# Out of scope: TurboQuant variant (run patches/04-turboquant/* manually if needed),
# Step 5 (launch) and Step 6 (benchmark) — those are runtime, see README.md.
#
# Idempotent: re-running skips steps whose outputs already exist.
# Run from anywhere: `./install.sh` from this repo, or `bash /path/to/install.sh`.
#
# Flags:
#   --no-cache       Force a clean rebuild: removes existing vllm-sm121 and
#                    vllm-qwen35-v2 images, prunes BuildKit cache, then runs
#                    Steps 3 & 4 from scratch. Use if a previous failed build
#                    left stale BuildKit layers. Adds 30-60 min.
#   --launch         After build, automatically launch the container (Step 5).
#                    Default: prompts interactively. With --launch, no prompt.
#   --no-launch      Never launch, never prompt. Useful for CI / unattended runs.
#   -h | --help      Print this help and exit.
#
# Sudo: this script never invokes sudo. If a prerequisite is missing (apt
# package, docker group membership, etc.), it prints the exact sudo command
# you need to run and then exits non-zero so you can fix it and re-run.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARK_VLLM_DIR="${PROJECT_DIR}/spark-vllm-docker"
HYBRID_DIR="${HOME}/models/qwen35-122b-hybrid-int4fp8"
SPARK_VLLM_PIN="49d6d9fefd7cd05e63af8b28e4b514e9d30d249f"

# ── Flags ─────────────────────────────────────────────────────────────────────
NO_CACHE=0
LAUNCH_MODE="prompt"   # prompt | yes | no
for arg in "$@"; do
    case "$arg" in
        --no-cache)  NO_CACHE=1 ;;
        --launch)    LAUNCH_MODE="yes" ;;
        --no-launch) LAUNCH_MODE="no" ;;
        -h|--help)
            sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown flag: $arg (use --help)" >&2; exit 2 ;;
    esac
done

# ── Pretty output ─────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[1;33m'
    C_BLU=$'\033[0;34m'; C_CYN=$'\033[0;36m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
    C_RED=""; C_GRN=""; C_YEL=""; C_BLU=""; C_CYN=""; C_DIM=""; C_OFF=""
fi

START_TS=$(date +%s)
STEP_TS=$START_TS
STEP_NUM=0
TOTAL_STEPS=7   # prereq + venv + Step 0 + Step 1 + Step 2 + Step 3 + Step 4

fmt_time() {
    local s=$1
    if [ "$s" -ge 3600 ]; then printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
    elif [ "$s" -ge 60 ]; then printf '%dm%02ds' $((s/60)) $((s%60))
    else printf '%ds' "$s"; fi
}

log()  { echo "${C_BLU}[install]${C_OFF} $*"; }
note() { echo "${C_DIM}          $*${C_OFF}"; }
ok()   { echo "${C_GRN}[ ok ]${C_OFF}    $*"; }
warn() { echo "${C_YEL}[warn]${C_OFF}    $*"; }
err()  { echo "${C_RED}[err ]${C_OFF}    $*" >&2; }

step_begin() {
    STEP_NUM=$((STEP_NUM + 1))
    STEP_TS=$(date +%s)
    echo
    log "${C_CYN}▶ [${STEP_NUM}/${TOTAL_STEPS}] $1${C_OFF}"
    [ -n "${2:-}" ] && note "$2"
}

step_end() {
    local now elapsed total
    now=$(date +%s)
    elapsed=$((now - STEP_TS))
    total=$((now - START_TS))
    ok "step done in $(fmt_time $elapsed)  ${C_DIM}(total $(fmt_time $total))${C_OFF}"
}

step_skip() {
    local now total
    now=$(date +%s)
    total=$((now - START_TS))
    ok "step skipped: $1  ${C_DIM}(total $(fmt_time $total))${C_OFF}"
}

abort() { err "$1"; exit 1; }

# ── Prerequisites ─────────────────────────────────────────────────────────────
step_begin "Checking prerequisites" "scans for everything needed; if anything is missing, prints exact install commands and exits"

# Collect missing items here. Each entry is a tab-separated:
#   "what is missing" \t "exact command to fix it"
missing=()
have_check() {
    local label="$1" present="$2" fix="$3"
    if [ "$present" = "1" ]; then
        echo "  ${C_GRN}✓${C_OFF} ${label}"
    else
        echo "  ${C_RED}✗${C_OFF} ${label}   ${C_DIM}— missing${C_OFF}"
        missing+=("${label}"$'\t'"${fix}")
    fi
}

# 1. python3
present=0; command -v python3 >/dev/null 2>&1 && present=1
have_check "python3" "$present" "sudo apt update && sudo apt install -y python3"

# 2. python3-venv (only checkable if python3 exists)
present=0
if command -v python3 >/dev/null 2>&1 && python3 -c 'import venv, ensurepip' 2>/dev/null; then
    present=1
fi
have_check "python3-venv + ensurepip" "$present" "sudo apt install -y python3-venv python3-pip"

# 3. git
present=0; command -v git >/dev/null 2>&1 && present=1
have_check "git" "$present" "sudo apt install -y git"

# 4. curl (used by upstream Dockerfile internally; harmless to require)
present=0; command -v curl >/dev/null 2>&1 && present=1
have_check "curl" "$present" "sudo apt install -y curl"

# 5. docker (binary)
present=0; command -v docker >/dev/null 2>&1 && present=1
have_check "docker (binary on PATH)" "$present" \
    "Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/"

# 6. docker daemon reachable WITHOUT sudo (i.e. user is in 'docker' group)
present=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    present=1
fi
have_check "docker daemon reachable as '$USER' (no sudo)" "$present" \
    "sudo usermod -aG docker $USER && newgrp docker  # then open a NEW terminal"

# 7. nvidia container runtime (so 'docker run --gpus all' works)
# Soft check: only meaningful if docker reachable. We probe by inspecting
# the runtimes list. Missing here is non-fatal for build, only for run.
nvidia_runtime_present=0
if [ "$present" = "1" ]; then
    if docker info 2>/dev/null | grep -qi 'nvidia'; then
        nvidia_runtime_present=1
    fi
fi
if [ "$nvidia_runtime_present" = "1" ]; then
    echo "  ${C_GRN}✓${C_OFF} nvidia container runtime (needed for --gpus all at run time)"
else
    echo "  ${C_YEL}~${C_OFF} nvidia container runtime not detected   ${C_DIM}— OK for build, required for launch${C_OFF}"
    note "if launch fails with 'unknown flag: --gpus' install nvidia-container-toolkit:"
    note "  sudo apt install -y nvidia-container-toolkit && sudo systemctl restart docker"
fi

# 8. Project sanity (no fix command — wrong dir, user must cd)
present=0
if [ -f "${PROJECT_DIR}/patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py" ] \
   && [ -f "${PROJECT_DIR}/docker/Dockerfile.v2" ]; then
    present=1
fi
have_check "project files at ${PROJECT_DIR}/{patches,docker}" "$present" \
    "Run install.sh from inside the cloned DGX_Spark_Qwen3.5-122B-A10B-AR-INT4 repo"

# 9. Disk space (need ~170 GB free in $HOME)
need_gb=170
free_gb=$(df -BG "${HOME}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
free_gb=${free_gb:-0}
if [ "$free_gb" -ge "$need_gb" ]; then
    echo "  ${C_GRN}✓${C_OFF} free disk in \$HOME: ${free_gb} GB (need ~${need_gb})"
else
    echo "  ${C_YEL}~${C_OFF} free disk in \$HOME: ${free_gb} GB   ${C_DIM}— recommended ${need_gb} GB, will try anyway${C_OFF}"
fi

# Verdict
if [ "${#missing[@]}" -gt 0 ]; then
    echo
    err "${#missing[@]} prerequisite(s) missing. Please install them and re-run ./install.sh:"
    echo
    n=1
    for item in "${missing[@]}"; do
        what="${item%%$'\t'*}"
        fix="${item#*$'\t'}"
        echo "  ${C_YEL}${n}.${C_OFF} ${what}"
        echo "     ${C_CYN}${fix}${C_OFF}"
        n=$((n + 1))
    done
    echo
    err "All commands above need sudo. Run them, open a fresh terminal if you added"
    err "yourself to the docker group, then re-run ./install.sh."
    exit 1
fi

note "all prerequisites OK"
step_end

# ── venv + host-side deps ─────────────────────────────────────────────────────
step_begin "Setting up Python venv and host-side dependencies" \
           "python3 -m venv .venv && pip install torch numpy safetensors huggingface_hub"

cd "${PROJECT_DIR}"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q torch numpy safetensors huggingface_hub
note "venv: $(python3 -c 'import sys;print(sys.prefix)')"
note "hf:   $(hf --version 2>/dev/null || echo 'not present')"
step_end

# ── Step 0: hf download ───────────────────────────────────────────────────────
step_begin "Step 0 — Downloading Intel/Qwen3.5-122B-A10B-int4-AutoRound" \
           "first time: ~75 GB with progress bars; cached: instant"

# Two-pass approach:
#   Pass 1 — verbose 'hf download' so the user sees progress bars on a
#            first-time 75 GB download (no tqdm = looks frozen for 10+ min).
#   Pass 2 — 'hf download --quiet' is a no-op against the now-populated
#            cache, but unlike pass 1 it prints *only* the snapshot
#            directory path on stdout, which is exactly what we need to
#            capture as INTEL_DIR. This replaces the previous
#            'find | head -1' dance, which was non-deterministic when
#            multiple snapshot directories coexisted in cache (e.g. the
#            user ran 'hf download' at different times and Intel shipped
#            a new revision in between).
hf download Intel/Qwen3.5-122B-A10B-int4-AutoRound
INTEL_DIR=$(hf download Intel/Qwen3.5-122B-A10B-int4-AutoRound --quiet)
[ -d "$INTEL_DIR" ] || abort "INTEL_DIR not found after hf download: '${INTEL_DIR}' is not a directory. Check your HF cache config (HF_HOME, HF_HUB_CACHE)."
note "INTEL_DIR=${INTEL_DIR}"
step_end

# ── Step 1: hybrid checkpoint ─────────────────────────────────────────────────
MODEL_DIR="${HYBRID_DIR}"
if [ -f "${HYBRID_DIR}/model.safetensors.index.json" ] \
    && [ -f "${HYBRID_DIR}/model-00014-of-00014.safetensors" ]; then
    STEP_NUM=$((STEP_NUM + 1))
    step_skip "Step 1 — hybrid checkpoint already exists at ${HYBRID_DIR}"
else
    step_begin "Step 1 — Building hybrid INT4+FP8 checkpoint" \
               "~20 min, output ~71 GB at ${HYBRID_DIR}"
    python "${PROJECT_DIR}/patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py" \
        --gptq-dir "${INTEL_DIR}" \
        --fp8-repo Qwen/Qwen3.5-122B-A10B-FP8 \
        --output "${HYBRID_DIR}" \
        --force
    step_end
fi

# ── Step 2: MTP weights ───────────────────────────────────────────────────────
if [ -f "${MODEL_DIR}/model_extra_tensors.safetensors" ] \
    && grep -q '"mtp\.' "${MODEL_DIR}/model.safetensors.index.json" 2>/dev/null; then
    STEP_NUM=$((STEP_NUM + 1))
    step_skip "Step 2 — MTP weights already present in ${MODEL_DIR}"
else
    step_begin "Step 2 — Adding MTP speculative decoding weights" \
               "copies model_extra_tensors.safetensors (~5 GB) and registers 785 tensors in the index"
    python "${PROJECT_DIR}/patches/02-mtp-speculative/add-mtp-weights.py" \
        --source "${INTEL_DIR}" \
        --target "${MODEL_DIR}"
    step_end
fi

# ── --no-cache: nuke existing images and BuildKit cache ──────────────────────
if [ "$NO_CACHE" = "1" ]; then
    log "${C_YEL}--no-cache: removing existing images and pruning BuildKit cache${C_OFF}"
    docker rmi -f vllm-qwen35-v2:latest 2>/dev/null || true
    docker rmi -f vllm-sm121:latest    2>/dev/null || true
    docker builder prune -af >/dev/null 2>&1 || true
    note "all stale layers gone — Step 3 will rebuild from scratch"
fi

# ── Step 3: build vllm-sm121 ──────────────────────────────────────────────────
if docker image inspect vllm-sm121:latest >/dev/null 2>&1; then
    STEP_NUM=$((STEP_NUM + 1))
    step_skip "Step 3 — vllm-sm121:latest already exists (delete with 'docker rmi vllm-sm121' to rebuild, or pass --no-cache)"
else
    step_begin "Step 3 — Building vllm-sm121 base image for SM121" \
               "first build: ~30-60 min (compiles vLLM, FlashInfer, NCCL for SM121); cached: ~3 min"

    # Clone or refresh upstream
    if [ ! -d "${SPARK_VLLM_DIR}/.git" ]; then
        note "cloning eugr/spark-vllm-docker into ${SPARK_VLLM_DIR}"
        git clone https://github.com/eugr/spark-vllm-docker.git "${SPARK_VLLM_DIR}"
    else
        note "spark-vllm-docker already cloned, refreshing"
        git -C "${SPARK_VLLM_DIR}" fetch --quiet origin
    fi

    # Pin to the exact commit our reference image was built with
    git -C "${SPARK_VLLM_DIR}" -c advice.detachedHead=false checkout --force "${SPARK_VLLM_PIN}"

    # Strip two upstream "TEMPORARY PATCH" RUN blocks (PR 35568, PR 38919).
    # Both were force-pushed after our 2026-04-04 build and no longer apply
    # to v0.19.0. Our reference image was verified to never have applied
    # them in the first place (marlin_utils.py md5 matches pristine v0.19.0).
    sed -i '/# TEMPORARY PATCH for broken FP8 kernels/,/&& rm pr35568.diff/d' \
        "${SPARK_VLLM_DIR}/Dockerfile"
    sed -i '/# TEMPORARY PATCH for broken compilation/,/&& rm pr38919.diff/d' \
        "${SPARK_VLLM_DIR}/Dockerfile"

    # Sanity: nothing should still reference those PRs
    if grep -qE 'pr35568|pr38919' "${SPARK_VLLM_DIR}/Dockerfile"; then
        abort "sed didn't strip the PR blocks cleanly — upstream Dockerfile may have changed shape."
    fi

    # Suppress deprecation warning spam from CUTLASS×CUDA13: when nvcc compiles
    # vllm-flash-attn against CUTLASS, the host gcc emits hundreds of
    # 'double4 is deprecated, use double4_16a' warnings (CUTLASS hasn't
    # migrated to CUDA 13.x's new aligned vector types yet). Harmless but
    # alarming. Inject NVCC_APPEND_FLAGS via ENV right after the existing
    # TORCH_CUDA_ARCH_LIST line in the vllm-builder stage.
    if ! grep -q 'NVCC_APPEND_FLAGS' "${SPARK_VLLM_DIR}/Dockerfile"; then
        sed -i '/^ENV TORCH_CUDA_ARCH_LIST=/a ENV NVCC_APPEND_FLAGS="-Xcompiler=-Wno-deprecated-declarations -diag-suppress=20012 -diag-suppress=20013 -diag-suppress=20014 -diag-suppress=20015"' \
            "${SPARK_VLLM_DIR}/Dockerfile"
    fi

    # Build (must use build-and-copy.sh, not bare 'docker build', because the
    # upstream Dockerfile COPYs build-metadata.yaml which the script generates
    # at build time and removes on exit). --vllm-ref v0.19.0 + --tf5 are not
    # script defaults — they match the build_args of our reference image.
    # Note: build-and-copy.sh has no --no-cache flag of its own; cache is
    # already invalidated above (we ran 'docker builder prune -af' if --no-cache
    # was passed to install.sh).
    (
        cd "${SPARK_VLLM_DIR}"
        ./build-and-copy.sh -t vllm-sm121 --vllm-ref v0.19.0 --tf5 2>&1
    )

    docker image inspect vllm-sm121:latest >/dev/null 2>&1 \
        || abort "vllm-sm121:latest is not in 'docker images' after build-and-copy.sh — something failed silently."
    step_end
fi

# ── Step 4: build vllm-qwen35-v2 ──────────────────────────────────────────────
if docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1; then
    STEP_NUM=$((STEP_NUM + 1))
    step_skip "Step 4 — vllm-qwen35-v2:latest already exists (delete with 'docker rmi vllm-qwen35-v2' to rebuild, or pass --no-cache)"
else
    step_begin "Step 4 — Building vllm-qwen35-v2 (final image)" \
               "thin layer on top of vllm-sm121: copies hybrid INC patch, INT8 LM Head v2, entrypoint"
    cd "${PROJECT_DIR}"
    docker build -t vllm-qwen35-v2 -f docker/Dockerfile.v2 .
    docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1 \
        || abort "vllm-qwen35-v2:latest is not in 'docker images' after build."
    step_end
fi

# ── Done ──────────────────────────────────────────────────────────────────────
TOTAL=$(( $(date +%s) - START_TS ))
echo
echo "${C_GRN}════════════════════════════════════════════════════════════════════${C_OFF}"
ok "All build steps complete in $(fmt_time $TOTAL)"
echo "${C_GRN}════════════════════════════════════════════════════════════════════${C_OFF}"
echo
log "Images:"
docker images vllm-sm121     --format '   {{.Repository}}:{{.Tag}}   {{.Size}}' | grep -v '^$' || true
docker images vllm-qwen35-v2 --format '   {{.Repository}}:{{.Tag}}   {{.Size}}' | grep -v '^$' || true
echo
log "Model:"
echo "   ${MODEL_DIR}"
echo

# ── Step 5: launch (interactive prompt or via --launch / --no-launch) ────────
MODEL_BASENAME=$(basename "${MODEL_DIR}")
MODELS_PARENT=$(dirname "${MODEL_DIR}")

LAUNCH_CMD="docker run -d --name vllm-qwen35 \\
    --gpus all --net=host --ipc=host \\
    -v ${MODELS_PARENT}:/models \\
    vllm-qwen35-v2 \\
    serve /models/${MODEL_BASENAME} \\
    --served-model-name qwen \\
    --port 8000 \\
    --max-model-len 262144 \\
    --gpu-memory-utilization 0.90 \\
    --reasoning-parser qwen3 \\
    --enable-auto-tool-choice \\
    --tool-call-parser qwen3_xml \\
    --attention-backend FLASHINFER \\
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'"

# Measured on a real DGX Spark first-launch run from this exact image:
# weights load 9m45s + compile/warmup 2m51s + graph capture/engine 29s
# + API server bind 17s = 13m22s total. Use this as the progress estimate.
EXPECTED_LAUNCH_SECS=802

print_launch_cmd() {
    cat <<EOF
${C_CYN}To launch manually later (Step 5 in README):${C_OFF}

  $LAUNCH_CMD

Wait ~13 min for model load + warmup, then:

  curl http://127.0.0.1:8000/health

For TurboQuant (4× KV cache, -22% speed) see the "Optional: TurboQuant
KV Cache Compression" section in README.md — that variant is intentionally
outside this install script. Benchmark with: ./bench_qwen35.sh "v2"
EOF
}

# When the vllm-qwen35 container dies during startup, collect enough info
# for the user (or us on the forum) to triage without another round-trip.
# Prints (in order):
#   1. First EngineCore Error/Traceback block (the real root cause)
#   2. Last 200 log lines (fallback for errors without a Python traceback)
#   3. GPU state (who's holding memory, driver version)
#   4. Host memory pressure
#   5. /dev/shm size (vLLM multiprocess IPC uses it via --ipc=host)
#   6. A retry hint — many first-run failures are transient (stale CUDA
#      contexts from prior experiments), and install.sh re-runs are
#      idempotent so the fix is usually just "run it again".
dump_post_mortem() {
    # Cache the full log once — we're going to slice it three different ways.
    local log_file="/tmp/vllm-qwen35-crash.log"
    docker logs vllm-qwen35 > "$log_file" 2>&1
    local total_lines
    total_lines=$(wc -l < "$log_file")
    note "full log saved to $log_file ($total_lines lines)"

    echo
    err "─── 1. ROOT CAUSE (first EngineCore error/traceback) ─────────────────"
    # Pass 1: awk for the first block of EngineCore lines starting at the
    # first Error/Traceback/Exception/Failed/FATAL. Prints up to 40 EngineCore
    # lines so a full Python traceback fits.
    local root_cause
    root_cause=$(awk '
        /^\(EngineCore/ && /Error|Traceback|Exception|Failed|FATAL/ {found=1}
        found && /^\(EngineCore/ {print; count++}
        count >= 40 {exit}
    ' "$log_file" 2>/dev/null)
    if [ -n "$root_cause" ]; then
        echo "$root_cause"
    else
        echo "(no EngineCore Python traceback found — crash may be below the"
        echo " vLLM Python layer. Checking for other error signals below.)"
    fi

    echo
    err "─── 2. ALL ERROR/TRACEBACK LINES ACROSS WHOLE LOG ────────────────────"
    # Pass 2: grep the entire log (not just EngineCore) for any error-ish
    # line with 2 lines of context before and 5 after. Catches lower-level
    # crashes: CUDA driver errors, nccl fails, Rust panics, assertion
    # failures from native .so, shm allocation errors, etc. Dedup via uniq
    # so repeated warnings don't flood. Cap at 80 lines so we don't dump
    # megabytes.
    local any_errors
    any_errors=$(grep -n -B 2 -A 5 -iE '\b(error|traceback|exception|fatal|failed|panic|assertion|sigkill|signal|core dumped|cannot allocate|out of memory|oom|no such file)\b' "$log_file" 2>/dev/null \
        | head -80)
    if [ -n "$any_errors" ]; then
        echo "$any_errors"
    else
        echo "(no explicit error keywords in log — very unusual, see tail)"
    fi

    echo
    err "─── 3. LAST 200 LOG LINES (fallback context) ────────────────────────"
    tail -200 "$log_file"

    echo
    err "─── HOST DIAGNOSTICS ─────────────────────────────────────────────────"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU state:"
        nvidia-smi --query-gpu=name,driver_version,memory.free,memory.used,memory.total \
            --format=csv 2>&1 | sed 's/^/  /'
        local running_apps
        running_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>&1)
        if [ -n "$running_apps" ] && [ "$running_apps" != "No running processes found" ]; then
            echo "Processes currently on GPU:"
            echo "$running_apps" | sed 's/^/  /'
        else
            echo "  (no other processes on GPU)"
        fi
    else
        echo "nvidia-smi not found on host"
    fi

    echo "Host memory:"
    free -h 2>&1 | sed 's/^/  /'

    echo "/dev/shm (used by --ipc=host for vLLM multiprocess IPC):"
    df -h /dev/shm 2>&1 | sed 's/^/  /'

    echo
    err "─── NEXT STEPS ───────────────────────────────────────────────────────"
    cat <<EOF
  1. If there's an EngineCore Python traceback above — that's the real error.
     Common ones:
       - 'CUDA out of memory' → another process is holding GPU memory.
         Check 'Processes currently on GPU' above; stop that process or
         add '--gpu-memory-utilization 0.70' to the launch command.
       - 'No such file or directory' for a model shard → Step 1 or Step 2
         didn't finish. Re-run ./install.sh (it's idempotent).
       - 'Tokenizer class ... does not exist' → you built with an upstream
         HEAD of eugr/spark-vllm-docker instead of our pinned commit.
         Run './install.sh --no-cache' to rebuild from the correct pin.

  2. If you see no Python traceback — the crash was below vLLM's level
     (CUDA driver, nvidia-container-toolkit, OOM SIGKILL). Try:
       docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi
     If that fails, install nvidia-container-toolkit:
       sudo apt install -y nvidia-container-toolkit
       sudo systemctl restart docker

  3. If everything above looks fine — this may be a transient failure
     (stale CUDA context from earlier experiments, half-dead worker, etc.).
     Just re-run the script — it's idempotent and will skip straight to
     the launch step:
       ./install.sh --launch
EOF
}

# Poll /health while showing a progress bar + the current vLLM startup stage,
# parsed live from container logs. Returns 0 when /health is 200, non-zero on
# timeout or container death. Uses 127.0.0.1 (not localhost) to avoid the
# IPv6 ::1 resolution gotcha on some Linux setups.
poll_health_with_progress() {
    local start_ts now elapsed pct bar_full bar i timeout=1500
    local stage_marker stage line
    start_ts=$(date +%s)

    note "model loading takes ~$(fmt_time $EXPECTED_LAUNCH_SECS) on first run (cached re-launch: ~5-7 min)"
    note "polling http://127.0.0.1:8000/health every 5 sec — Ctrl-C to detach (container keeps running)"
    echo

    while true; do
        now=$(date +%s)
        elapsed=$((now - start_ts))

        # Hard timeout
        if [ "$elapsed" -gt "$timeout" ]; then
            echo
            err "timeout after $(fmt_time $elapsed) — vLLM did not become ready"
            dump_post_mortem
            return 1
        fi

        # Container died?
        if ! docker ps --filter name=vllm-qwen35 --format '{{.Names}}' | grep -qx vllm-qwen35; then
            echo
            err "container 'vllm-qwen35' has died after $(fmt_time $elapsed)"
            dump_post_mortem
            return 1
        fi

        # Health probe (silent, just exit code)
        if curl -sf -m 2 http://127.0.0.1:8000/health -o /dev/null 2>&1; then
            echo
            echo
            ok "${C_GRN}vLLM is ready!${C_OFF}  Total startup: $(fmt_time $elapsed)  ${C_DIM}(estimated $(fmt_time $EXPECTED_LAUNCH_SECS))${C_OFF}"
            return 0
        fi

        # Detect current stage by tailing the log for known marker lines
        stage_marker=$(docker logs vllm-qwen35 2>&1 | grep -oE 'Loading safetensors checkpoint shards: *[0-9]+%|Loading weights took|torch\.compile took|DGX_SPARK_V2: LM Head|Graph capturing finished|init engine.*took|Starting vLLM server|Started server process|Application startup complete' 2>/dev/null | tail -1)

        case "$stage_marker" in
            *"Application startup complete"*) stage="API server up — verifying health" ;;
            *"Started server process"*)       stage="FastAPI / uvicorn starting" ;;
            *"Starting vLLM server"*)         stage="API server starting on :8000" ;;
            *"init engine"*"took"*)           stage="Engine init done — handing off to API server" ;;
            *"Graph capturing finished"*)     stage="CUDA graphs captured — finalizing" ;;
            *"DGX_SPARK_V2: LM Head"*)        stage="INT8 LM Head v2 patch applied — capturing CUDA graphs" ;;
            *"torch.compile took"*)           stage="torch.compile done — running profiling/warmup" ;;
            *"Loading weights took"*)         stage="Weights loaded — torch.compile starting" ;;
            *"Loading safetensors"*)
                pct=$(echo "$stage_marker" | grep -oE '[0-9]+%' | tail -1)
                stage="Loading weights ${pct}"
                ;;
            *) stage="initializing (entrypoint, vLLM CLI)" ;;
        esac

        # Build progress bar (24 chars wide). Cap at 99% until health is OK.
        pct=$((elapsed * 100 / EXPECTED_LAUNCH_SECS))
        [ "$pct" -gt 99 ] && pct=99
        bar_full=$((pct * 24 / 100))
        bar=""
        for ((i=0; i<24; i++)); do
            if [ "$i" -lt "$bar_full" ]; then bar+="█"; else bar+="░"; fi
        done

        # Single-line progress (\r overwrite, \033[K clears to end of line)
        line="  ${C_CYN}[%3d%%]${C_OFF} ${bar}  ${C_DIM}%s / ~%s${C_OFF}  %s"
        printf '\r\033[K'"$line" "$pct" "$(fmt_time $elapsed)" "$(fmt_time $EXPECTED_LAUNCH_SECS)" "$stage"

        sleep 5
    done
}

do_launch() {
    log "Launching vllm-qwen35..."

    # Remove any stale container with the same name
    if docker ps -a --format '{{.Names}}' | grep -qx vllm-qwen35; then
        warn "container 'vllm-qwen35' already exists — removing it first"
        docker rm -f vllm-qwen35 >/dev/null
    fi

    # Warn if port 8000 is occupied (don't refuse — user may have intentional
    # setup with --net=host that supersedes this check anyway)
    if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ':8000 '; then
        warn "port 8000 is already in use on the host. Container will likely fail to bind."
        warn "Stop the other process first, or change --port in the docker run command."
    fi

    # Launch
    eval "$LAUNCH_CMD" || { err "docker run failed"; return 1; }
    ok "container started in background as 'vllm-qwen35'"

    # Poll /health with live progress + stage parsing
    if poll_health_with_progress; then
        echo
        note "endpoint: http://127.0.0.1:8000/v1/chat/completions"
        note "health:   curl http://127.0.0.1:8000/health"
        note "logs:     docker logs -f vllm-qwen35"
        note "stop:     docker stop vllm-qwen35"

        # Smoke test: hit /v1/models so we know it's not just /health alive
        if curl -sf -m 5 http://127.0.0.1:8000/v1/models -o /tmp/.vllm_models.$$ 2>/dev/null; then
            if grep -q '"qwen"' /tmp/.vllm_models.$$ 2>/dev/null; then
                ok "model 'qwen' is registered and serving requests"
            fi
            rm -f /tmp/.vllm_models.$$
        fi
    fi
}

case "$LAUNCH_MODE" in
    yes)
        echo
        do_launch
        ;;
    no)
        echo
        print_launch_cmd
        ;;
    prompt)
        echo
        # If stdin isn't a TTY (piped/CI), default to no-launch
        if [ ! -t 0 ]; then
            note "non-interactive shell — skipping launch prompt"
            print_launch_cmd
        else
            read -r -p "${C_CYN}Launch the container now? [y/N] ${C_OFF}" reply
            if [[ "${reply}" =~ ^[Yy]$ ]]; then
                do_launch
            else
                print_launch_cmd
            fi
        fi
        ;;
esac
