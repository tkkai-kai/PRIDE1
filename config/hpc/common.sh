#!/usr/bin/env bash
# =============================================================================
# PRIDE HPC layer — Sheffield (Stanage) vs Dawn (CSD3)
# =============================================================================
#
# Layout:
#   config/hpc/sheffield.env | dawn.env     cluster Slurm + conda settings
#   config/hpc/common.sh                     shared functions (this file)
#   config/hpc/sbatch.sh                     login: wrap sbatch with cluster args
#   scripts/hpc/setup_dawn_env.sh            one-time Dawn conda env (login node)
#   scripts/hpc/submit_*.sh                login: submit many Slurm jobs
#   scripts/.../submit_*_slurm.sh          Slurm job body (#SBATCH + train loop)
#   scripts/.../run_*.sh                   local/interactive (no Slurm)
#
# Call flow (single job):
#   PRIDE_CLUSTER=dawn bash config/hpc/sbatch.sh scripts/.../submit_PRIDE_ablation_slurm.sh
#     -> sbatch adds partition/account from dawn.env
#     -> job runs: pride_slurm_job_init -> python train_PRIDE.py ...
#
# Multi-job experiment (login node):
#   PRIDE_CLUSTER=dawn bash scripts/hpc/submit_baseline_vs_g05_g06.sh
#
# Call flow (Dawn first-time setup):
#   bash scripts/hpc/setup_dawn_env.sh   # once on login node
#   PRIDE_CLUSTER=dawn bash config/hpc/sbatch.sh scripts/.../submit_*_slurm.sh
#
# =============================================================================

pride_hpc_dir() {
    if [[ -n "${PRIDE_HPC_DIR:-}" ]]; then
        printf '%s\n' "${PRIDE_HPC_DIR}"
        return 0
    fi
    printf '%s\n' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
}

pride_detect_cluster() {
    if [[ -n "${PRIDE_CLUSTER:-}" ]]; then
        printf '%s' "${PRIDE_CLUSTER}"
        return 0
    fi

    local host="${HOSTNAME:-$(hostname -f 2>/dev/null || hostname)}"
    local partition="${SLURM_JOB_PARTITION:-}"

    case "${partition}" in
        pvc9|ampere|ukaea-amp|ukaea-mi300x)
            printf 'dawn'
            return 0
            ;;
        gpu|gpu-h100)
            printf 'sheffield'
            return 0
            ;;
    esac

    case "${host}" in
        *data.cluster*)
            printf 'dawn'
            return 0
            ;;
        *stanage*|*shef*)
            printf 'sheffield'
            return 0
            ;;
    esac

    echo "ERROR: cannot detect HPC cluster; set PRIDE_CLUSTER=sheffield or PRIDE_CLUSTER=dawn" >&2
    return 1
}

pride_load_hpc_config() {
    local cluster config
    cluster="$(pride_detect_cluster)" || return 1
    PRIDE_CLUSTER="${cluster}"
    config="$(pride_hpc_dir)/${cluster}.env"
    if [[ ! -f "${config}" ]]; then
        echo "ERROR: missing cluster config ${config}" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${config}"
    export PRIDE_CLUSTER CONDA_MODULE CONDA_MODULE_CANDIDATES CONDA_ENV CONDA_ENV_PREFIX
    export CONDA_ENV_FILE PRIDE_DEVICE MUJOCO_GL PRIDE_IMPORT_IPEX
    export DAWN_MODULEFILES CONDA_CLONE_SOURCE SLURM_ACCOUNT SLURM_PARTITION SLURM_QOS SLURM_GRES SLURM_TIME
    export SLURM_EXCLUDE SLURM_MAIL_USER SLURM_MAIL_TYPE
}

pride_init_modules() {
    if [[ -n "${_PRIDE_MODULES_INIT:-}" ]]; then
        return 0
    fi
    # Already initialized (e.g. login shell): 'module' is defined, nothing to do.
    if type module &>/dev/null; then
        _PRIDE_MODULES_INIT=1
        return 0
    fi
    # Locate the module init script across clusters:
    #   Sheffield/Stanage -> Lmod at $MODULESHOME/init/bash (non-standard prefix)
    #   Dawn/CSD3 or classic environment-modules -> /etc/profile.d/modules.sh
    local candidate
    for candidate in \
        "${MODULESHOME:+${MODULESHOME}/init/bash}" \
        "${LMOD_PKG:+${LMOD_PKG}/init/bash}" \
        /etc/profile.d/modules.sh \
        /etc/profile.d/lmod.sh \
        /etc/profile.d/z00_lmod.sh \
        /usr/share/lmod/lmod/init/bash \
        /usr/share/Modules/init/bash; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            # shellcheck disable=SC1090
            source "${candidate}"
            _PRIDE_MODULES_INIT=1
            return 0
        fi
    done
    echo "ERROR: could not locate a module init script (tried MODULESHOME, LMOD_PKG, /etc/profile.d)." >&2
    return 1
}

pride_ensure_dawn_modulepath() {
    pride_init_modules
    local dawn_moddir="${DAWN_MODULEFILES:-/usr/local/dawn/software/modulefiles}"
    if [[ -d "${dawn_moddir}" ]]; then
        module use "${dawn_moddir}" 2>/dev/null || true
    fi
    if ! module avail intelpython-conda &>/dev/null; then
        echo "ERROR: intelpython-conda not in MODULEPATH (host=${HOSTNAME:-unknown})." >&2
        echo "       Dawn jobs need pvc9; ensure ${dawn_moddir} exists." >&2
        return 1
    fi
}

pride_sbatch_cluster_args() {
    if [[ -z "${PRIDE_CLUSTER:-}" ]]; then
        pride_load_hpc_config || return 1
    fi
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        printf '%s\n' --account="${SLURM_ACCOUNT}"
    fi
    if [[ -n "${SLURM_PARTITION:-}" ]]; then
        printf '%s\n' -p "${SLURM_PARTITION}"
    fi
    if [[ -n "${SLURM_QOS:-}" ]]; then
        printf '%s\n' --qos="${SLURM_QOS}"
    fi
    if [[ -n "${SLURM_GRES:-}" ]]; then
        printf '%s\n' --gres="${SLURM_GRES}"
    fi
    if [[ -n "${SLURM_TIME:-}" ]]; then
        printf '%s\n' --time="${SLURM_TIME}"
    fi
    if [[ -n "${SLURM_EXCLUDE:-}" ]]; then
        printf '%s\n' --exclude="${SLURM_EXCLUDE}"
    fi
    if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
        printf '%s\n' --mail-user="${SLURM_MAIL_USER}"
        printf '%s\n' --mail-type="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL,TIME_LIMIT}"
    fi
}

# --- Repo root (Slurm spool-safe) ---------------------------------------------

pride_repo_root() {
    if [[ -n "${PRIDE_ROOT:-}" ]]; then
        printf '%s\n' "${PRIDE_ROOT}"
        return 0
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        printf '%s\n' "${SLURM_SUBMIT_DIR}"
        return 0
    fi
    local script="${1:-${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}}"
    local dir
    dir="$(cd "$(dirname "${script}")" && pwd)"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/config/hpc/common.sh" ]]; then
            printf '%s\n' "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo "ERROR: cannot find PRIDE repo root" >&2
    return 1
}

# --- Slurm job scripts (run inside allocated node) ---------------------------

pride_slurm_job_init() {
    local check_file="${1:-train_PRIDE.py}"
    local root
    root="$(pride_repo_root "${BASH_SOURCE[1]}")"
    cd "${root}"
    mkdir -p slurm_output outputs
    if [[ ! -f "${check_file}" ]]; then
        echo "ERROR: ${check_file} not found under $(pwd)." >&2
        echo "Submit from repo root: PRIDE_CLUSTER=... bash config/hpc/sbatch.sh <job.sh>" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(pwd)/config/hpc/common.sh"
    pride_load_hpc_config
    pride_activate_runtime
    pride_import_preflight "${check_file%.py}"
}

# Fail fast (seconds, not hours) if the activated env cannot import the training
# stack — e.g. when submit-shell state shadows the conda interpreter.
pride_import_preflight() {
    local entry_module="${1:-train_PRIDE}"
    echo "python=$(command -v python)"
    PRIDE_ENTRY_MODULE="${entry_module}" python - <<'PY'
import importlib
import os
import sys

print(f"sys.executable={sys.executable}")
for module_name in ("termcolor", "logger", "utils", os.environ["PRIDE_ENTRY_MODULE"]):
    importlib.import_module(module_name)
    print(f"import_ok={module_name}")
PY
}

pride_slurm_batch_timer_start() {
    local tag="${1:?RUN_TAG required}"
    RUN_TAG="${tag}"
    BATCH_START_EPOCH=$(date +%s)
    BATCH_START_TIME=$(date '+%Y-%m-%d %H:%M:%S %z')
    trap pride_slurm_batch_timer_finish EXIT
    echo "=== ${RUN_TAG} start: ${BATCH_START_TIME} ==="
}

pride_slurm_batch_timer_finish() {
    local ec=$?
    local end_epoch end_time elapsed h m s
    end_epoch=$(date +%s)
    end_time=$(date '+%Y-%m-%d %H:%M:%S %z')
    elapsed=$((end_epoch - BATCH_START_EPOCH))
    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))
    s=$((elapsed % 60))
    echo "=== ${RUN_TAG} end: ${end_time} (exit=${ec}) ==="
    echo "Elapsed seconds: ${elapsed}"
    printf 'Elapsed (H:M:S): %d:%02d:%02d\n' "${h}" "${m}" "${s}"
}

# --- Login-node multi-job submitters ------------------------------------------

pride_hpc_init() {
    local repo_root="${1:?repo root required}"
    cd "${repo_root}"
    # shellcheck disable=SC1091
    source "${repo_root}/config/hpc/common.sh"
    pride_load_hpc_config
}

pride_sbatch_submit() {
    local -a cluster_args=()
    mapfile -t cluster_args < <(pride_sbatch_cluster_args)
    sbatch "${cluster_args[@]}" "$@"
}

pride_dawn_conda_env_exists() {
    conda env list | awk -v name="${CONDA_ENV}" 'NR>2 && $1==name {found=1} END{exit !found}'
}

pride_dawn_env_python_version() {
    conda run -n "${CONDA_ENV}" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown"
}

pride_dawn_env_is_healthy() {
    conda run -n "${CONDA_ENV}" python -c "import torch; import intel_extension_for_pytorch" &>/dev/null
}

pride_remove_dawn_conda_env() {
    local dst_path="${HOME}/.conda/envs/${CONDA_ENV}"
    echo "Removing env ${CONDA_ENV} at ${dst_path}..."
    rm -rf "${dst_path}"
}

pride_create_dawn_conda_env_if_missing() {
    local clone_source="${CONDA_CLONE_SOURCE:-pytorch-gpu}"
    if pride_dawn_conda_env_exists; then
        local pyver
        pyver="$(pride_dawn_env_python_version)"
        if [[ "${pyver}" != "3.9" ]]; then
            echo "WARNING: ${CONDA_ENV} has Python ${pyver}, expected 3.9; recreating from ${clone_source}."
            pride_remove_dawn_conda_env
        elif ! pride_dawn_env_is_healthy; then
            echo "WARNING: ${CONDA_ENV} is missing torch/ipex (often caused by 'conda env update --prune'); recreating."
            pride_remove_dawn_conda_env
        else
            return 0
        fi
    fi

    local src_path dst_path
    src_path="$(conda env list | awk -v name="${clone_source}" 'NR>2 && $1==name {print $NF; exit}')"
    dst_path="${HOME}/.conda/envs/${CONDA_ENV}"

    if [[ -z "${src_path}" || ! -d "${src_path}" ]]; then
        echo "ERROR: source env ${clone_source} not found." >&2
        return 1
    fi

    if [[ -e "${dst_path}" ]]; then
        pride_remove_dawn_conda_env
    fi

    echo "Copying ${clone_source} -> ${CONDA_ENV} (local filesystem, no download)..."
    echo "  src: ${src_path}"
    echo "  dst: ${dst_path}"
    mkdir -p "$(dirname "${dst_path}")"
    cp -a "${src_path}" "${dst_path}"

    echo "Rewriting env prefix paths..."
    grep -rIl "${src_path}" "${dst_path}" 2>/dev/null | while IFS= read -r f; do
        sed -i "s|${src_path}|${dst_path}|g" "$f"
    done
}

pride_dawn_install_mesalib() {
    # Install mesalib only — never use 'conda env update --prune' (removes torch/ipex).
    # --no-update-deps: avoid re-solving the whole Intel stack (slow + risky).
    if python -c "import os; import glob; exit(0 if glob.glob(os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'libOSMesa*.so*')) else 1)" 2>/dev/null; then
        echo "mesalib/OSMesa libs already present, skipping conda install."
        return 0
    fi
    conda install -y -c conda-forge mesalib --no-update-deps
}

pride_dawn_pip_install() {
    # mujoco: must use a cp39 wheel — unpinned pip may try to build 3.9.x from source (needs MUJOCO_PATH).
    pip install --only-binary=mujoco 'mujoco==3.2.3'
    pip install -r requirements_dawn.txt
    pip install 'git+https://github.com/facebookresearch/hydra@0.11_branch'
    pip install --no-deps 'dm_control==1.0.23'
    pip install --no-deps 'accelerate==0.12.0' 'ema-pytorch==0.2.3'
}

pride_setup_osmesa_symlinks() {
    # mesalib may not be installed yet on first setup; never fail activation if absent.
    if [[ "${MUJOCO_GL:-}" != "osmesa" || -z "${CONDA_PREFIX:-}" ]]; then
        return 0
    fi
    if [[ -e "${CONDA_PREFIX}/lib/libOSMesa.so" ]]; then
        return 0
    fi
    local _osmesa_lib _osmesa_libs=()
    shopt -s nullglob
    _osmesa_libs=("${CONDA_PREFIX}"/lib/libOSMesa32.so.*)
    shopt -u nullglob
    if [[ ${#_osmesa_libs[@]} -eq 0 ]]; then
        return 0
    fi
    _osmesa_lib="${_osmesa_libs[0]}"
    ln -sf "$(basename "${_osmesa_lib}")" "${CONDA_PREFIX}/lib/libOSMesa.so"
    ln -sf "$(basename "${_osmesa_lib}")" "${CONDA_PREFIX}/lib/libOSMesa.so.0"
}

# Compute and login nodes can serve different module trees (EL9 vs EL7 on
# Stanage), so try every known base-conda module and keep the first that yields a
# usable `conda`.
pride_load_conda_module() {
    local candidate
    for candidate in ${CONDA_MODULE_CANDIDATES:-${CONDA_MODULE:-}}; do
        if module load "${candidate}" 2>/dev/null && command -v conda &>/dev/null; then
            echo "conda module=${candidate}"
            return 0
        fi
        module unload "${candidate}" 2>/dev/null || true
    done
    return 1
}

# Last resort when no base-conda module is reachable: the env is self-contained,
# so putting its bin dir first is equivalent to `conda activate` for our purposes.
pride_activate_conda_prefix() {
    local prefix="${CONDA_ENV_PREFIX:-${HOME}/.conda/envs/${CONDA_ENV}}"
    if [[ ! -x "${prefix}/bin/python" ]]; then
        echo "ERROR: no conda module loaded and no usable env at ${prefix}." >&2
        echo "       Tried modules: ${CONDA_MODULE_CANDIDATES:-${CONDA_MODULE:-none}}" >&2
        return 1
    fi
    echo "conda module=none (activating prefix ${prefix})"
    export CONDA_PREFIX="${prefix}"
    export CONDA_DEFAULT_ENV="${CONDA_ENV}"
    export PATH="${prefix}/bin:${PATH}"
}

pride_activate_runtime() {
    if [[ -z "${PRIDE_CLUSTER:-}" || -z "${CONDA_MODULE:-}" ]]; then
        pride_load_hpc_config || return 1
    fi
    pride_init_modules
    if [[ "${PRIDE_CLUSTER}" == "dawn" ]]; then
        pride_ensure_dawn_modulepath || return 1
    fi
    module purge 2>/dev/null || true
    # conda.sh reads $PS1; Slurm batch shells are non-interactive and omit it.
    # Caller scripts often use `set -u`, which would abort on that reference.
    : "${PS1:=}"
    _pride_nounset=0
    [[ $- == *u* ]] && _pride_nounset=1
    set +u
    # Drop submit-shell conda state so compute-node activation is clean.
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE
    if pride_load_conda_module; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if [[ "${PRIDE_CLUSTER}" == "dawn" ]]; then
            if [[ "${PRIDE_CREATE_DAWN_ENV:-0}" == "1" ]]; then
                pride_create_dawn_conda_env_if_missing || return 1
            elif ! pride_dawn_conda_env_exists; then
                echo "ERROR: conda env '${CONDA_ENV}' not found." >&2
                echo "       Run once: bash scripts/hpc/setup_dawn_env.sh" >&2
                return 1
            fi
        fi
        conda activate "${CONDA_ENV}"
    else
        pride_activate_conda_prefix || return 1
    fi
    [[ "${_pride_nounset}" -eq 1 ]] && set -u
    unset _pride_nounset
    if [[ "${PRIDE_IMPORT_IPEX:-0}" == "1" ]]; then
        python -c "import intel_extension_for_pytorch" 2>/dev/null || {
            echo "WARNING: intel_extension_for_pytorch not found; run bash scripts/hpc/setup_dawn_env.sh" >&2
        }
    fi
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PRIDE_DEVICE="${PRIDE_DEVICE:-cuda}"
    # PyOpenGL/dm_control dlopen MuJoCo render libs (e.g. libOSMesa.so) by name;
    # ensure the conda lib dir is searched so osmesa rendering can load.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        pride_setup_osmesa_symlinks
    fi
    echo "HPC cluster=${PRIDE_CLUSTER} conda_env=${CONDA_ENV} device=${PRIDE_DEVICE}"
}
