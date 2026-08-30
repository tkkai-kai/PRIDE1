#!/usr/bin/env bash

pride_hpc_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

pride_load_hpc_config() {
    local config
    config="$(pride_hpc_dir)/sheffield.env"
    if [[ ! -f "${config}" ]]; then
        echo "ERROR: missing Sheffield configuration: ${config}" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${config}"
    export PRIDE_CLUSTER SLURM_PARTITION SLURM_QOS SLURM_GRES SLURM_TIME SLURM_EXCLUDE
    export SLURM_MAIL_USER SLURM_MAIL_TYPE
    export CONDA_MODULE CONDA_MODULE_CANDIDATES CONDA_ENV CONDA_ENV_PREFIX
    export PRIDE_DEVICE MUJOCO_GL
}

pride_hpc_init() {
    local repo_root="${1:?repository root required}"
    cd "${repo_root}"
    pride_load_hpc_config
}

pride_sbatch_submit() {
    pride_load_hpc_config
    local -a cluster_args=(
        --partition="${SLURM_PARTITION}"
        --qos="${SLURM_QOS}"
        --gres="${SLURM_GRES}"
        --time="${SLURM_TIME}"
    )
    if [[ -n "${SLURM_EXCLUDE}" ]]; then
        cluster_args+=(--exclude="${SLURM_EXCLUDE}")
    fi
    if [[ -n "${SLURM_MAIL_USER}" ]]; then
        cluster_args+=(
            --mail-user="${SLURM_MAIL_USER}"
            --mail-type="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL,TIME_LIMIT}"
        )
    fi
    sbatch "${cluster_args[@]}" "$@"
}

pride_init_modules() {
    if type module &>/dev/null; then
        return 0
    fi

    local candidate
    for candidate in \
        "${MODULESHOME:+${MODULESHOME}/init/bash}" \
        /etc/profile.d/modules.sh \
        /etc/profile.d/lmod.sh \
        /usr/share/lmod/lmod/init/bash; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            # shellcheck disable=SC1090
            source "${candidate}"
            return 0
        fi
    done

    echo "ERROR: unable to initialize the environment-modules command" >&2
    return 1
}

# Login and compute nodes can serve different module trees (EL7 vs EL9 on
# Stanage), so try every known base-conda module and keep the first that yields
# a usable `conda`.
pride_load_conda_module() {
    local candidate
    for candidate in ${CONDA_MODULE_CANDIDATES:-${CONDA_MODULE}}; do
        if module load "${candidate}" 2>/dev/null && command -v conda &>/dev/null; then
            echo "conda module=${candidate}"
            return 0
        fi
        module unload "${candidate}" 2>/dev/null || true
    done
    return 1
}

# Last resort when no base-conda module is reachable: the env is self-contained,
# so putting its bin directory first is equivalent to `conda activate` here.
pride_activate_conda_prefix() {
    local prefix="${CONDA_ENV_PREFIX:-${HOME}/.conda/envs/${CONDA_ENV}}"
    if [[ ! -x "${prefix}/bin/python" ]]; then
        echo "ERROR: no conda module loaded and no usable env at ${prefix}." >&2
        echo "       Tried modules: ${CONDA_MODULE_CANDIDATES:-${CONDA_MODULE:-none}}" >&2
        return 1
    fi
    export CONDA_PREFIX="${prefix}"
    export CONDA_DEFAULT_ENV="${CONDA_ENV}"
    export PATH="${prefix}/bin:${PATH}"
    echo "conda module=none (activating prefix ${prefix})"
}

pride_slurm_job_init() {
    local entrypoint="${1:?training entrypoint required}"
    local repo_root="${PRIDE_ROOT:-${SLURM_SUBMIT_DIR:-}}"

    if [[ -z "${repo_root}" || ! -f "${repo_root}/${entrypoint}" ]]; then
        echo "ERROR: submit from the PRIDE repository root" >&2
        return 1
    fi

    cd "${repo_root}"
    pride_load_hpc_config
    pride_init_modules
    module purge 2>/dev/null || true

    : "${PS1:=}"
    set +u
    # Drop submit-shell conda state so compute-node activation is clean.
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE
    if pride_load_conda_module; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    else
        pride_activate_conda_prefix || return 1
    fi
    set -u

    # GPU Anaconda modules can activate a namesake env that is not the home prefix
    # (interactive `conda activate synprefenv` then has no gym). Prefer the home env.
    home_env="${HOME}/.conda/envs/${CONDA_ENV}"
    if [[ -x "${home_env}/bin/python" ]]; then
        export PATH="${home_env}/bin:${PATH}"
        export CONDA_PREFIX="${home_env}"
        export CONDA_DEFAULT_ENV="${CONDA_ENV}"
    fi

    export MUJOCO_GL PRIDE_DEVICE
    export CPATH="${CONDA_PREFIX}/include${CPATH:+:${CPATH}}"
    mujoco_bin="${HOME}/.mujoco/mujoco210/bin"
    ld_parts=()
    if [[ -d "${mujoco_bin}" ]]; then
        ld_parts+=("${mujoco_bin}")
    fi
    if [[ -d /usr/lib/nvidia ]]; then
        ld_parts+=("/usr/lib/nvidia")
    fi
    ld_parts+=("${CONDA_PREFIX}/lib")
    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        ld_parts+=("${LD_LIBRARY_PATH}")
    fi
    export LD_LIBRARY_PATH="$(IFS=:; echo "${ld_parts[*]}")"
    export PRIDE_ENTRY_MODULE="${entrypoint%.py}"
    mkdir -p outputs slurm_output
    echo "cluster=${PRIDE_CLUSTER} partition=${SLURM_PARTITION} env=${CONDA_DEFAULT_ENV:-${CONDA_ENV}} device=${PRIDE_DEVICE}"
    echo "python=$(command -v python)"
    python - <<'PY'
import importlib
import os
import sys
print(f"sys.executable={sys.executable}")
for module_name in ("termcolor", "logger", "utils", os.environ["PRIDE_ENTRY_MODULE"]):
    importlib.import_module(module_name)
    print(f"import_ok={module_name}")
PY
}
if false; then
#!/usr/bin/env bash

pride_hpc_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

pride_load_hpc_config() {
    local config
    config="$(pride_hpc_dir)/sheffield.env"
    if [[ ! -f "${config}" ]]; then
        echo "ERROR: missing Sheffield configuration: ${config}" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${config}"
    export PRIDE_CLUSTER SLURM_PARTITION SLURM_QOS SLURM_GRES SLURM_TIME SLURM_EXCLUDE
    export CONDA_MODULE CONDA_ENV PRIDE_DEVICE MUJOCO_GL
}

pride_hpc_init() {
    local repo_root="${1:?repository root required}"
    cd "${repo_root}"
    pride_load_hpc_config
}

pride_sbatch_submit() {
    pride_load_hpc_config
    local -a cluster_args=(
        --partition="${SLURM_PARTITION}" \
        --qos="${SLURM_QOS}" \
        --gres="${SLURM_GRES}" \
        --time="${SLURM_TIME}"
    )
    if [[ -n "${SLURM_EXCLUDE}" ]]; then
        cluster_args+=(--exclude="${SLURM_EXCLUDE}")
    fi
    sbatch "${cluster_args[@]}" "$@"
}

pride_init_modules() {
    if type module &>/dev/null; then
        return 0
    fi

    local candidate
    for candidate in \
        "${MODULESHOME:+${MODULESHOME}/init/bash}" \
        /etc/profile.d/modules.sh \
        /etc/profile.d/lmod.sh \
        /usr/share/lmod/lmod/init/bash; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            # shellcheck disable=SC1090
            source "${candidate}"
            return 0
        fi
    done

    echo "ERROR: unable to initialize the environment-modules command" >&2
    return 1
}

pride_slurm_job_init() {
    local entrypoint="${1:?training entrypoint required}"
    local repo_root="${PRIDE_ROOT:-${SLURM_SUBMIT_DIR:-}}"

    if [[ -z "${repo_root}" || ! -f "${repo_root}/${entrypoint}" ]]; then
        echo "ERROR: submit from the PRIDE repository root" >&2
        return 1
    fi

    cd "${repo_root}"
    pride_load_hpc_config
    pride_init_modules
    module purge
    module load "${CONDA_MODULE}"

    : "${PS1:=}"
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u

    export MUJOCO_GL PRIDE_DEVICE
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    mkdir -p outputs slurm_output
    echo "cluster=${PRIDE_CLUSTER} partition=${SLURM_PARTITION} env=${CONDA_ENV} device=${PRIDE_DEVICE}"
}
fi
