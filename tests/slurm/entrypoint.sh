#!/usr/bin/env bash
# entrypoint.sh — Role-based entrypoint for the 2-node SLURM+MPI test cluster.
#
# ROLE=controller  starts munged + slurmctld + slurmd, generates the shared
#                  munge key on the shared volume, and waits for both nodes.
# ROLE=compute     waits for the shared munge key, then starts munged + slurmd.
set -euo pipefail

SHARED_DIR="/shared"
MUNGE_KEY_PATH="${SHARED_DIR}/munge.key"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_fix_slurm_dirs() {
    chown -R slurm:slurm /var/spool/slurm /var/log/slurm /var/run/slurm
}

_start_munge() {
    echo "[entrypoint] Starting munged..."
    chown -R munge:munge /var/log/munge /var/lib/munge /run/munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
    runuser -u munge -- /usr/sbin/munged --foreground &
    MUNGE_PID=$!
    # Give munged a moment to create its socket
    sleep 2
    if ! kill -0 "${MUNGE_PID}" 2>/dev/null; then
        echo "[entrypoint] ERROR: munged failed to start"
        exit 1
    fi
    echo "[entrypoint] munged running (PID ${MUNGE_PID})"
}

_verify_munge() {
    echo "[entrypoint] Verifying munge..."
    if munge -n | unmunge > /dev/null 2>&1; then
        echo "[entrypoint] munge authentication OK"
    else
        echo "[entrypoint] ERROR: munge authentication failed"
        exit 1
    fi
}

_start_dbus() {
    # dbus is required by slurmd's cgroup/v2 plugin in SLURM 23.11+
    echo "[entrypoint] Starting dbus..."
    mkdir -p /run/dbus
    if [[ ! -S /run/dbus/system_bus_socket ]]; then
        dbus-daemon --system --fork 2>/dev/null || echo "[entrypoint] WARNING: dbus start failed (may already be running)"
        sleep 1
    fi
}

_prepare_cgroup() {
    # Docker containers with cgroup v2 don't have system.slice (created by systemd).
    # slurmd's cgroup/v2 plugin needs the scope directory to exist.
    # Create it manually so slurmd can initialize.
    local hostname
    hostname=$(hostname)
    local scope_dir="/sys/fs/cgroup/system.slice/${hostname}_slurmstepd.scope"
    if [[ ! -d /sys/fs/cgroup/system.slice ]]; then
        echo "[entrypoint] Creating cgroup system.slice for slurmd..."
        mkdir -p /sys/fs/cgroup/system.slice 2>/dev/null || true
    fi
    if [[ ! -d "${scope_dir}" ]]; then
        mkdir -p "${scope_dir}" 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
# controller role
# ---------------------------------------------------------------------------

_run_controller() {
    echo "[entrypoint] Role: controller"

    # 1. Generate munge key and write to shared volume
    echo "[entrypoint] Generating munge key on shared volume..."
    dd if=/dev/urandom bs=1 count=1024 > "${MUNGE_KEY_PATH}" 2>/dev/null
    chmod 644 "${MUNGE_KEY_PATH}"   # readable by compute containers before they chown it

    # 2. Install munge key locally
    cp "${MUNGE_KEY_PATH}" /etc/munge/munge.key
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key

    # 3. Start munged
    _start_munge

    # 4. Verify munge
    _verify_munge

    # 5. Start slurmctld
    echo "[entrypoint] Starting slurmctld..."
    _fix_slurm_dirs
    /usr/sbin/slurmctld -D &
    SLURMCTLD_PID=$!
    sleep 2
    if ! kill -0 "${SLURMCTLD_PID}" 2>/dev/null; then
        echo "[entrypoint] ERROR: slurmctld failed to start"
        cat /var/log/slurm/slurmctld.log || true
        exit 1
    fi
    echo "[entrypoint] slurmctld running (PID ${SLURMCTLD_PID})"

    # 6. Wait for slurmctld to become responsive
    echo "[entrypoint] Waiting for slurmctld to respond..."
    for i in $(seq 1 30); do
        if scontrol ping > /dev/null 2>&1; then
            echo "[entrypoint] slurmctld is responsive (attempt ${i})"
            break
        fi
        if [[ "${i}" -eq 30 ]]; then
            echo "[entrypoint] ERROR: slurmctld did not respond within 30s"
            cat /var/log/slurm/slurmctld.log || true
            exit 1
        fi
        sleep 1
    done

    # 7. Prepare cgroup and dbus (required by slurmd cgroup/v2)
    _start_dbus
    _prepare_cgroup

    # 8. Start slurmd on the controller node as well
    echo "[entrypoint] Starting slurmd (controller node)..."
    /usr/sbin/slurmd -D &
    SLURMD_PID=$!
    sleep 2
    if ! kill -0 "${SLURMD_PID}" 2>/dev/null; then
        echo "[entrypoint] ERROR: slurmd failed to start on controller"
        cat /var/log/slurm/slurmd.log || true
        exit 1
    fi
    echo "[entrypoint] slurmd running (PID ${SLURMD_PID})"

    # 9. Wait for controller's own slurmd to register (not node2 — that would deadlock
    #    because node2 depends_on controller being healthy via docker-compose).
    echo "[entrypoint] Waiting for controller node to register as idle..."
    for i in $(seq 1 30); do
        if sinfo -N --noheader 2>/dev/null | grep -q "controller.*idle"; then
            echo "[entrypoint] Controller node is idle (attempt ${i})"
            break
        fi
        if [[ "${i}" -eq 30 ]]; then
            echo "[entrypoint] ERROR: controller node did not become idle within 30s"
            sinfo -N || true
            scontrol show nodes || true
            exit 1
        fi
        sleep 1
    done

    # 10. Mark container healthy — this unblocks node2's depends_on
    touch /tmp/container-healthy
    echo "[entrypoint] Controller is healthy. Waiting for node2 to join..."

    # 11. Now wait for both nodes to appear idle (node2 is starting in parallel)
    for i in $(seq 1 60); do
        IDLE_COUNT=$(sinfo -N --state=idle --noheader 2>/dev/null | wc -l || echo 0)
        if [[ "${IDLE_COUNT}" -ge 2 ]]; then
            echo "[entrypoint] Both nodes are idle (attempt ${i}). Cluster status:"
            sinfo -N
            break
        fi
        if [[ "${i}" -eq 60 ]]; then
            echo "[entrypoint] WARNING: node2 did not become idle within 120s"
            sinfo -N || true
        fi
        sleep 2
    done

    # 10. Wait for any background service to exit
    wait -n "${MUNGE_PID}" "${SLURMCTLD_PID}" "${SLURMD_PID}"
    echo "[entrypoint] ERROR: a service exited unexpectedly"
    exit 1
}

# ---------------------------------------------------------------------------
# compute role
# ---------------------------------------------------------------------------

_run_compute() {
    echo "[entrypoint] Role: compute"

    # 1. Wait for munge key from controller
    echo "[entrypoint] Waiting for munge key on shared volume..."
    for i in $(seq 1 60); do
        if [[ -f "${MUNGE_KEY_PATH}" ]]; then
            echo "[entrypoint] munge key found (attempt ${i})"
            break
        fi
        if [[ "${i}" -eq 60 ]]; then
            echo "[entrypoint] ERROR: munge key did not appear within 60s"
            exit 1
        fi
        sleep 1
    done

    # 2. Install munge key locally
    cp "${MUNGE_KEY_PATH}" /etc/munge/munge.key
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key

    # 3. Start munged
    _start_munge

    # 4. Verify munge
    _verify_munge

    # 5. Prepare cgroup and dbus (required by slurmd cgroup/v2)
    _start_dbus
    _prepare_cgroup

    # 6. Start slurmd
    echo "[entrypoint] Starting slurmd..."
    _fix_slurm_dirs
    /usr/sbin/slurmd -D &
    SLURMD_PID=$!
    sleep 2
    if ! kill -0 "${SLURMD_PID}" 2>/dev/null; then
        echo "[entrypoint] ERROR: slurmd failed to start"
        cat /var/log/slurm/slurmd.log || true
        exit 1
    fi
    echo "[entrypoint] slurmd running (PID ${SLURMD_PID})"

    # 6. Mark container healthy
    touch /tmp/container-healthy
    echo "[entrypoint] Compute node ready"

    # 7. Wait for background service to exit
    wait -n "${MUNGE_PID}" "${SLURMD_PID}"
    echo "[entrypoint] ERROR: a service exited unexpectedly"
    exit 1
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

: "${ROLE:?ROLE environment variable must be set to 'controller' or 'compute'}"

case "${ROLE}" in
    controller)
        _run_controller
        ;;
    compute)
        _run_compute
        ;;
    *)
        echo "[entrypoint] ERROR: unknown ROLE '${ROLE}' — must be 'controller' or 'compute'"
        exit 1
        ;;
esac
