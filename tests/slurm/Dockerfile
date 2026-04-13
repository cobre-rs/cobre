# tests/slurm/Dockerfile
# 2-node SLURM + MPICH 4.2.3 cluster image for MPI integration testing.
#
# The munge key is NOT generated at build time — it is generated at runtime
# by the controller container and shared via the 'slurm-shared' volume.
#
# Build:  docker compose -f tests/slurm/docker-compose.yml build
# Usage:  docker compose -f tests/slurm/docker-compose.yml up -d --wait

FROM ubuntu:24.04

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # SLURM and munge authentication
    slurm-wlm \
    slurmctld \
    slurmd \
    munge \
    libmunge-dev \
    # Build toolchain for MPICH from source
    build-essential \
    wget \
    # dbus is required by slurmd cgroup/v2 plugin in SLURM 23.11+
    dbus \
    # Runtime utilities
    curl \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# pyarrow (needed by run-tests.sh for parquet comparison in ticket-004)
# ---------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir --break-system-packages pyarrow

# ---------------------------------------------------------------------------
# Build MPICH 4.2.3 from source with ch4:ofi device
# ch4:ofi falls back to TCP inside Docker containers; ch3:nemesis is for
# compile-time headers only (used in release-mpi.yml) and cannot run MPI.
# ---------------------------------------------------------------------------
ARG MPICH_VERSION=4.2.3
ARG MPICH_URL=https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz

RUN wget -q "${MPICH_URL}" -O /tmp/mpich.tar.gz \
    && tar -xzf /tmp/mpich.tar.gz -C /tmp \
    && cd /tmp/mpich-${MPICH_VERSION} \
    && ./configure \
        --prefix=/opt/mpich \
        --disable-fortran \
        --disable-cxx \
        --disable-romio \
        --disable-doc \
        --with-device=ch4:ofi \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/mpich.tar.gz /tmp/mpich-${MPICH_VERSION}

ENV PATH="/opt/mpich/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/mpich/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# SLURM runtime directories
# ---------------------------------------------------------------------------
RUN mkdir -p \
    /var/spool/slurm/ctld \
    /var/spool/slurm/d \
    /var/log/slurm \
    /var/run/slurm \
    /var/run/munge \
    /shared

RUN chown -R slurm:slurm \
    /var/spool/slurm \
    /var/log/slurm \
    /var/run/slurm

# ---------------------------------------------------------------------------
# SLURM configuration
# (munge key is intentionally NOT generated here — see entrypoint.sh)
# ---------------------------------------------------------------------------
COPY slurm.conf   /etc/slurm/slurm.conf
COPY cgroup.conf  /etc/slurm/cgroup.conf

RUN chown slurm:slurm /etc/slurm/slurm.conf /etc/slurm/cgroup.conf \
    && chmod 644 /etc/slurm/slurm.conf /etc/slurm/cgroup.conf

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ---------------------------------------------------------------------------
# Health check — SLURM must be running and this node must be registered idle
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=6 \
    CMD sinfo -N | grep idle && test -f /tmp/container-healthy

ENTRYPOINT ["/entrypoint.sh"]
