# tests/slurm/Dockerfile
# 2-node SLURM + MPICH 4.2.3 cluster image for MPI integration testing.
#
# Uses the pre-built base image from GHCR that contains Ubuntu 24.04,
# SLURM, MPICH 4.2.3, and pyarrow. This Dockerfile only adds the
# cluster-specific config files and entrypoint.
#
# The munge key is NOT generated at build time — it is generated at runtime
# by the controller container and shared via the 'slurm-shared' volume.
#
# Base image rebuild: .github/workflows/slurm-base-image.yml
# Build:  docker compose -f tests/slurm/docker-compose.yml build
# Usage:  docker compose -f tests/slurm/docker-compose.yml up -d --wait

ARG BASE_IMAGE=ghcr.io/cobre-rs/slurm-mpich-base:4.2.3
FROM ${BASE_IMAGE}

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
