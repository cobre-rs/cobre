//! Execution topology types for cobre-comm.
//!
//! This module defines [`ExecutionTopology`] and its component structs, which
//! describe the process layout, communication backend metadata, and optional
//! scheduler information for a running Cobre job.
//!
//! Topology is gathered once during backend initialization and cached. All
//! subsequent queries are non-collective and allocation-free.

use crate::factory::BackendKind;

/// Execution topology gathered at communicator initialization.
///
/// Describes the process layout, communication backend metadata, and optional
/// scheduler information. Built once during backend creation and queryable
/// thereafter (no further collectives needed).
#[derive(Debug, Clone)]
pub struct ExecutionTopology {
    /// Which backend is active.
    pub backend: BackendKind,
    /// Total number of processes.
    pub world_size: usize,
    /// Per-host rank assignments, ordered by first rank on each host.
    pub hosts: Vec<HostInfo>,
    /// MPI runtime metadata (None for non-MPI backends).
    pub mpi: Option<MpiRuntimeInfo>,
    /// SLURM job metadata (None when not under SLURM or feature not enabled).
    pub slurm: Option<SlurmJobInfo>,
}

impl ExecutionTopology {
    /// Number of distinct hosts.
    #[must_use]
    pub fn num_hosts(&self) -> usize {
        self.hosts.len()
    }

    /// Returns `true` if all hosts have the same number of ranks.
    ///
    /// Returns `true` for empty host lists (vacuously homogeneous) and for
    /// single-host deployments.
    #[must_use]
    pub fn is_homogeneous(&self) -> bool {
        let mut iter = self.hosts.iter().map(|h| h.ranks.len());
        match iter.next() {
            None => true,
            Some(first) => iter.all(|n| n == first),
        }
    }

    /// Hostname of the first (or only) host.
    ///
    /// Useful for local/single-node display. Returns `"unknown"` if the host
    /// list is empty.
    #[must_use]
    pub fn leader_hostname(&self) -> &str {
        self.hosts
            .first()
            .map_or("unknown", |h| h.hostname.as_str())
    }
}

/// A single host and its assigned ranks.
#[derive(Debug, Clone)]
pub struct HostInfo {
    /// Hostname as reported by the backend (`MPI_Get_processor_name` or OS).
    pub hostname: String,
    /// Sorted list of global ranks on this host.
    pub ranks: Vec<usize>,
}

/// MPI runtime metadata.
#[derive(Debug, Clone)]
pub struct MpiRuntimeInfo {
    /// Implementation version, e.g. `"Open MPI v4.1.6"`.
    pub library_version: String,
    /// Standard version, e.g. `"MPI 4.0"`.
    pub standard_version: String,
    /// Negotiated thread safety level, e.g. `"Funneled"`.
    pub thread_level: String,
}

/// SLURM job metadata.
#[derive(Debug, Clone)]
pub struct SlurmJobInfo {
    /// `SLURM_JOB_ID`.
    pub job_id: String,
    /// Compact node list, e.g. `"compute-[01-04]"`.
    pub node_list: Option<String>,
    /// CPUs allocated per task.
    pub cpus_per_task: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::{ExecutionTopology, HostInfo, SlurmJobInfo};
    use crate::BackendKind;

    fn make_topology(host_rank_counts: &[usize]) -> ExecutionTopology {
        let hosts = host_rank_counts
            .iter()
            .enumerate()
            .map(|(i, &n)| HostInfo {
                hostname: format!("host-{i:02}"),
                ranks: (0..n).collect(),
            })
            .collect();
        ExecutionTopology {
            backend: BackendKind::Local,
            world_size: host_rank_counts.iter().sum(),
            hosts,
            mpi: None,
            slurm: None,
        }
    }

    #[test]
    fn test_num_hosts_empty() {
        let topo = make_topology(&[]);
        assert_eq!(topo.num_hosts(), 0);
    }

    #[test]
    fn test_num_hosts_single() {
        let topo = make_topology(&[4]);
        assert_eq!(topo.num_hosts(), 1);
    }

    #[test]
    fn test_num_hosts_multiple() {
        let topo = make_topology(&[4, 4]);
        assert_eq!(topo.num_hosts(), 2);
    }

    #[test]
    fn test_is_homogeneous_empty() {
        let topo = make_topology(&[]);
        assert!(topo.is_homogeneous());
    }

    #[test]
    fn test_is_homogeneous_single_host() {
        let topo = make_topology(&[4]);
        assert!(topo.is_homogeneous());
    }

    #[test]
    fn test_is_homogeneous_equal_rank_counts() {
        let topo = make_topology(&[4, 4]);
        assert!(topo.is_homogeneous());
    }

    #[test]
    fn test_is_homogeneous_unequal_rank_counts() {
        let topo = make_topology(&[4, 3]);
        assert!(!topo.is_homogeneous());
    }

    #[test]
    fn test_leader_hostname_empty() {
        let topo = make_topology(&[]);
        assert_eq!(topo.leader_hostname(), "unknown");
    }

    #[test]
    fn test_leader_hostname_returns_first() {
        let topo = make_topology(&[4, 4]);
        assert_eq!(topo.leader_hostname(), "host-00");
    }

    #[test]
    fn test_debug_clone_derive() {
        let topo = make_topology(&[2, 2]);
        let cloned = topo.clone();
        assert_eq!(cloned.num_hosts(), 2);
        let debug_str = format!("{topo:?}");
        assert!(debug_str.contains("ExecutionTopology"));
    }

    #[test]
    fn test_slurm_job_info_fields() {
        let info = SlurmJobInfo {
            job_id: "123456".to_string(),
            node_list: Some("compute-[01-04]".to_string()),
            cpus_per_task: Some(8),
        };
        assert_eq!(info.job_id, "123456");
        assert_eq!(info.node_list.as_deref(), Some("compute-[01-04]"));
        assert_eq!(info.cpus_per_task, Some(8));
    }
}
