//! CPU topology discovery and opt-in worker affinity.
//!
//! The planning layer is platform-neutral and deterministic: callers can test
//! arbitrary, sparse CPU identifiers without touching host affinity. Linux
//! discovery and binding are compiled behind the `affinity` feature. Other
//! builds retain the same API and report [`AffinityError::Unsupported`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};

/// Return the operating-system CPUs on which the current thread may execute.
///
/// # Errors
///
/// Returns [`AffinityError::Unsupported`] unless built for Linux with the
/// `affinity` feature, or a native error when the OS query fails.
pub fn current_thread_cpu_set() -> Result<Vec<usize>, AffinityError> {
    platform::current_cpu_set()
}

/// Worker-to-CPU binding policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AffinityPolicy {
    /// Preserve operating-system or launcher scheduling.
    #[default]
    None,
    /// Fill physical cores before assigning workers to SMT siblings.
    Core,
    /// Spread workers across NUMA nodes, filling physical cores before SMT.
    Numa,
}

impl AffinityPolicy {
    /// Stable lowercase policy name used by CLI/Python and metadata.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Core => "core",
            Self::Numa => "numa",
        }
    }

    /// Whether the policy requests an operating-system affinity change.
    #[must_use]
    pub const fn binds_workers(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl fmt::Display for AffinityPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AffinityPolicy {
    type Err = AffinityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "core" => Ok(Self::Core),
            "numa" => Ok(Self::Numa),
            _ => Err(AffinityError::InvalidPolicy {
                value: value.to_string(),
            }),
        }
    }
}

/// One logical processing unit visible to the process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingUnit {
    /// Operating-system CPU identifier used by affinity APIs.
    pub os_index: usize,
    /// Physical package/socket identifier.
    pub package_id: usize,
    /// Physical core identifier within the package.
    pub core_id: usize,
    /// NUMA node containing this processing unit, when known.
    pub numa_node: Option<usize>,
}

/// One NUMA node intersected with the process/rank CPU allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaNode {
    /// Operating-system NUMA-node identifier.
    pub id: usize,
    /// Visible CPUs belonging to the node, sorted by OS identifier.
    pub cpus: Vec<usize>,
    /// Node memory capacity reported by the operating system, when available.
    pub memory_bytes: Option<u64>,
}

/// Memory policy and node allocation visible to the current process/thread.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryBinding {
    /// Active Linux memory policy name, when readable.
    pub policy: Option<String>,
    /// NUMA nodes selected by the active memory policy.
    pub policy_nodes: Vec<usize>,
    /// NUMA nodes allowed by the process cpuset.
    pub allowed_nodes: Vec<usize>,
    /// Non-fatal native discovery failure.
    pub discovery_error: Option<String>,
}

/// CPU resources visible to the current process or MPI rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuTopology {
    processing_units: Vec<ProcessingUnit>,
    numa_nodes: Vec<NumaNode>,
    online_processing_units: usize,
    memory_binding: MemoryBinding,
}

impl CpuTopology {
    /// Build and normalize a topology obtained from a synthetic or native source.
    ///
    /// Input enumeration order is discarded. CPU and NUMA identifiers must be
    /// unique, and every NUMA CPU must exist in `processing_units`.
    ///
    /// # Errors
    ///
    /// Returns [`AffinityError::InvalidTopology`] for an empty topology,
    /// duplicate identifiers, inconsistent NUMA membership, or an online CPU
    /// count smaller than the visible allocation.
    pub fn new(
        mut processing_units: Vec<ProcessingUnit>,
        mut numa_nodes: Vec<NumaNode>,
        online_processing_units: usize,
    ) -> Result<Self, AffinityError> {
        if processing_units.is_empty() {
            return Err(AffinityError::InvalidTopology {
                message: "the visible CPU set is empty".to_string(),
            });
        }

        processing_units.sort_unstable_by_key(|unit| unit.os_index);
        if processing_units
            .windows(2)
            .any(|pair| pair[0].os_index == pair[1].os_index)
        {
            return Err(AffinityError::InvalidTopology {
                message: "duplicate processing-unit OS identifier".to_string(),
            });
        }
        if online_processing_units < processing_units.len() {
            return Err(AffinityError::InvalidTopology {
                message: format!(
                    "online CPU count {online_processing_units} is smaller than visible CPU count {}",
                    processing_units.len()
                ),
            });
        }

        let visible: BTreeSet<usize> = processing_units.iter().map(|unit| unit.os_index).collect();
        numa_nodes.sort_unstable_by_key(|node| node.id);
        if numa_nodes.windows(2).any(|pair| pair[0].id == pair[1].id) {
            return Err(AffinityError::InvalidTopology {
                message: "duplicate NUMA-node identifier".to_string(),
            });
        }
        let mut cpu_nodes = BTreeMap::new();
        for node in &mut numa_nodes {
            node.cpus.sort_unstable();
            node.cpus.dedup();
            if let Some(cpu) = node.cpus.iter().find(|cpu| !visible.contains(cpu)) {
                return Err(AffinityError::InvalidTopology {
                    message: format!("NUMA node {} contains invisible CPU {cpu}", node.id),
                });
            }
            for &cpu in &node.cpus {
                if let Some(previous) = cpu_nodes.insert(cpu, node.id) {
                    return Err(AffinityError::InvalidTopology {
                        message: format!(
                            "CPU {cpu} belongs to both NUMA nodes {previous} and {}",
                            node.id
                        ),
                    });
                }
            }
        }
        for unit in &processing_units {
            if let Some(expected) = unit.numa_node {
                match cpu_nodes.get(&unit.os_index) {
                    Some(&actual) if actual == expected => {}
                    Some(&actual) => {
                        return Err(AffinityError::InvalidTopology {
                            message: format!(
                                "CPU {} reports NUMA node {expected} but belongs to node {actual}",
                                unit.os_index
                            ),
                        });
                    }
                    None => {
                        return Err(AffinityError::InvalidTopology {
                            message: format!(
                                "CPU {} reports NUMA node {expected} but is absent from that node",
                                unit.os_index
                            ),
                        });
                    }
                }
            }
        }

        Ok(Self {
            processing_units,
            numa_nodes,
            online_processing_units,
            memory_binding: MemoryBinding::default(),
        })
    }

    /// Discover the CPU topology visible to the current process/rank.
    ///
    /// # Errors
    ///
    /// Returns [`AffinityError::Unsupported`] unless built for Linux with the
    /// `affinity` feature. Native discovery errors retain their path/context.
    pub fn discover() -> Result<Self, AffinityError> {
        platform::discover()
    }

    /// Visible processing units in ascending operating-system order.
    #[must_use]
    pub fn processing_units(&self) -> &[ProcessingUnit] {
        &self.processing_units
    }

    /// Visible NUMA nodes in ascending operating-system order.
    #[must_use]
    pub fn numa_nodes(&self) -> &[NumaNode] {
        &self.numa_nodes
    }

    /// Current memory policy and allowed NUMA-node set.
    #[must_use]
    pub const fn memory_binding(&self) -> &MemoryBinding {
        &self.memory_binding
    }

    /// Number of logical CPUs online on the host, including CPUs outside the
    /// process/rank allocation.
    #[must_use]
    pub const fn online_processing_units(&self) -> usize {
        self.online_processing_units
    }

    /// Number of logical CPUs visible to the current process/rank.
    #[must_use]
    pub fn visible_processing_units(&self) -> usize {
        self.processing_units.len()
    }

    /// Number of visible physical cores, deduplicating SMT siblings.
    #[must_use]
    pub fn physical_cores(&self) -> usize {
        self.processing_units
            .iter()
            .map(|unit| (unit.package_id, unit.core_id))
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Visible OS CPU identifiers in ascending order.
    #[must_use]
    pub fn visible_cpus(&self) -> Vec<usize> {
        self.processing_units
            .iter()
            .map(|unit| unit.os_index)
            .collect()
    }
}

/// Resolved topology and worker placement persisted by callers as operational metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinityReport {
    /// Requested binding policy.
    pub policy: AffinityPolicy,
    /// Logical CPUs online on the host, if discovery succeeded.
    pub online_processing_units: Option<usize>,
    /// Logical CPUs visible to the process/rank, if discovery succeeded.
    pub visible_processing_units: Option<usize>,
    /// Visible physical cores, if discovery succeeded.
    pub physical_cores: Option<usize>,
    /// Visible NUMA-node count, if discovery succeeded.
    pub numa_nodes: Option<usize>,
    /// Visible OS CPU identifiers.
    pub visible_cpus: Vec<usize>,
    /// Active memory policy, when readable.
    pub memory_policy: Option<String>,
    /// NUMA nodes selected by the active memory policy.
    pub memory_policy_nodes: Vec<usize>,
    /// NUMA nodes allowed for memory allocation.
    pub allowed_memory_nodes: Vec<usize>,
    /// Non-fatal memory-binding discovery failure.
    pub memory_discovery_error: Option<String>,
    /// Resolved worker-index to OS-CPU mapping. Empty for `none`.
    pub worker_cpus: Vec<usize>,
    /// Discovery failure retained for diagnostics when policy is `none`.
    pub discovery_error: Option<String>,
}

impl AffinityReport {
    /// Whether workers outnumber the visible logical CPUs.
    #[must_use]
    pub fn is_oversubscribed(&self, workers: usize) -> bool {
        self.visible_processing_units
            .is_some_and(|visible| workers > visible)
    }

    /// Whether the inherited CPU set is a strict subset of online host CPUs.
    #[must_use]
    pub fn is_cpuset_restricted(&self) -> bool {
        matches!(
            (self.visible_processing_units, self.online_processing_units),
            (Some(visible), Some(online)) if visible < online
        )
    }
}

/// Prepared worker affinity shared by Rayon pool construction paths.
///
/// Clone this value into a pool's worker-start hook and call
/// [`WorkerAffinity::bind_worker`] with Rayon's stable worker index. After pool
/// construction, [`WorkerAffinity::verify`] surfaces the first binding failure.
#[derive(Debug, Clone)]
pub struct WorkerAffinity {
    report: AffinityReport,
    worker_cpus: Arc<Vec<usize>>,
    first_failure: Arc<OnceLock<String>>,
}

impl WorkerAffinity {
    /// Discover topology and resolve a mapping for `workers`.
    ///
    /// Discovery failure is diagnostic-only for [`AffinityPolicy::None`]. An
    /// explicit binding policy fails rather than silently running unbound.
    ///
    /// # Errors
    ///
    /// Returns a topology, policy, or platform error for an explicit policy, or
    /// [`AffinityError::InvalidWorkerCount`] when `workers == 0`.
    pub fn prepare(policy: AffinityPolicy, workers: usize) -> Result<Self, AffinityError> {
        if workers == 0 {
            return Err(AffinityError::InvalidWorkerCount);
        }

        let topology = match CpuTopology::discover() {
            Ok(topology) => topology,
            Err(error) if !policy.binds_workers() => {
                return Ok(Self {
                    report: AffinityReport {
                        policy,
                        online_processing_units: None,
                        visible_processing_units: None,
                        physical_cores: None,
                        numa_nodes: None,
                        visible_cpus: Vec::new(),
                        memory_policy: None,
                        memory_policy_nodes: Vec::new(),
                        allowed_memory_nodes: Vec::new(),
                        memory_discovery_error: None,
                        worker_cpus: Vec::new(),
                        discovery_error: Some(error.to_string()),
                    },
                    worker_cpus: Arc::new(Vec::new()),
                    first_failure: Arc::new(OnceLock::new()),
                });
            }
            Err(error) => return Err(error),
        };

        let worker_cpus = if policy.binds_workers() {
            plan_worker_cpus(&topology, policy, workers)
        } else {
            Vec::new()
        };
        let report = AffinityReport {
            policy,
            online_processing_units: Some(topology.online_processing_units()),
            visible_processing_units: Some(topology.visible_processing_units()),
            physical_cores: Some(topology.physical_cores()),
            numa_nodes: Some(topology.numa_nodes().len()),
            visible_cpus: topology.visible_cpus(),
            memory_policy: topology.memory_binding().policy.clone(),
            memory_policy_nodes: topology.memory_binding().policy_nodes.clone(),
            allowed_memory_nodes: topology.memory_binding().allowed_nodes.clone(),
            memory_discovery_error: topology.memory_binding().discovery_error.clone(),
            worker_cpus: worker_cpus.clone(),
            discovery_error: None,
        };

        Ok(Self {
            report,
            worker_cpus: Arc::new(worker_cpus),
            first_failure: Arc::new(OnceLock::new()),
        })
    }

    /// Build a worker mapping from a supplied topology without native discovery.
    ///
    /// This constructor is intended for deterministic mapping tests and tools
    /// that already own authoritative topology data. It never binds by itself.
    ///
    /// # Errors
    ///
    /// Returns [`AffinityError::InvalidWorkerCount`] when `workers == 0`.
    pub fn from_topology(
        policy: AffinityPolicy,
        workers: usize,
        topology: &CpuTopology,
    ) -> Result<Self, AffinityError> {
        if workers == 0 {
            return Err(AffinityError::InvalidWorkerCount);
        }
        let worker_cpus = if policy.binds_workers() {
            plan_worker_cpus(topology, policy, workers)
        } else {
            Vec::new()
        };
        Ok(Self {
            report: AffinityReport {
                policy,
                online_processing_units: Some(topology.online_processing_units()),
                visible_processing_units: Some(topology.visible_processing_units()),
                physical_cores: Some(topology.physical_cores()),
                numa_nodes: Some(topology.numa_nodes().len()),
                visible_cpus: topology.visible_cpus(),
                memory_policy: topology.memory_binding().policy.clone(),
                memory_policy_nodes: topology.memory_binding().policy_nodes.clone(),
                allowed_memory_nodes: topology.memory_binding().allowed_nodes.clone(),
                memory_discovery_error: topology.memory_binding().discovery_error.clone(),
                worker_cpus: worker_cpus.clone(),
                discovery_error: None,
            },
            worker_cpus: Arc::new(worker_cpus),
            first_failure: Arc::new(OnceLock::new()),
        })
    }

    /// Bind the current thread for a Rayon worker index.
    ///
    /// Failures are retained and returned later by [`WorkerAffinity::verify`]
    /// because Rayon start hooks cannot return a `Result`.
    pub fn bind_worker(&self, worker_index: usize) {
        let Some(&cpu) = self.worker_cpus.get(worker_index) else {
            if self.report.policy.binds_workers() {
                let _ = self.first_failure.set(format!(
                    "rayon worker index {worker_index} has no resolved CPU mapping"
                ));
            }
            return;
        };

        if let Err(error) = platform::bind_current_thread(cpu) {
            let _ = self.first_failure.set(error.to_string());
        }
    }

    /// Surface a binding failure captured by a worker-start hook.
    ///
    /// # Errors
    ///
    /// Returns [`AffinityError::WorkerBindingFailed`] with the first captured
    /// failure. No error is returned for an unbound policy.
    pub fn verify(&self) -> Result<(), AffinityError> {
        match self.first_failure.get() {
            Some(message) => Err(AffinityError::WorkerBindingFailed {
                message: message.clone(),
            }),
            None => Ok(()),
        }
    }

    /// Resolved topology and mapping report.
    #[must_use]
    pub const fn report(&self) -> &AffinityReport {
        &self.report
    }
}

/// CPU-affinity errors.
#[derive(Debug, thiserror::Error)]
pub enum AffinityError {
    /// User-facing policy string is not one of the supported values.
    #[error("unknown CPU binding policy '{value}' (expected: none, core, numa)")]
    InvalidPolicy {
        /// Rejected policy value.
        value: String,
    },
    /// A worker pool cannot contain zero workers.
    #[error("CPU affinity requires at least one worker")]
    InvalidWorkerCount,
    /// Native discovery or binding is not available in this build/platform.
    #[error("CPU affinity is unsupported: {reason}")]
    Unsupported {
        /// Actionable platform/build explanation.
        reason: String,
    },
    /// Synthetic or native topology violates required invariants.
    #[error("invalid CPU topology: {message}")]
    InvalidTopology {
        /// Invariant violation.
        message: String,
    },
    /// A native topology file could not be read or parsed.
    #[error("failed to read CPU topology from '{}': {source}", path.display())]
    TopologyIo {
        /// Failing topology path.
        path: std::path::PathBuf,
        /// Underlying I/O or parse failure.
        source: std::io::Error,
    },
    /// Native affinity rejected an operating-system CPU identifier.
    #[error("CPU {cpu} exceeds the native affinity mask capacity")]
    CpuOutOfRange {
        /// Rejected OS CPU identifier.
        cpu: usize,
    },
    /// Operating-system binding failed.
    #[error("failed to bind current worker to CPU {cpu}: {source}")]
    BindFailed {
        /// Target OS CPU identifier.
        cpu: usize,
        /// Operating-system error.
        source: std::io::Error,
    },
    /// A worker-start hook captured an error that the hook could not return.
    #[error("worker CPU binding failed: {message}")]
    WorkerBindingFailed {
        /// First captured worker failure.
        message: String,
    },
}

fn plan_worker_cpus(topology: &CpuTopology, policy: AffinityPolicy, workers: usize) -> Vec<usize> {
    let order = match policy {
        AffinityPolicy::None => return Vec::new(),
        AffinityPolicy::Core => physical_first_order(topology.processing_units()),
        AffinityPolicy::Numa => numa_spread_order(topology.processing_units()),
    };

    order.iter().copied().cycle().take(workers).collect()
}

fn split_physical_and_smt(units: &[ProcessingUnit]) -> (Vec<usize>, Vec<usize>) {
    let mut cores: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for unit in units {
        cores
            .entry((unit.package_id, unit.core_id))
            .or_default()
            .push(unit.os_index);
    }
    for siblings in cores.values_mut() {
        siblings.sort_unstable();
    }

    let physical = cores.values().map(|siblings| siblings[0]).collect();
    let max_siblings = cores.values().map(Vec::len).max().unwrap_or(1);
    let mut smt = Vec::new();
    for sibling_index in 1..max_siblings {
        for siblings in cores.values() {
            if let Some(&cpu) = siblings.get(sibling_index) {
                smt.push(cpu);
            }
        }
    }
    (physical, smt)
}

fn physical_first_order(units: &[ProcessingUnit]) -> Vec<usize> {
    let (mut physical, smt) = split_physical_and_smt(units);
    physical.extend(smt);
    physical
}

fn numa_spread_order(units: &[ProcessingUnit]) -> Vec<usize> {
    let mut per_node: BTreeMap<Option<usize>, Vec<ProcessingUnit>> = BTreeMap::new();
    for unit in units {
        per_node
            .entry(unit.numa_node)
            .or_default()
            .push(unit.clone());
    }

    let mut physical_by_node = Vec::with_capacity(per_node.len());
    let mut smt_by_node = Vec::with_capacity(per_node.len());
    for node_units in per_node.values() {
        let (physical, smt) = split_physical_and_smt(node_units);
        physical_by_node.push(physical);
        smt_by_node.push(smt);
    }

    let mut order = round_robin(&physical_by_node);
    order.extend(round_robin(&smt_by_node));
    order
}

fn round_robin(groups: &[Vec<usize>]) -> Vec<usize> {
    let max_len = groups.iter().map(Vec::len).max().unwrap_or(0);
    let mut result = Vec::new();
    for index in 0..max_len {
        for group in groups {
            if let Some(&cpu) = group.get(index) {
                result.push(cpu);
            }
        }
    }
    result
}

#[cfg(all(feature = "affinity", target_os = "linux"))]
mod platform {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::io;
    use std::mem::{size_of, zeroed};
    use std::path::{Path, PathBuf};

    use super::{AffinityError, CpuTopology, MemoryBinding, NumaNode, ProcessingUnit};

    const SYS_CPU_ROOT: &str = "/sys/devices/system/cpu";
    const SYS_NODE_ROOT: &str = "/sys/devices/system/node";

    pub(super) fn discover() -> Result<CpuTopology, AffinityError> {
        let allowed = current_cpu_set()?;
        let online = read_cpu_list(&Path::new(SYS_CPU_ROOT).join("online"))?.len();
        discover_from_sysfs(
            Path::new(SYS_CPU_ROOT),
            Path::new(SYS_NODE_ROOT),
            &allowed,
            online,
        )
    }

    pub(super) fn bind_current_thread(cpu: usize) -> Result<(), AffinityError> {
        if cpu >= libc::CPU_SETSIZE as usize {
            return Err(AffinityError::CpuOutOfRange { cpu });
        }

        // SAFETY: cpu_set_t is a plain C bitset. CPU_ZERO/CPU_SET receive a
        // valid mutable reference and `cpu` was checked against CPU_SETSIZE.
        let mut set: libc::cpu_set_t = unsafe { zeroed() };
        unsafe {
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
        }
        // SAFETY: pid 0 means the calling thread on Linux; `set` points to a
        // fully initialized cpu_set_t of the supplied size for the call only.
        let rc =
            unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &raw const set) };
        if rc == 0 {
            Ok(())
        } else {
            Err(AffinityError::BindFailed {
                cpu,
                source: io::Error::last_os_error(),
            })
        }
    }

    pub(super) fn current_cpu_set() -> Result<Vec<usize>, AffinityError> {
        // SAFETY: cpu_set_t is a plain C bitset initialized before the syscall.
        let mut set: libc::cpu_set_t = unsafe { zeroed() };
        // SAFETY: pid 0 selects the calling thread; the buffer is valid for the
        // exact cpu_set_t size and remains alive until the call returns.
        let rc = unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &raw mut set) };
        if rc != 0 {
            return Err(AffinityError::TopologyIo {
                path: PathBuf::from("sched_getaffinity"),
                source: io::Error::last_os_error(),
            });
        }

        let cpus = (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| {
                // SAFETY: `cpu` is within CPU_SETSIZE and `set` is initialized.
                unsafe { libc::CPU_ISSET(cpu, &set) }
            })
            .collect::<Vec<_>>();
        if cpus.is_empty() {
            return Err(AffinityError::InvalidTopology {
                message: "sched_getaffinity returned an empty CPU set".to_string(),
            });
        }
        Ok(cpus)
    }

    fn discover_from_sysfs(
        cpu_root: &Path,
        node_root: &Path,
        allowed: &[usize],
        online: usize,
    ) -> Result<CpuTopology, AffinityError> {
        let allowed_set: BTreeSet<usize> = allowed.iter().copied().collect();
        let mut nodes = read_numa_nodes(node_root, &allowed_set)?;
        if nodes.is_empty() {
            nodes.push(NumaNode {
                id: 0,
                cpus: allowed.to_vec(),
                memory_bytes: None,
            });
        }

        let cpu_nodes: BTreeMap<usize, usize> = nodes
            .iter()
            .flat_map(|node| node.cpus.iter().map(move |&cpu| (cpu, node.id)))
            .collect();
        let mut units = Vec::with_capacity(allowed.len());
        for &cpu in allowed {
            let topology_root = cpu_root.join(format!("cpu{cpu}/topology"));
            units.push(ProcessingUnit {
                os_index: cpu,
                package_id: read_usize(&topology_root.join("physical_package_id"))?,
                core_id: read_usize(&topology_root.join("core_id"))?,
                numa_node: cpu_nodes.get(&cpu).copied(),
            });
        }

        let mut topology = CpuTopology::new(units, nodes, online)?;
        topology.memory_binding = discover_memory_binding();
        Ok(topology)
    }

    fn discover_memory_binding() -> MemoryBinding {
        let mut errors = Vec::new();
        let allowed_nodes = match read_allowed_memory_nodes(Path::new("/proc/self/status")) {
            Ok(nodes) => nodes,
            Err(error) => {
                errors.push(error.to_string());
                Vec::new()
            }
        };
        let (policy, policy_nodes) = match current_memory_policy(&allowed_nodes) {
            Ok((policy, nodes)) => (Some(policy), nodes),
            Err(error) => {
                errors.push(format!("get_mempolicy failed: {error}"));
                (None, Vec::new())
            }
        };
        MemoryBinding {
            policy,
            policy_nodes,
            allowed_nodes,
            discovery_error: (!errors.is_empty()).then(|| errors.join("; ")),
        }
    }

    fn read_allowed_memory_nodes(path: &Path) -> Result<Vec<usize>, AffinityError> {
        let contents = fs::read_to_string(path).map_err(|source| AffinityError::TopologyIo {
            path: path.to_path_buf(),
            source,
        })?;
        let value = contents
            .lines()
            .find_map(|line| line.strip_prefix("Mems_allowed_list:"))
            .ok_or_else(|| AffinityError::TopologyIo {
                path: path.to_path_buf(),
                source: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "missing Mems_allowed_list field",
                ),
            })?;
        parse_cpu_list(value).map_err(|source| AffinityError::TopologyIo {
            path: path.to_path_buf(),
            source,
        })
    }

    fn current_memory_policy(allowed_nodes: &[usize]) -> Result<(String, Vec<usize>), io::Error> {
        let mut mode: libc::c_int = 0;
        let max_node = allowed_nodes.last().copied().unwrap_or(0) + 1;
        let word_bits = usize::BITS as usize;
        let mut node_mask = vec![0 as libc::c_ulong; max_node.div_ceil(word_bits)];
        // SAFETY: `mode` and `node_mask` are writable for the supplied bit count;
        // the optional address pointer is null for a calling-thread query.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_get_mempolicy,
                &raw mut mode,
                node_mask.as_mut_ptr(),
                max_node as libc::c_ulong,
                std::ptr::null_mut::<libc::c_void>(),
                0 as libc::c_ulong,
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        let base_mode = mode
            & !(libc::MPOL_F_NUMA_BALANCING
                | libc::MPOL_F_RELATIVE_NODES
                | libc::MPOL_F_STATIC_NODES);
        let policy = match base_mode {
            libc::MPOL_DEFAULT => "default".to_string(),
            libc::MPOL_PREFERRED => "preferred".to_string(),
            libc::MPOL_BIND => "bind".to_string(),
            libc::MPOL_INTERLEAVE => "interleave".to_string(),
            libc::MPOL_LOCAL => "local".to_string(),
            value => format!("unknown({value})"),
        };
        let policy_nodes = (0..max_node)
            .filter(|&node| node_mask[node / word_bits] & (1 << (node % word_bits)) != 0)
            .collect();
        Ok((policy, policy_nodes))
    }

    fn read_numa_nodes(
        node_root: &Path,
        allowed: &BTreeSet<usize>,
    ) -> Result<Vec<NumaNode>, AffinityError> {
        let entries = match fs::read_dir(node_root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(AffinityError::TopologyIo {
                    path: node_root.to_path_buf(),
                    source,
                });
            }
        };

        let mut nodes = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| AffinityError::TopologyIo {
                path: node_root.to_path_buf(),
                source,
            })?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Some(id) = name.strip_prefix("node").and_then(|id| id.parse().ok()) else {
                continue;
            };
            let path = entry.path();
            let cpus = read_cpu_list(&path.join("cpulist"))?
                .into_iter()
                .filter(|cpu| allowed.contains(cpu))
                .collect::<Vec<_>>();
            if cpus.is_empty() {
                continue;
            }
            nodes.push(NumaNode {
                id,
                cpus,
                memory_bytes: read_node_memory(&path.join("meminfo"))?,
            });
        }
        Ok(nodes)
    }

    fn read_node_memory(path: &Path) -> Result<Option<u64>, AffinityError> {
        let contents = match fs::read_to_string(path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(AffinityError::TopologyIo {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        for line in contents.lines() {
            if !line.contains("MemTotal") {
                continue;
            }
            let value = line
                .split_whitespace()
                .find_map(|word| word.parse::<u64>().ok());
            return value
                .and_then(|kb| kb.checked_mul(1024))
                .map(Some)
                .ok_or_else(|| AffinityError::TopologyIo {
                    path: path.to_path_buf(),
                    source: io::Error::new(io::ErrorKind::InvalidData, "invalid MemTotal line"),
                });
        }
        Ok(None)
    }

    fn read_usize(path: &Path) -> Result<usize, AffinityError> {
        let value = fs::read_to_string(path).map_err(|source| AffinityError::TopologyIo {
            path: path.to_path_buf(),
            source,
        })?;
        value
            .trim()
            .parse::<usize>()
            .map_err(|source| AffinityError::TopologyIo {
                path: path.to_path_buf(),
                source: io::Error::new(io::ErrorKind::InvalidData, source),
            })
    }

    fn read_cpu_list(path: &Path) -> Result<Vec<usize>, AffinityError> {
        let value = fs::read_to_string(path).map_err(|source| AffinityError::TopologyIo {
            path: path.to_path_buf(),
            source,
        })?;
        parse_cpu_list(&value).map_err(|source| AffinityError::TopologyIo {
            path: path.to_path_buf(),
            source,
        })
    }

    fn parse_cpu_list(value: &str) -> Result<Vec<usize>, io::Error> {
        let mut cpus = BTreeSet::new();
        for item in value.trim().split(',').filter(|item| !item.is_empty()) {
            let mut bounds = item.split('-');
            let start = bounds
                .next()
                .and_then(|part| part.parse::<usize>().ok())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid CPU list"))?;
            let end = match bounds.next() {
                Some(part) => part
                    .parse::<usize>()
                    .map_err(|source| io::Error::new(io::ErrorKind::InvalidData, source))?,
                None => start,
            };
            if bounds.next().is_some() || end < start {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid CPU range",
                ));
            }
            cpus.extend(start..=end);
        }
        if cpus.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty CPU list"));
        }
        Ok(cpus.into_iter().collect())
    }

    #[cfg(test)]
    mod tests {
        use super::parse_cpu_list;

        #[test]
        fn cpu_list_parser_handles_sparse_ranges() {
            assert_eq!(
                parse_cpu_list("0-3,8,10-11\n").unwrap(),
                vec![0, 1, 2, 3, 8, 10, 11]
            );
        }

        #[test]
        fn cpu_list_parser_rejects_descending_range() {
            assert!(parse_cpu_list("4-2").is_err());
        }
    }
}

#[cfg(not(all(feature = "affinity", target_os = "linux")))]
mod platform {
    use super::{AffinityError, CpuTopology};

    pub(super) fn discover() -> Result<CpuTopology, AffinityError> {
        Err(AffinityError::Unsupported {
            reason: "requires Linux and a build with the 'affinity' feature".to_string(),
        })
    }

    pub(super) fn bind_current_thread(cpu: usize) -> Result<(), AffinityError> {
        let _ = cpu;
        Err(AffinityError::Unsupported {
            reason: "requires Linux and a build with the 'affinity' feature".to_string(),
        })
    }

    pub(super) fn current_cpu_set() -> Result<Vec<usize>, AffinityError> {
        Err(AffinityError::Unsupported {
            reason: "requires Linux and a build with the 'affinity' feature".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{AffinityPolicy, CpuTopology, NumaNode, ProcessingUnit, WorkerAffinity};

    fn asymmetric_topology(reversed: bool) -> CpuTopology {
        let mut units = vec![
            ProcessingUnit {
                os_index: 8,
                package_id: 0,
                core_id: 0,
                numa_node: Some(0),
            },
            ProcessingUnit {
                os_index: 40,
                package_id: 0,
                core_id: 0,
                numa_node: Some(0),
            },
            ProcessingUnit {
                os_index: 2,
                package_id: 0,
                core_id: 1,
                numa_node: Some(0),
            },
            ProcessingUnit {
                os_index: 18,
                package_id: 1,
                core_id: 0,
                numa_node: Some(3),
            },
            ProcessingUnit {
                os_index: 50,
                package_id: 1,
                core_id: 0,
                numa_node: Some(3),
            },
            ProcessingUnit {
                os_index: 12,
                package_id: 1,
                core_id: 1,
                numa_node: Some(3),
            },
            ProcessingUnit {
                os_index: 30,
                package_id: 1,
                core_id: 2,
                numa_node: Some(3),
            },
        ];
        let mut nodes = vec![
            NumaNode {
                id: 0,
                cpus: vec![40, 8, 2],
                memory_bytes: Some(64),
            },
            NumaNode {
                id: 3,
                cpus: vec![50, 30, 18, 12],
                memory_bytes: Some(96),
            },
        ];
        if reversed {
            units.reverse();
            nodes.reverse();
        }
        CpuTopology::new(units, nodes, 64).unwrap()
    }

    #[test]
    fn topology_normalizes_sparse_enumeration() {
        let topology = asymmetric_topology(true);
        assert_eq!(topology.visible_cpus(), vec![2, 8, 12, 18, 30, 40, 50]);
        assert_eq!(topology.physical_cores(), 5);
        assert_eq!(
            topology
                .numa_nodes()
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![0, 3]
        );
    }

    #[test]
    fn core_policy_uses_every_physical_core_before_smt() {
        let topology = asymmetric_topology(false);
        let affinity = WorkerAffinity::from_topology(AffinityPolicy::Core, 7, &topology).unwrap();
        assert_eq!(
            affinity.report().worker_cpus,
            vec![8, 2, 18, 12, 30, 40, 50]
        );
    }

    #[test]
    fn numa_policy_spreads_asymmetric_nodes_before_smt() {
        let topology = asymmetric_topology(false);
        let affinity = WorkerAffinity::from_topology(AffinityPolicy::Numa, 7, &topology).unwrap();
        assert_eq!(
            affinity.report().worker_cpus,
            vec![8, 18, 2, 12, 30, 40, 50]
        );
    }

    #[test]
    fn mapping_is_independent_of_input_enumeration() {
        let ordered = asymmetric_topology(false);
        let reversed = asymmetric_topology(true);
        for policy in [AffinityPolicy::Core, AffinityPolicy::Numa] {
            let lhs = WorkerAffinity::from_topology(policy, 11, &ordered).unwrap();
            let rhs = WorkerAffinity::from_topology(policy, 11, &reversed).unwrap();
            assert_eq!(lhs.report().worker_cpus, rhs.report().worker_cpus);
        }
    }

    #[test]
    fn oversubscription_repeats_only_after_all_visible_cpus() {
        let topology = asymmetric_topology(false);
        let affinity = WorkerAffinity::from_topology(AffinityPolicy::Core, 9, &topology).unwrap();
        assert_eq!(
            &affinity.report().worker_cpus[..7],
            &[8, 2, 18, 12, 30, 40, 50]
        );
        assert_eq!(&affinity.report().worker_cpus[7..], &[8, 2]);
        assert!(affinity.report().is_oversubscribed(9));
    }

    #[test]
    fn none_policy_reports_topology_without_mapping() {
        let topology = asymmetric_topology(false);
        let affinity = WorkerAffinity::from_topology(AffinityPolicy::None, 4, &topology).unwrap();
        assert!(affinity.report().worker_cpus.is_empty());
        assert_eq!(affinity.report().visible_processing_units, Some(7));
        assert!(affinity.report().is_cpuset_restricted());
    }

    #[test]
    fn topology_rejects_cpu_assigned_to_multiple_numa_nodes() {
        let units = vec![ProcessingUnit {
            os_index: 2,
            package_id: 0,
            core_id: 0,
            numa_node: Some(0),
        }];
        let nodes = vec![
            NumaNode {
                id: 0,
                cpus: vec![2],
                memory_bytes: None,
            },
            NumaNode {
                id: 1,
                cpus: vec![2],
                memory_bytes: None,
            },
        ];

        assert!(CpuTopology::new(units, nodes, 1).is_err());
    }

    #[test]
    fn topology_rejects_processing_unit_numa_mismatch() {
        let units = vec![ProcessingUnit {
            os_index: 2,
            package_id: 0,
            core_id: 0,
            numa_node: Some(1),
        }];
        let nodes = vec![NumaNode {
            id: 0,
            cpus: vec![2],
            memory_bytes: None,
        }];

        assert!(CpuTopology::new(units, nodes, 1).is_err());
    }

    #[cfg(not(all(feature = "affinity", target_os = "linux")))]
    #[test]
    fn unavailable_discovery_is_diagnostic_for_none_and_fatal_for_binding() {
        let unbound = WorkerAffinity::prepare(AffinityPolicy::None, 1).unwrap();
        assert!(unbound.report().discovery_error.is_some());
        assert!(WorkerAffinity::prepare(AffinityPolicy::Core, 1).is_err());
    }
}
