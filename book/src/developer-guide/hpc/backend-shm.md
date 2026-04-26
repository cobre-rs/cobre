# Shared Memory Backend

> **Status: Not Implemented.** This specification describes planned behavior for a future backend. No implementation exists in the current codebase.

## Purpose

The shared memory (shm) backend enables multi-process SDDP execution on a single node using POSIX shared memory primitives (`shm_open`/`mmap`), without requiring an MPI runtime. It wraps OS-level shared memory behind the `Communicator` and `SharedMemoryProvider` traits defined in [Communicator Trait §1](./communicator-trait.md) and [Communicator Trait §4](./communicator-trait.md), providing both collective communication (via shared buffers with atomic synchronization) and true shared memory regions (via POSIX shm) for the `SharedMemoryProvider` trait. Unlike the local and TCP backends, which use `HeapFallback` for `SharedMemoryProvider` ([Communicator Trait §4.4](./communicator-trait.md)), the shm backend provides genuine inter-process shared memory -- enabling the memory-saving benefits of `SharedWindow<T>` semantics (scenario storage, cut pool sharing, input case data) described in [Shared Memory Aggregation §1](./shared-memory-aggregation.md). The backend is gated behind the `shm` Cargo feature flag as specified in [Backend Registration and Selection §1.2](./backend-selection.md), and targets single-node deployments where multiple processes exploit NUMA locality but MPI installation is unavailable -- most notably Python multi-process execution on workstations and single-node containers.

## 1. Architecture

### 1.1 Process Model

The shm backend coordinates multiple independent OS processes on the same physical node. Each process has its own address space but shares memory regions via POSIX shared memory mappings. This is analogous to MPI ranks within a single node but without the MPI runtime -- rank identification and synchronization are handled through shared memory primitives and environment variables.

```
+------------------------------------------------------------------+
|                    Physical Node                                  |
|                                                                   |
|  +------------+   +------------+        +------------+            |
|  | Process 0  |   | Process 1  |  ...   | Process R-1|            |
|  | (rank 0)   |   | (rank 1)   |        | (rank R-1) |            |
|  |            |   |            |        |            |            |
|  | Private    |   | Private    |        | Private    |            |
|  | address    |   | address    |        | address    |            |
|  | space      |   | space      |        | space      |            |
|  +-----+------+   +-----+------+        +-----+------+            |
|        |                |                      |                  |
|        +--------+-------+-------+--------------+                  |
|                 |               |                                  |
|        +--------v--------+  +---v-----------------+               |
|        | Control Region  |  | Data Region(s)      |               |
|        | (small, fixed)  |  | (collective buffers, |               |
|        |                 |  |  shared regions)     |               |
|        | - atomic rank   |  |                      |               |
|        |   counter       |  | - allgatherv buffer  |               |
|        | - barrier state |  | - allreduce slots    |               |
|        | - ready flag    |  | - broadcast buffer   |               |
|        | - generation    |  | - SharedRegion<T>    |               |
|        |   counter       |  |   allocations        |               |
|        +-----------------+  +----------------------+               |
|                                                                   |
|          mmap'd via shm_open (POSIX shared memory)                |
+------------------------------------------------------------------+
```

**Key properties:**

- Each process is an independent OS process (not threads) with its own address space, matching the isolation model of MPI ranks.
- Shared memory regions are created via `shm_open` + `ftruncate` + `mmap`, visible to all processes on the same node that open the same named segment.
- A **control region** (small, fixed-size shared buffer) holds synchronization primitives: atomic counters for barrier coordination, a rank registration counter, a ready flag, and generation counters for collective sequencing.
- **Data regions** hold collective communication buffers (allgatherv, allreduce, broadcast) and `SharedRegion<T>` allocations for the `SharedMemoryProvider` trait.

### 1.2 Struct Definition

```rust
/// Shared-memory communication backend for single-node multi-process SDDP.
///
/// Uses POSIX shared memory (`shm_open`/`mmap`) for both collective
/// communication and true shared memory regions. All processes must
/// reside on the same physical node. The backend provides the same
/// observable behavior as the ferrompi backend for single-node
/// deployments, without requiring MPI installation.
#[cfg(feature = "shm")]
pub struct ShmBackend {
    /// This process's rank index (0..size).
    rank: usize,

    /// Total number of ranks sharing the memory segment.
    size: usize,

    /// Name of the POSIX shared memory segment (e.g., "/cobre_comm_abc123").
    shm_name: String,

    /// Memory-mapped pointer to the control region.
    /// Contains atomic counters, barrier state, and ready flag.
    control: *mut ControlRegion,

    /// Memory-mapped pointer to the data region.
    /// Contains collective buffers sized for the maximum expected payload.
    data: *mut u8,

    /// Total size of the data region in bytes.
    data_size: usize,

    /// File descriptor for the shared memory segment.
    shm_fd: i32,

    /// Total mapped size (control + data) for munmap on drop.
    mapped_size: usize,
}

/// SAFETY: ShmBackend is Send + Sync because:
/// - The shared memory pointers are valid for the lifetime of the mapping.
/// - All concurrent access is mediated by atomic operations in the control
///   region (no unsynchronized mutable aliasing).
/// - The Communicator trait requires Send + Sync for hybrid execution.
#[cfg(feature = "shm")]
unsafe impl Send for ShmBackend {}
#[cfg(feature = "shm")]
unsafe impl Sync for ShmBackend {}
```

### 1.3 Control Region Layout

The control region is a fixed-size structure at the beginning of the shared memory segment. All fields use atomic types to ensure safe concurrent access from multiple processes.

```rust
/// Fixed-size control region at the start of the shared memory segment.
///
/// All fields are atomically accessed. The layout is stable across
/// compilations (repr(C)) to ensure all processes interpret the
/// shared memory identically.
#[repr(C)]
struct ControlRegion {
    /// Number of ranks that have registered (atomic increment during startup).
    /// Reaches `size` when all ranks have attached.
    rank_counter: AtomicU32,

    /// Total expected rank count (written by rank 0 during creation).
    expected_size: AtomicU32,

    /// Ready flag: set to 1 by rank 0 after all ranks have registered.
    /// Other ranks spin on this flag during startup.
    ready: AtomicU32,

    /// Barrier generation counter. Incremented each time a barrier completes.
    /// Used to distinguish consecutive barriers and prevent early wake.
    barrier_generation: AtomicU64,

    /// Barrier arrival counter. Each rank increments this on barrier entry.
    /// Reset to 0 when the barrier completes.
    barrier_count: AtomicU32,

    /// Collective sequence number. Incremented by rank 0 after each
    /// collective operation completes. Used to sequence collectives
    /// and prevent data races between consecutive operations.
    collective_seq: AtomicU64,

    /// Padding to cache line boundary (avoid false sharing with data region).
    _padding: [u8; 128],
}
```

### 1.4 Synchronization Primitives

The shm backend uses two synchronization mechanisms, selected based on platform availability:

| Mechanism                | Platform | System Call                                   | Use Case                                             |
| ------------------------ | -------- | --------------------------------------------- | ---------------------------------------------------- |
| Atomic counter + `futex` | Linux    | `futex(FUTEX_WAIT)` / `futex(FUTEX_WAKE_ALL)` | Barrier, collective synchronization (lowest latency) |
| Atomic counter + condvar | macOS    | `pthread_cond_broadcast` in shared memory     | Barrier, collective synchronization (portable)       |

On Linux, `futex` operates directly on the atomic counter in shared memory (`FUTEX_WAIT` blocks until the counter changes; `FUTEX_WAKE_ALL` wakes all waiters). This avoids the need for a `pthread_mutex_t` or `pthread_cond_t` in shared memory, which requires `PTHREAD_PROCESS_SHARED` attribute setup.

On macOS, `PTHREAD_PROCESS_SHARED` mutexes and condition variables placed in the shared memory segment provide the equivalent functionality. The control region includes space for a `pthread_mutex_t` and `pthread_cond_t` (initialized by rank 0 with the `PTHREAD_PROCESS_SHARED` attribute).

**Barrier algorithm** (used by all collective operations as well as the explicit `barrier()` call):

1. Each rank atomically increments `barrier_count`.
2. If `barrier_count` reaches $R$ (all ranks have arrived):
   - The arriving rank resets `barrier_count` to 0.
   - The arriving rank increments `barrier_generation`.
   - The arriving rank wakes all waiters (`futex(FUTEX_WAKE_ALL)` on Linux, `pthread_cond_broadcast` on macOS).
3. If `barrier_count` has not reached $R$:
   - The rank waits on the `barrier_generation` counter (`futex(FUTEX_WAIT)` on Linux, `pthread_cond_wait` on macOS) until the generation advances.

The generation counter prevents the ABA problem: a rank cannot mistake a completed barrier for the current one, because each barrier completion increments the generation.

## 2. Process Coordination

### 2.1 Startup Sequence

Process coordination uses a named shared memory segment that all ranks discover via the `COBRE_SHM_NAME` environment variable. Rank 0 creates the segment; other ranks open it by name and register their presence.

**Rank 0 (creator):**

1. Call `shm_open(COBRE_SHM_NAME, O_CREAT | O_EXCL | O_RDWR, 0o600)` to create the named shared memory segment. The `O_EXCL` flag ensures that rank 0 creates a fresh segment (avoids stale segments from previous runs).
2. Call `ftruncate(fd, total_size)` to set the segment size to `sizeof(ControlRegion) + data_region_size`. The `data_region_size` is derived from `COBRE_SHM_SIZE` and the collective buffer sizing (see SS3).
3. Call `mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)` to map the segment into the process address space.
4. Initialize the `ControlRegion`: set `expected_size` to `COBRE_SHM_SIZE`, set `rank_counter` to 1 (rank 0 is implicitly registered), set `ready` to 0, initialize all other fields to 0. On macOS, initialize the `pthread_mutex_t` and `pthread_cond_t` with `PTHREAD_PROCESS_SHARED` attribute.
5. Wait until `rank_counter` reaches `expected_size` (all ranks have registered). Use a spin-wait with exponential backoff, or `futex`/condvar wait on `rank_counter`.
6. Set `ready` to 1 and wake all waiters. All ranks are now synchronized and may proceed to collective operations.

**Ranks 1 through $R-1$ (openers):**

1. Call `shm_open(COBRE_SHM_NAME, O_RDWR, 0)` to open the existing named shared memory segment. Retry with exponential backoff if the segment does not yet exist (rank 0 may not have created it yet).
2. Call `mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)` to map the segment. The `total_size` is computed identically to rank 0 (all ranks use the same `COBRE_SHM_SIZE` and buffer sizing formula).
3. Atomically increment `rank_counter` to register this rank's presence.
4. Wait until `ready` is set to 1 (spin-wait with exponential backoff, or `futex`/condvar wait on `ready`). This ensures that all ranks are present before any collective operation begins.

### 2.2 Name Collision Prevention

The `COBRE_SHM_NAME` must be unique per concurrent Cobre session on the same node. The calling orchestrator (Python script, shell wrapper) is responsible for generating unique names. Recommended strategies:

- UUID suffix: `/cobre_comm_<uuid4>` (e.g., `/cobre_comm_a1b2c3d4`)
- PID suffix: `/cobre_comm_<rank0_pid>` (simple, sufficient when rank 0's PID is known)

If `shm_open` with `O_EXCL` fails with `EEXIST`, rank 0 reports `BackendError::InitializationFailed` with a message indicating the stale segment. The user must remove the stale segment (`shm_unlink`) or choose a different name.

### 2.3 Shutdown and Cleanup

```rust
#[cfg(feature = "shm")]
impl Drop for ShmBackend {
    fn drop(&mut self) {
        // Unmap the shared memory region.
        // Safety: mapped_size was recorded at mmap time; pointer is valid.
        unsafe {
            libc::munmap(self.control as *mut libc::c_void, self.mapped_size);
        }

        // Close the file descriptor.
        unsafe {
            libc::close(self.shm_fd);
        }

        // Only rank 0 unlinks the shared memory segment.
        // This removes the name from the filesystem, but the segment
        // remains accessible to processes that already have it mapped
        // until they unmap or exit.
        if self.rank == 0 {
            let c_name = std::ffi::CString::new(self.shm_name.clone()).unwrap();
            unsafe {
                libc::shm_unlink(c_name.as_ptr());
            }
        }
    }
}
```

**Ordering constraint:** All ranks must complete their collective operations before any rank drops the `ShmBackend`. The training loop ensures this by synchronizing shutdown (all ranks call a final barrier before dropping the backend). Rank 0 unlinks the segment name on drop, but the underlying memory remains accessible to other ranks that still have it mapped -- `shm_unlink` removes the name, not the mapping.

## 3. Collective Protocols

All collectives are implemented using shared memory buffers in the data region with atomic synchronization via the control region. No TCP sockets, no network I/O -- all communication is through memory reads, writes, and atomic operations on the shared segment.

The data region is partitioned into fixed-size buffers for each collective operation. Buffer offsets are computed deterministically from the rank count and maximum expected payload sizes, ensuring all ranks agree on the layout without explicit negotiation.

### 3.1 allgatherv

Gathers variable-length data from all ranks into all ranks, with data ordered by rank index. This is the most performance-critical operation, called twice per iteration ([Communication Patterns §1.1](./communication-patterns.md)).

**Shared buffer layout:**

The allgatherv buffer in the data region has capacity for the maximum total payload. Each rank writes to a pre-computed offset corresponding to its rank index.

**Protocol:**

1. **Each rank writes its data.** Rank $r$ copies its `send` buffer to the shared allgatherv buffer at byte offset `displs[r] * sizeof(T)`. The write is a direct `memcpy` into the mmap'd region.

2. **Barrier.** All ranks execute a shared memory barrier (SS1.4). This ensures that all writes from step 1 are visible to all ranks before any rank proceeds to read.

3. **All ranks read the complete buffer.** After the barrier, each rank copies the complete assembled data from the shared buffer into its local `recv` buffer. The data is already arranged in rank order (rank 0's data at `displs[0]`, rank 1's data at `displs[1]`, ..., rank $R-1$'s data at `displs[R-1]`) because each rank wrote to its designated offset.

**Determinism guarantee:** Data is ordered by rank index in the shared buffer because each rank writes exclusively to its own offset (`displs[r]`). The barrier guarantees that all writes are globally visible before any read. This preserves the rank-ordered receive invariant from [Communication Patterns §6.1](./communication-patterns.md), producing identical `recv` buffers on all ranks.

**Data movement analysis:** Per `allgatherv` call with total payload $D$ bytes across $R$ ranks:

| Direction          | Data Volume           | Description                                            |
| ------------------ | --------------------- | ------------------------------------------------------ |
| Each rank writes   | $D / R$ (average)     | Each rank writes its contribution to the shared buffer |
| Each rank reads    | $D$                   | Each rank reads the complete assembled buffer          |
| **Total per rank** | $D / R + D \approx D$ | Dominated by the read of the full buffer               |
| **Shared bus**     | $D + R \times D$      | Total memory bandwidth (write once, read $R$ times)    |

Compared to the TCP backend (SS6 of [TCP Backend §6.2](./backend-tcp.md)), shared memory eliminates all network I/O and coordinator bottleneck overhead. The cost is purely memory bandwidth on the shared bus.

### 3.2 allreduce

Reduces data element-wise across all ranks using the specified operation, with the result available on all ranks. The payload is minimal (32 bytes for convergence statistics).

**Shared buffer layout:**

The allreduce buffer contains $R$ per-rank input slots (each sized for the maximum allreduce payload) and one output slot for the result.

**Protocol:**

1. **Each rank writes its data to its per-rank slot.** Rank $r$ copies its `send` buffer to slot $r$ in the shared allreduce buffer.

2. **Barrier (first).** All ranks execute a shared memory barrier to ensure all input slots are populated.

3. **Rank 0 reduces.** Rank 0 reads all $R$ slots and computes the element-wise reduction (applying `ReduceOp::Sum`, `ReduceOp::Min`, or `ReduceOp::Max` as specified). The reduction is performed in rank order (slot 0, slot 1, ..., slot $R-1$) for determinism. Rank 0 writes the result to the output slot.

4. **Barrier (second).** All ranks execute a shared memory barrier to ensure rank 0's result is visible.

5. **All ranks read the result.** Each rank copies the output slot into its local `recv` buffer.

**Floating-point note:** Because rank 0 reduces in a fixed sequential order (rank 0, 1, ..., $R-1$), the floating-point result is deterministic for a given rank count and data. This matches the TCP backend's sequential reduction and is stricter than the MPI specification's implementation-defined reduction tree ([Communicator Trait §2.2](./communicator-trait.md)).

### 3.3 broadcast

Distributes data from a designated root rank to all other ranks. Used only during initialization -- not on the per-iteration hot path.

**Shared buffer layout:**

The broadcast buffer is a single shared region sized for the maximum broadcast payload.

**Protocol:**

1. **Root writes.** The root rank copies its `buf` contents to the shared broadcast buffer.

2. **Barrier.** All ranks execute a shared memory barrier to ensure the root's write is visible.

3. **Non-root ranks read.** Each non-root rank copies the shared broadcast buffer into its local `buf`.

**Determinism guarantee:** Broadcast is inherently deterministic -- all ranks receive an identical copy of the root rank's data.

### 3.4 barrier

Blocks until all ranks have entered the barrier. Used for checkpoint synchronization and as the building block for all collective protocols above.

**Protocol:**

The barrier uses the algorithm described in SS1.4 (atomic counter + futex/condvar):

1. Each rank atomically increments `barrier_count`.
2. The last rank to arrive (`barrier_count` reaches $R$) resets the counter to 0, increments `barrier_generation`, and wakes all waiters.
3. All other ranks wait on the generation counter until it advances.

**Error detection:** If a rank crashes before reaching the barrier, the remaining ranks will wait indefinitely. A configurable timeout (defaulting to 60 seconds) limits the wait time. If the timeout expires, the barrier returns `CommError::CollectiveFailed` with a message indicating a suspected rank crash.

### 3.5 Collective Sequencing

Consecutive collective operations must not interfere with each other. The `collective_seq` counter in the control region sequences collectives: rank 0 increments the sequence number after each collective completes. All ranks wait for the sequence number to match before starting the next collective, preventing a fast rank from overwriting a shared buffer while a slow rank is still reading from the previous collective.

## 4. SharedMemoryProvider

The shm backend implements `SharedMemoryProvider` using POSIX shared memory for true inter-process shared memory. This is the key differentiator from the local and TCP backends, which use `HeapFallback` ([Communicator Trait §4.4](./communicator-trait.md)). The shm backend's shared regions match the semantics of the ferrompi backend's MPI windows ([Ferrompi Backend §3](./backend-ferrompi.md)), enabling the same memory savings from the leader/follower allocation pattern ([Shared Memory Aggregation §1.1](./shared-memory-aggregation.md)).

### 4.1 Implementation

```rust
#[cfg(feature = "shm")]
impl SharedMemoryProvider for ShmBackend {
    type Region<T: CommData> = ShmRegion<T>;

    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, CommError> {
        // Generate a unique name for this shared region.
        // The name is derived from the base segment name and a
        // monotonically increasing region counter.
        let region_name = format!("{}_region_{}", self.shm_name, next_region_id());
        let byte_size = count * std::mem::size_of::<T>();

        if self.rank == 0 {
            // Leader: create and allocate the full shared region.
            // shm_open with O_CREAT | O_EXCL | O_RDWR creates a new segment.
            let fd = shm_open(&region_name, O_CREAT | O_EXCL | O_RDWR, 0o600)?;

            // Set the segment size to hold `count` elements of type T.
            ftruncate(fd, byte_size)?;

            // Map the segment into the process address space.
            let ptr = mmap(
                std::ptr::null_mut(),
                byte_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )?;

            Ok(ShmRegion {
                ptr: ptr as *mut T,
                count,
                byte_size,
                fd,
                name: region_name,
                is_leader: true,
                _marker: std::marker::PhantomData,
            })
        } else {
            // Follower: open the existing segment created by the leader.
            // Retry with exponential backoff until the segment is available.
            let fd = shm_open_with_retry(&region_name, O_RDWR, 0)?;

            // Map the segment (same size as leader).
            let ptr = mmap(
                std::ptr::null_mut(),
                byte_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )?;

            Ok(ShmRegion {
                ptr: ptr as *mut T,
                count,
                byte_size,
                fd,
                name: region_name,
                is_leader: false,
                _marker: std::marker::PhantomData,
            })
        }
    }

    fn split_local(&self) -> Result<Box<dyn Communicator>, CommError> {
        // All shm ranks are on the same node; the intra-node communicator
        // is the shm communicator itself (same rank, same size).
        Ok(Box::new(ShmBackend {
            rank: self.rank,
            size: self.size,
            shm_name: self.shm_name.clone(),
            control: self.control,
            data: self.data,
            data_size: self.data_size,
            shm_fd: self.shm_fd,
            mapped_size: self.mapped_size,
        }))
    }

    fn is_leader(&self) -> bool {
        // Rank 0 is the leader, consistent with the ferrompi convention.
        self.rank == 0
    }
}
```

**Leader determination:** Rank 0 is the leader, consistent with the ferrompi backend convention ([Hybrid Parallelism §6 Step 3](./hybrid-parallelism.md)) and the leader/follower pattern in [Communicator Trait §4.3](./communicator-trait.md). Since all shm ranks are on the same node, local rank 0 is simply rank 0.

### 4.2 ShmRegion Wrapper

The `ShmRegion<T>` type wraps a POSIX shared memory mapping and implements the `SharedRegion<T>` trait. The wrapper encapsulates the `unsafe` pointer dereference into the mmap'd region, presenting a safe Rust interface to the training loop.

```rust
/// Shared memory region backed by POSIX shared memory (`shm_open`/`mmap`).
///
/// Lifecycle follows [Communicator Trait §4.2](./communicator-trait.md).
/// RAII: dropping calls `munmap` and (for the leader) `shm_unlink`.
#[cfg(feature = "shm")]
pub struct ShmRegion<T: CommData> {
    /// Pointer to the mmap'd shared memory region.
    ptr: *mut T,

    /// Number of elements of type T in the region.
    count: usize,

    /// Size of the mapping in bytes (count * sizeof(T)).
    byte_size: usize,

    /// File descriptor for the POSIX shared memory segment.
    fd: i32,

    /// Name of the POSIX shared memory segment (for shm_unlink).
    name: String,

    /// Whether this rank is the leader (rank 0).
    is_leader: bool,

    /// PhantomData for the element type.
    _marker: std::marker::PhantomData<T>,
}
```

```rust
#[cfg(feature = "shm")]
impl<T: CommData> SharedRegion<T> for ShmRegion<T> {
    fn as_slice(&self) -> &[T] {
        // Safety: pointer is valid for `count` elements (set at mmap time).
        // Caller must call fence() before reading to ensure no data races.
        unsafe { std::slice::from_raw_parts(self.ptr, self.count) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // Only the leader writes during population. Followers have a
        // PROT_READ | PROT_WRITE mapping but the protocol prevents writes.
        // Safety: pointer valid for `count` elements; &mut self prevents
        // concurrent access through this handle.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.count) }
    }

    fn fence(&self) -> Result<(), CommError> {
        // SeqCst fence commits all local stores to the mapping, then
        // shared_barrier() ensures all ranks see each other's writes.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        self.shared_barrier()
    }
}
```

The `shared_barrier()` call uses the barrier algorithm from SS1.4. This is a collective operation -- all ranks sharing the region must call `fence()` for it to complete.

```rust
#[cfg(feature = "shm")]
impl<T: CommData> Drop for ShmRegion<T> {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.byte_size);
            libc::close(self.fd);
        }

        // Only the leader unlinks the segment; followers only unmap/close.
        if self.is_leader {
            let c_name = std::ffi::CString::new(self.name.clone()).unwrap();
            unsafe {
                libc::shm_unlink(c_name.as_ptr());
            }
        }
    }
}
```

### 4.3 Shared Data Candidates

The shm backend supports the same shared data candidates as the ferrompi backend, with identical memory savings:

| Data                | Population Strategy                  | Size (production scale) | Reference                                                          |
| ------------------- | ------------------------------------ | ----------------------- | ------------------------------------------------------------------ |
| Opening tree        | Distributed across intra-node ranks  | ~0.8 MB                 | [Scenario Generation §2.3](../architecture/scenario-generation.md) |
| Input case data     | Leader loads, followers read         | ~20 MB                  | [Shared Memory Aggregation §1.3](./shared-memory-aggregation.md)   |
| Cut pool (optional) | Leader integrates after `allgatherv` | ~250 MB at capacity     | [Shared Memory Aggregation §1.4](./shared-memory-aggregation.md)   |

**Memory footprint comparison:**

| Configuration              | Opening Tree | Case Data | Total Shareable | Savings vs. Replicated |
| -------------------------- | -----------: | --------: | --------------: | ---------------------: |
| HeapFallback (4 ranks)     |       3.2 MB |     80 MB |         83.2 MB |                   none |
| True shared, shm (4 ranks) |       0.8 MB |     20 MB |         20.8 MB |               ~62.4 MB |
| True shared, MPI (4 ranks) |       0.8 MB |     20 MB |         20.8 MB |               ~62.4 MB |

The shm backend achieves identical memory savings to the ferrompi backend for single-node deployments, consistent with [Memory Architecture §2.2](./memory-architecture.md).

## 5. NUMA Considerations

Shared memory regions created by the shm backend may span NUMA domains on multi-socket systems. This introduces performance considerations that do not arise with per-process heap allocation.

### 5.1 First-Touch Policy

On Linux, the default NUMA memory allocation policy for `mmap`'d shared memory is first-touch: physical pages are allocated on the NUMA node of the first process (or thread) to write them. For the shm backend:

- **Leader-populated regions** (case data, cut pool): All pages are allocated on the leader's (rank 0's) NUMA node, because rank 0 writes the data during the population phase. Followers on other NUMA nodes experience remote memory access latency when reading.
- **Distributed-population regions** (opening tree): If each rank writes its assigned portion during the population phase, pages are distributed across NUMA nodes proportionally. This achieves better NUMA locality for subsequent reads by each rank.

### 5.2 Performance Trade-Off

| Access Pattern                | NUMA-Local (per-rank heap) | NUMA-Remote (shared, leader-populated) | Impact                                    |
| ----------------------------- | -------------------------: | -------------------------------------: | ----------------------------------------- |
| Read latency (per cache line) |                     ~70 ns |                                ~120 ns | ~1.7x slower for cross-NUMA reads         |
| Read bandwidth (sequential)   |                   ~40 GB/s |                               ~20 GB/s | ~2x lower for cross-NUMA sequential scans |
| Memory footprint (4 ranks)    |               $4 \times D$ |                                    $D$ | 4x memory savings with shared memory      |

The trade-off is memory footprint versus access latency. For read-heavy workloads on shared data (opening tree reads during backward pass, case data reads during LP construction), the latency impact is measurable but typically dominated by LP solve time. Profiling should guide whether to use distributed population for NUMA-sensitive data.

### 5.3 Mitigation Strategies

1. **Distributed population** for the opening tree: each rank writes its assigned portion via `as_mut_slice()` at the correct offset, triggering first-touch allocation on each rank's NUMA node. This matches the generation protocol in [Shared Memory Aggregation §1.2](./shared-memory-aggregation.md).

2. **NUMA interleave policy**: On Linux, the orchestrator can set `COBRE_SHM_NUMA_INTERLEAVE=1` (future extension) to request `mbind(MPOL_INTERLEAVE)` on the shared memory region, distributing pages round-robin across NUMA nodes. This provides uniform-average access latency at the cost of slightly higher average latency than NUMA-local access.

3. **One rank per NUMA domain**: The recommended deployment (one MPI rank per NUMA domain, [Hybrid Parallelism §8](./hybrid-parallelism.md)) also applies to the shm backend. When each rank is pinned to a NUMA domain, shared memory regions populated by distributed first-touch achieve good NUMA locality.

## 6. Platform Requirements

### 6.1 Supported Platforms

| Platform | Support | POSIX shm APIs                                          | Notes                                              |
| -------- | :-----: | ------------------------------------------------------- | -------------------------------------------------- |
| Linux    |   Yes   | `shm_open`, `shm_unlink`, `mmap`, `munmap`, `ftruncate` | Requires `librt` linkage; `futex` for barriers     |
| macOS    |   Yes   | `shm_open`, `shm_unlink`, `mmap`, `munmap`, `ftruncate` | Uses `PTHREAD_PROCESS_SHARED` condvar for barriers |
| Windows  |   No    | N/A                                                     | No POSIX shm; would require `CreateFileMapping`    |

### 6.2 Required System Calls and Libraries

| API                      | Header            | Library    | Purpose                                        |
| ------------------------ | ----------------- | ---------- | ---------------------------------------------- |
| `shm_open`               | `<sys/mman.h>`    | `librt`    | Create or open named shared memory segment     |
| `shm_unlink`             | `<sys/mman.h>`    | `librt`    | Remove named shared memory segment             |
| `mmap`                   | `<sys/mman.h>`    | libc       | Map shared memory into process address space   |
| `munmap`                 | `<sys/mman.h>`    | libc       | Unmap shared memory from process address space |
| `ftruncate`              | `<unistd.h>`      | libc       | Set size of shared memory segment              |
| `close`                  | `<unistd.h>`      | libc       | Close shared memory file descriptor            |
| `futex` (Linux)          | `<linux/futex.h>` | libc       | Atomic wait/wake for barrier synchronization   |
| `pthread_cond_*` (macOS) | `<pthread.h>`     | libpthread | Condition variable for barrier synchronization |

### 6.3 System Limits

The following system limits constrain shared memory segment sizes:

| Limit                                   | Default (Linux)       | How to Query / Modify                                                            |
| --------------------------------------- | --------------------- | -------------------------------------------------------------------------------- |
| Maximum shared memory segment size      | 50% of physical RAM   | `/proc/sys/kernel/shmmax` (System V); POSIX shm limited by `/dev/shm` tmpfs size |
| Maximum total shared memory             | `/dev/shm` tmpfs size | `df /dev/shm`; resize via `mount -o remount,size=16G /dev/shm`                   |
| Maximum number of shared memory objects | System-dependent      | Typically unlimited on modern kernels                                            |

For production-scale SDDP (4 ranks, ~300 MB shared data), the default `/dev/shm` size (typically 50% of RAM) is more than sufficient. If the system has limited `/dev/shm`, the orchestrator must resize it before launching the Cobre processes.

### 6.4 Feature Gating

The shm backend is gated behind the `shm` Cargo feature flag:

```toml
[features]
default = []
shm = ["dep:libc"]   # POSIX shm APIs via the libc crate

[dependencies]
libc = { version = "0.2", optional = true }
```

When `shm` is not enabled:

- The `ShmBackend` and `ShmRegion` types do not exist.
- No POSIX shm headers or `librt` linkage is required at build time.
- The `create_communicator()` factory function ([Backend Registration and Selection §4](./backend-selection.md)) does not include a shm branch.
- `COBRE_COMM_BACKEND=shm` produces an error listing available backends.

When `shm` is enabled:

- The `libc` crate provides FFI bindings for `shm_open`, `mmap`, etc.
- `librt` is linked on Linux (for `shm_open`/`shm_unlink`).
- The shm backend is available for selection via the factory function or `COBRE_COMM_BACKEND=shm`.
- In the auto-detection priority chain ([Backend Registration and Selection §2.2](./backend-selection.md)), the shm backend is selected when `COBRE_SHM_NAME` is set and neither the `mpi` nor `tcp` backend was selected first.

### 6.5 Build Profile Integration

The shm backend is included in the following build profiles from [Backend Registration and Selection §1.3](./backend-selection.md):

| Build Profile | Includes `shm`? | Rationale                                                            |
| ------------- | :-------------: | -------------------------------------------------------------------- |
| CLI / HPC     |       No        | MPI is the production communication layer for cluster deployment     |
| Python wheel  |       Yes       | Shared memory enables efficient intra-node communication without MPI |
| Test / CI     |       No        | Only `local` backend; no external dependencies                       |
| Development   |       Yes       | All backends compiled for testing                                    |

## 7. Configuration

### 7.1 Environment Variables

The shm backend is configured via environment variables following the `COBRE_` prefix convention from [Backend Registration and Selection §3.2](./backend-selection.md).

| Variable         | Required | Default  | Description                                                                                                                                       |
| ---------------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `COBRE_SHM_NAME` | Yes      | _(none)_ | Name of the POSIX shared memory segment (e.g., `/cobre_comm_abc123`). Must begin with `/` per POSIX convention. All ranks must use the same name. |
| `COBRE_SHM_RANK` | Yes      | _(none)_ | Rank index of this process (`0..COBRE_SHM_SIZE`). Rank 0 creates the segment; all others open it.                                                 |
| `COBRE_SHM_SIZE` | Yes      | _(none)_ | Total number of ranks sharing the memory segment. Must be consistent across all processes. Minimum 1, maximum limited by system resources.        |

**Rank assignment:** Ranks are assigned externally by the orchestrator (Python script, shell wrapper). Each process must be started with a unique `COBRE_SHM_RANK` in `0..COBRE_SHM_SIZE`. Rank 0 is the leader responsible for segment creation and cleanup.

### 7.2 Example Invocation

```bash
# Generate a unique shared memory segment name
SHM_NAME="/cobre_comm_$(uuidgen | head -c 8)"

# Start rank 0 (creates the shared memory segment)
COBRE_COMM_BACKEND=shm \
COBRE_SHM_NAME="$SHM_NAME" \
COBRE_SHM_RANK=0 \
COBRE_SHM_SIZE=4 \
cobre run /path/to/case &

# Start ranks 1-3 (open the existing segment)
for rank in 1 2 3; do
    COBRE_COMM_BACKEND=shm \
    COBRE_SHM_NAME="$SHM_NAME" \
    COBRE_SHM_RANK=$rank \
    COBRE_SHM_SIZE=4 \
    cobre run /path/to/case &
done

wait
```

### 7.3 Python Multi-Process Usage

The shm backend is the recommended backend for Python multi-process SDDP on a single node, providing true shared memory without MPI. The high-level API handles all worker spawning, rank assignment, and shared memory segment naming internally:

```python
import cobre

case = cobre.CaseLoader.load("/path/to/case")
result = cobre.train(case, num_workers=4, backend="shm")

# result is rank 0's TrainingResult; result.workers has per-worker metadata
print(f"Converged in {result.iterations} iterations")
for w in result.workers:
    print(f"  Worker {w.rank}: {w.wall_time_ms} ms, backend={w.backend}")
```

Internally, `cobre.train()` with `num_workers > 1` performs the following steps (see [Python Bindings](../interfaces/python-bindings.md) SS2.1a for the full lifecycle):

1. Generates a unique POSIX shared memory segment name (e.g., `/cobre_comm_<random_hex>`).
2. Spawns `num_workers` child processes via `multiprocessing.Process` with `start_method="spawn"`.
3. Each child creates a `ShmBackend` communicator for its assigned rank and runs the SDDP loop.
4. The parent waits for all children via `Process.join()` and collects rank 0's result.

The low-level per-rank parameters (`shm_name`, `shm_rank`, `shm_size`) are set by the library on each worker process's environment; users do not need to manage them directly.

## 8. Error Mapping

Shm backend failures are mapped to existing `CommError` variants from [Communicator Trait §1.4](./communicator-trait.md) and [Communicator Trait §4.6](./communicator-trait.md). No new `CommError` variants are introduced by this backend.

| Shm Failure                                          | CommError Variant                                                                  |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `shm_open` fails (`EACCES`, `ENOENT`, `EEXIST`)      | `CommError::CollectiveFailed { operation: "init", mpi_error_code: 0, message }`    |
| `ftruncate` fails (size exceeds limit)               | `CommError::AllocationFailed { requested_bytes, message }`                         |
| `mmap` fails (`ENOMEM`)                              | `CommError::AllocationFailed { requested_bytes, message }`                         |
| Barrier timeout (rank crash suspected)               | `CommError::CollectiveFailed { operation: "barrier", mpi_error_code: 0, message }` |
| Unexpected `barrier_generation` (protocol violation) | `CommError::CollectiveFailed { operation, mpi_error_code: 0, message }`            |
| Handshake failure (wrong size)                       | Reported as `BackendError::InitializationFailed` (initialization-time error)       |

The `mpi_error_code` field is set to `0` for all shm errors because there is no MPI error code to report. The `message` field contains the specific OS error description for diagnostic logging.

## Cross-References

- [Communicator Trait §1](./communicator-trait.md) -- `Communicator` trait definition, `CommData`, `ReduceOp`, `CommError` type definitions implemented by this backend
- [Communicator Trait §2](./communicator-trait.md) -- Method contracts (preconditions, postconditions, determinism guarantees) that this backend preserves via shared memory protocols in SS3
- [Communicator Trait §3](./communicator-trait.md) -- Generic parameterization pattern (`train<C: Communicator>`) enabling compile-time monomorphization of this backend
- [Communicator Trait §4](./communicator-trait.md) -- `SharedMemoryProvider` trait, `SharedRegion<T>` lifecycle phases, leader/follower pattern, drop behavior table
- [Communicator Trait §4.2](./communicator-trait.md) -- `SharedRegion<T>` lifecycle (allocation, population, read-only access) and RAII drop semantics implemented by `ShmRegion<T>`
- [Communicator Trait §4.3](./communicator-trait.md) -- Leader/follower allocation pattern: leader allocates, followers get handle to leader's memory
- [Communicator Trait §4.6](./communicator-trait.md) -- `CommError::AllocationFailed` variant used for `mmap`/`ftruncate` failures
- [Communication Patterns §1.1](./communication-patterns.md) -- Three collective operations per iteration (allgatherv x2, allreduce x1), plus initialization-only broadcast and barrier
- [Communication Patterns §2](./communication-patterns.md) -- Data payloads for trial points (§2.1), cuts (§2.2), and convergence statistics (§2.3) transmitted through the shm backend
- [Communication Patterns §5](./communication-patterns.md) -- `SharedWindow<T>` capabilities and shared data candidates; the shm backend provides equivalent functionality via POSIX shm
- [Communication Patterns §6.1](./communication-patterns.md) -- Deterministic rank-ordered receive invariant preserved by rank-ordered writes in SS3.1
- [Communication Patterns §6.2](./communication-patterns.md) -- Floating-point reduction tolerance; the shm backend's sequential reduction is deterministic for fixed rank count
- [Shared Memory Aggregation §1.1](./shared-memory-aggregation.md) -- Leader allocation pattern, SharedWindow usage model replicated by the shm backend's POSIX shm regions
- [Shared Memory Aggregation §1.2](./shared-memory-aggregation.md) -- Opening tree generation protocol: distributed population, fence synchronization
- [Shared Memory Aggregation §1.3](./shared-memory-aggregation.md) -- Shared input case data candidates and sizes
- [Shared Memory Aggregation §1.4](./shared-memory-aggregation.md) -- Cut pool as shared memory optimization candidate
- [Hybrid Parallelism §1.3](./hybrid-parallelism.md) -- Shared memory layout: scenario storage and cut pool regions shared within a node
- [Hybrid Parallelism §4.4](./hybrid-parallelism.md) -- NUMA allocation policies (local allocation, first-touch) that interact with POSIX shared memory page placement
- [Hybrid Parallelism §6 Step 3](./hybrid-parallelism.md) -- Shared memory communicator creation; the shm backend provides equivalent functionality without MPI
- [Memory Architecture §2.2](./memory-architecture.md) -- SharedWindow savings: per-node memory reduction from sharing opening tree and case data
- [Ferrompi Backend §3](./backend-ferrompi.md) -- Reference `SharedMemoryProvider` implementation using MPI windows; the shm backend provides equivalent semantics via POSIX shm
- [TCP Backend §7](./backend-tcp.md) -- TCP backend's HeapFallback `SharedMemoryProvider`; the shm backend differs by providing true shared memory
- [Backend Registration and Selection §1.2](./backend-selection.md) -- Feature flag matrix; `shm` feature gates the POSIX shared memory dependency
- [Backend Registration and Selection §2.2](./backend-selection.md) -- Auto-detection priority chain; shm is selected when `COBRE_SHM_NAME` is set and neither MPI nor TCP was selected first
- [Backend Registration and Selection §3.1](./backend-selection.md) -- Per-backend environment variable table for the shm backend (`COBRE_SHM_NAME`, `COBRE_SHM_RANK`, `COBRE_SHM_SIZE`)
- [Backend Registration and Selection §4](./backend-selection.md) -- Factory pattern returning `ShmBackend` in the `CommBackend` enum for multi-feature builds
- [Solver Abstraction §10](../architecture/solver-abstraction.md) -- Compile-time selection pattern via generic parameters and Cargo feature flags; the architectural precedent for this backend's feature-gated design
