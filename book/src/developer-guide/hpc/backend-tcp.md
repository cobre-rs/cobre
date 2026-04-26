# TCP Backend

> **Status: Not Implemented.** This specification describes planned behavior for a future backend. No implementation exists in the current codebase.

## Purpose

The TCP backend enables multi-process SDDP execution using standard TCP sockets (`std::net`), eliminating the dependency on an MPI runtime. It uses a coordinator-based star topology where rank 0 runs a TCP server and ranks 1 through $R-1$ connect as clients. All collective operations flow through the coordinator, which is simpler than a peer-to-peer mesh and sufficient for SDDP's collective-only communication pattern ([Communication Patterns §1.2](./communication-patterns.md)). This backend targets two deployment scenarios: container-friendly environments where MPI installation is impractical, and Python multi-process orchestration where the GIL is incompatible with MPI launchers ([Python Bindings §7](../interfaces/python-bindings.md)). The backend is gated behind the `tcp` Cargo feature flag as specified in [Backend Registration and Selection §1.2](./backend-selection.md).

## 1. Architecture

### 1.1 Coordinator Pattern

The TCP backend implements a star topology with a single coordinator (rank 0) and $R-1$ worker clients. The coordinator manages all collective operations by receiving data from workers, performing the required aggregation or synchronization, and distributing results back to all participants.

```
                    +-------------------+
                    |   Coordinator     |
                    |   (rank 0)        |
                    |                   |
                    |   TCP listener    |
                    |   port: P         |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
         +----+----+   +----+----+   +----+----+
         | Worker  |   | Worker  |   | Worker  |
         | rank 1  |   | rank 2  |   | rank R-1|
         |         |   |         |   |         |
         | TCP     |   | TCP     |   | TCP     |
         | client  |   | client  |   | client  |
         +---------+   +---------+   +---------+
```

**Design rationale:** SDDP uses only collective operations, no point-to-point messaging ([Communication Patterns §1.2](./communication-patterns.md)). A coordinator-based star topology maps naturally to collectives: every operation passes through a single hub that enforces ordering, performs reductions, and detects failures. A peer-to-peer mesh would add $O(R^2)$ connection complexity without benefit.

### 1.2 Struct Definition

The `TcpBackend` struct holds the persistent TCP connections and rank metadata established during initialization. The coordinator holds $R-1$ accepted connections (one per worker), while each worker holds a single connection to the coordinator.

```rust
/// TCP-based communication backend for MPI-free multi-process SDDP.
///
/// Uses a coordinator pattern where rank 0 runs a TCP server and all
/// other ranks connect as clients. All collective operations are
/// mediated by the coordinator. Connections are persistent for the
/// lifetime of the training run (no per-operation connect/disconnect).
#[cfg(feature = "tcp")]
pub struct TcpBackend {
    /// This process's rank index (0..size).
    rank: usize,

    /// Total number of ranks in the communicator.
    size: usize,

    /// Timeout for collective operations and connection attempts.
    timeout: Duration,

    /// Connection state, determined by whether this rank is the coordinator.
    role: TcpRole,
}

/// The role of this rank in the TCP communicator.
#[cfg(feature = "tcp")]
enum TcpRole {
    /// Rank 0: holds accepted connections from all R-1 workers,
    /// indexed by worker rank (1..size).
    Coordinator {
        listener: TcpListener,
        workers: Vec<TcpStream>,  // workers[i] = connection to rank i+1
    },

    /// Ranks 1..R-1: holds a single connection to the coordinator.
    Worker {
        conn: TcpStream,
    },
}
```

### 1.3 Communicator Trait Implementation

The `TcpBackend` implements `Communicator` by routing each collective operation through the coordinator using the protocols defined in SS3. The coordinator participates as both a server (mediating the collective) and a rank (contributing its own data).

```rust
#[cfg(feature = "tcp")]
impl Communicator for TcpBackend {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), CommError> {
        // Protocol: SS3.1
        // ...
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: ReduceOp,
    ) -> Result<(), CommError> {
        // Protocol: SS3.2
        // ...
    }

    fn broadcast<T: CommData>(
        &self,
        buf: &mut [T],
        root: usize,
    ) -> Result<(), CommError> {
        // Protocol: SS3.3
        // ...
    }

    fn barrier(&self) -> Result<(), CommError> {
        // Protocol: SS3.4
        // ...
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }
}
```

**Error mapping:** All TCP I/O errors are mapped to `CommError` variants from [Communicator Trait §1.4](./communicator-trait.md). No new variants are introduced. See SS5 for the complete mapping.

## 2. Connection Establishment

### 2.1 Startup Sequence

Connection establishment occurs once during the Startup phase ([CLI and Lifecycle §5](../architecture/cli-and-lifecycle.md)), before any collective operation. The sequence differs for the coordinator and workers:

**Coordinator (rank 0):**

1. Bind a `TcpListener` on `0.0.0.0:COBRE_TCP_PORT` (default: 29500).
2. Accept exactly $R-1$ connections, one from each worker rank.
3. For each accepted connection, execute the handshake protocol (SS2.2) to verify the worker's identity.
4. Store the authenticated connections in rank order (`workers[0]` = rank 1, `workers[1]` = rank 2, ...).
5. The coordinator is ready for collective operations once all $R-1$ workers have completed the handshake.

**Worker (rank $r$, where $1 \leq r \leq R-1$):**

1. Read `COBRE_TCP_COORDINATOR` (host) and `COBRE_TCP_PORT` (port) from the environment.
2. Connect to the coordinator at `COBRE_TCP_COORDINATOR:COBRE_TCP_PORT` with a configurable timeout (SS8).
3. Execute the handshake protocol (SS2.2) to register with the coordinator.
4. The worker is ready for collective operations once the handshake acknowledgment is received.

### 2.2 Handshake Protocol

After a TCP connection is established, the worker and coordinator execute a handshake to verify rank identity and communicator size consistency.

**Step 1 -- Worker sends handshake:**

The worker sends a `Handshake` message (SS4 framing) containing:

| Field  | Type       | Size    | Description                                       |
| ------ | ---------- | ------- | ------------------------------------------------- |
| `rank` | `u32` (BE) | 4 bytes | The worker's rank index (`COBRE_TCP_RANK`)        |
| `size` | `u32` (BE) | 4 bytes | The expected communicator size (`COBRE_TCP_SIZE`) |

**Step 2 -- Coordinator verifies:**

The coordinator validates the received handshake:

- `rank` must be in the range `1..size` (rank 0 is the coordinator itself).
- `rank` must not have been registered by a previous connection (no duplicate ranks).
- `size` must match the coordinator's own `COBRE_TCP_SIZE` value.

If validation fails, the coordinator sends an error response and closes the connection.

**Step 3 -- Coordinator sends acknowledgment:**

On successful validation, the coordinator sends an `Ack` message (SS4 framing) containing:

| Field  | Type       | Size    | Description                                   |
| ------ | ---------- | ------- | --------------------------------------------- |
| `size` | `u32` (BE) | 4 bytes | The confirmed communicator size (echoed back) |

The worker verifies that the echoed `size` matches its own `COBRE_TCP_SIZE`. If the values disagree, the worker reports an initialization error and terminates.

### 2.3 Persistent Connections

All TCP connections established during the handshake phase remain open for the entire lifetime of the training run. There is no per-operation connect/disconnect cycle. Persistent connections avoid:

- TCP handshake latency (~1 ms per connection) on every collective operation.
- Port exhaustion from rapid connect/disconnect cycles during the ~100-200 iterations of SDDP training.
- TIME_WAIT socket state accumulation.

Connections are closed during the shutdown phase, after all collective operations have completed.

### 2.4 TCP Socket Configuration

All persistent connections are configured for low-latency communication:

| Socket Option  | Value          | Rationale                                                   |
| -------------- | -------------- | ----------------------------------------------------------- |
| `TCP_NODELAY`  | `true`         | Disable Nagle's algorithm to avoid buffering small messages |
| `SO_KEEPALIVE` | `true`         | Detect broken connections from rank crashes                 |
| Read timeout   | `self.timeout` | Configurable timeout for detecting unresponsive ranks (SS5) |
| Write timeout  | `self.timeout` | Configurable timeout for detecting blocked writes (SS5)     |

## 3. Collective Protocols

Each collective operation defined in [Communicator Trait §1.1](./communicator-trait.md) is implemented as a coordinator-mediated message exchange. The coordinator handles both its own data (as rank 0) and the relay of worker data. All messages use the framing format defined in SS4.

### 3.1 allgatherv

Gathers variable-length data from all ranks and distributes the assembled result to all ranks, with data ordered by rank index. This is the most performance-critical operation, called twice per iteration ([Communication Patterns §1.1](./communication-patterns.md)).

**Message exchange sequence:**

1. **Workers send data to coordinator.** Each worker rank $r$ ($1 \leq r \leq R-1$) sends an `AllgathervSend` message containing its local data buffer (`send[0..counts[r]]`). Messages arrive asynchronously -- workers do not wait for each other.

2. **Coordinator collects all contributions.** The coordinator:
   - Places its own data (rank 0's `send` buffer) into the result buffer at offset `displs[0]`.
   - Receives each worker's `AllgathervSend` message and places the received data at offset `displs[r]` in the result buffer, where $r$ is the sending worker's rank.
   - Blocks until all $R-1$ worker contributions have been received (implicit barrier).

3. **Coordinator sends assembled result to all.** Once all contributions are collected, the coordinator sends an `AllgathervRecv` message to each worker containing the complete assembled buffer (`recv[0..sum(counts)]`).

4. **Workers receive the assembled result.** Each worker receives the `AllgathervRecv` message and copies the contents into its `recv` buffer.

**Determinism guarantee:** The coordinator assembles received data into the result buffer in rank order (rank 0 at `displs[0]`, rank 1 at `displs[1]`, ..., rank $R-1$ at `displs[R-1]`), regardless of the order in which worker messages arrive on the network. This preserves the rank-ordered receive invariant from [Communication Patterns §6.1](./communication-patterns.md) and ensures that all ranks construct identical cut pools and trial point sets after synchronization.

**Data movement analysis:** Per `allgatherv` call with total payload $D$ bytes across $R$ ranks:

| Direction         | Data Volume               | Description                                   |
| ----------------- | ------------------------- | --------------------------------------------- |
| Workers to coord  | $D - D_0$ (rank 0's data) | Each worker sends its contribution            |
| Coord to workers  | $(R-1) \times D$          | Coordinator sends the full assembled buffer   |
| **Total on wire** | $\approx R \times D$      | Dominated by the coordinator's fan-out to all |

### 3.2 allreduce

Reduces data element-wise across all ranks using the specified operation, with the result available on all ranks. The payload is minimal (32 bytes for convergence statistics), so coordinator overhead is negligible.

**Message exchange sequence:**

1. **Workers send data to coordinator.** Each worker rank $r$ sends an `AllreduceSend` message containing its local `send` buffer and the `ReduceOp` tag (encoded in the operation tag byte, SS4).

2. **Coordinator reduces element-wise.** The coordinator:
   - Initializes the accumulator from its own `send` buffer (rank 0's contribution).
   - Receives each worker's `AllreduceSend` message and applies the reduction operation element-wise:
     - `ReduceOp::Sum`: `acc[i] += recv[i]` for all $i$.
     - `ReduceOp::Min`: `acc[i] = min(acc[i], recv[i])` for all $i$.
     - `ReduceOp::Max`: `acc[i] = max(acc[i], recv[i])` for all $i$.
   - The reduction order is deterministic: rank 0 first, then ranks $1, 2, \ldots, R-1$ in ascending order.

3. **Coordinator sends result to all.** The coordinator sends an `AllreduceRecv` message to each worker containing the reduced result buffer.

4. **Workers receive the result.** Each worker copies the received result into its `recv` buffer.

**Floating-point note:** Because the coordinator reduces in a fixed sequential order (rank 0, 1, ..., $R-1$), the floating-point result is deterministic for a given rank count and data. This is stricter than the MPI specification, which allows implementation-defined reduction tree shapes ([Communicator Trait §2.2](./communicator-trait.md)). The TCP backend's sequential reduction eliminates one source of floating-point non-determinism compared to the ferrompi backend.

### 3.3 broadcast

Distributes data from a designated root rank to all other ranks. Used only during initialization for configuration and case data -- not on the per-iteration hot path.

**Message exchange sequence:**

**Case 1: root = 0 (coordinator is root).**

1. The coordinator sends a `Broadcast` message to each worker containing the `buf` contents.
2. Each worker receives the `Broadcast` message and copies the contents into `buf`.

**Case 2: root $\neq$ 0 (a worker is root).**

1. The root worker sends a `Broadcast` message to the coordinator containing its `buf` contents.
2. The coordinator receives the message and copies the contents into its own `buf`.
3. The coordinator forwards the `Broadcast` message to all other workers (ranks $\neq$ root, $\neq$ 0).
4. Each non-root worker receives the `Broadcast` message and copies the contents into `buf`.

### 3.4 barrier

Blocks until all ranks have entered the barrier. Used only for checkpoint synchronization -- not on the per-iteration hot path.

**Message exchange sequence:**

1. **Workers send ready signal.** Each worker sends a `BarrierReady` message (zero-byte payload) to the coordinator.

2. **Coordinator waits for all.** The coordinator:
   - Considers itself (rank 0) as implicitly ready.
   - Blocks until all $R-1$ `BarrierReady` messages have been received.

3. **Coordinator releases all.** Once all ranks are accounted for, the coordinator sends a `BarrierGo` message (zero-byte payload) to each worker.

4. **Workers unblock.** Each worker receives the `BarrierGo` message and returns from the `barrier()` call.

## 4. Message Framing

### 4.1 Wire Format

All messages exchanged between the coordinator and workers use a length-prefixed binary format over TCP byte streams. The frame structure is:

```
+------+-----+---------+
| LEN  | TAG | PAYLOAD |
+------+-----+---------+
  4 B    1 B   LEN - 1 B
```

| Field     | Type       | Size            | Description                                                     |
| --------- | ---------- | --------------- | --------------------------------------------------------------- |
| `LEN`     | `u32` (BE) | 4 bytes         | Length of `TAG + PAYLOAD` in bytes (big-endian unsigned 32-bit) |
| `TAG`     | `u8`       | 1 byte          | Operation type identifier                                       |
| `PAYLOAD` | raw bytes  | `LEN - 1` bytes | Operation-specific data                                         |

The `LEN` field encodes the number of bytes following the length header (i.e., `TAG` + `PAYLOAD`), not the total frame size. The maximum payload size is $2^{32} - 2$ bytes ($\approx$ 4 GB), which is far larger than any SDDP collective payload (~587 MB total across all operations per iteration).

### 4.2 Operation Tags

| Tag Value | Name             | Direction       | Payload Contents                                   |
| --------- | ---------------- | --------------- | -------------------------------------------------- |
| `0x01`    | `AllgathervSend` | Worker to coord | Raw bytes of the worker's `send` buffer            |
| `0x02`    | `AllgathervRecv` | Coord to worker | Raw bytes of the assembled `recv` buffer           |
| `0x03`    | `AllreduceSend`  | Worker to coord | 1-byte `ReduceOp` tag + raw bytes of `send` buffer |
| `0x04`    | `AllreduceRecv`  | Coord to worker | Raw bytes of the reduced `recv` buffer             |
| `0x05`    | `Broadcast`      | Bidirectional   | Raw bytes of `buf`                                 |
| `0x06`    | `BarrierReady`   | Worker to coord | Empty (zero bytes)                                 |
| `0x07`    | `BarrierGo`      | Coord to worker | Empty (zero bytes)                                 |
| `0x08`    | `Handshake`      | Worker to coord | `rank` (u32 BE) + `size` (u32 BE)                  |
| `0x09`    | `Ack`            | Coord to worker | `size` (u32 BE)                                    |
| `0x0A`    | `Shutdown`       | Coord to worker | Empty (zero bytes)                                 |

### 4.3 Byte Order and Payload Encoding

All multi-byte integers in the framing layer (`LEN`, `rank`, `size` in handshake) use **big-endian** byte order for unambiguous cross-platform interpretation.

Payload data (the `CommData` buffers transmitted during collectives) is transmitted as raw bytes in the sender's native byte order. Because SDDP deployment targets use homogeneous architectures (all ranks on the same platform), byte order conversion for payload data is not required. The `CommData` trait bound (`Copy + Send + Sync + 'static`) from [Communicator Trait §1.2](./communicator-trait.md) ensures that all payload types have a well-defined memory layout suitable for bitwise transmission.

### 4.4 Reading and Writing Frames

Frame I/O uses Rust's `std::io::Read` and `std::io::Write` traits on `TcpStream`, with `read_exact` and `write_all` to handle partial reads/writes:

```rust
/// Read a complete framed message, returning the operation tag and payload.
fn read_frame(stream: &mut TcpStream) -> Result<(u8, Vec<u8>), CommError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let body_len = u32::from_be_bytes(len_buf) as usize;

    let mut body = vec![0u8; body_len];
    stream.read_exact(&mut body)?;

    let tag = body[0];
    let payload = body[1..].to_vec();
    Ok((tag, payload))
}

/// Write a complete framed message to the stream.
fn write_frame(
    stream: &mut TcpStream,
    tag: u8,
    payload: &[u8],
) -> Result<(), CommError> {
    let body_len = (1 + payload.len()) as u32;
    stream.write_all(&body_len.to_be_bytes())?;
    stream.write_all(&[tag])?;
    stream.write_all(payload)?;
    stream.flush()?;
    Ok(())
}
```

## 5. Failure Handling

### 5.1 Timeout-Based Detection

All TCP read and write operations are subject to a configurable timeout. When a read or write does not complete within the timeout period, the operation fails with `CommError::CollectiveFailed`. The timeout covers the following failure modes:

| Failure Mode       | Detection Mechanism                                     | Timeout Applies To        |
| ------------------ | ------------------------------------------------------- | ------------------------- |
| Rank process crash | TCP connection closed (read returns 0 bytes) or timeout | All collective operations |
| Rank process hang  | Read timeout expires without receiving expected data    | All collective operations |
| Network partition  | Read/write timeout expires                              | All collective operations |
| Coordinator crash  | Workers detect closed connection or timeout             | All collective operations |

**Default timeout:** 60 seconds. This is configurable via the `COBRE_TCP_TIMEOUT_SECS` environment variable (SS8). The default is chosen to accommodate:

- Slow LP solves at production scale (~25 ms per solve, ~192 solves per stage, ~60 stages = ~288 seconds of computation per iteration) where ranks may take significantly different amounts of time to reach the next collective.
- Network transients that may delay message delivery by seconds.

The 60-second default exceeds the expected per-stage wall-clock time (~5 seconds at production scale) by a factor of 12, providing a generous margin while still detecting genuine failures within a reasonable timeframe.

### 5.2 Graceful Shutdown

When the training loop completes (convergence or iteration limit), the coordinator initiates a graceful shutdown:

1. The coordinator sends a `Shutdown` message (tag `0x0A`, zero-byte payload) to each worker.
2. Each worker receives the `Shutdown` message and closes its connection.
3. The coordinator closes all worker connections and the listener socket.

If a worker terminates before receiving the `Shutdown` message (e.g., due to an error in the training loop), the coordinator detects the closed connection during the next collective operation and reports the failure.

### 5.3 Error Mapping

TCP backend failures are mapped to existing `CommError` variants from [Communicator Trait §1.4](./communicator-trait.md). No new variants are introduced.

| TCP Failure                                      | CommError Variant                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `std::io::ErrorKind::TimedOut`                   | `CommError::CollectiveFailed { operation, mpi_error_code: 0, message }`      |
| `std::io::ErrorKind::ConnectionReset`            | `CommError::CollectiveFailed { operation, mpi_error_code: 0, message }`      |
| `std::io::ErrorKind::UnexpectedEof` (rank crash) | `CommError::CollectiveFailed { operation, mpi_error_code: 0, message }`      |
| `std::io::ErrorKind::ConnectionRefused`          | `CommError::CollectiveFailed { operation, mpi_error_code: 0, message }`      |
| Unexpected operation tag                         | `CommError::CollectiveFailed { operation, mpi_error_code: 0, message }`      |
| Length mismatch (frame size vs. expected)        | `CommError::InvalidBufferSize { operation, expected, actual }`               |
| Handshake failure (wrong rank/size)              | Reported as `BackendError::InitializationFailed` (initialization-time error) |

The `mpi_error_code` field is set to `0` for all TCP errors because there is no MPI error code to report. The `message` field contains the specific TCP error description for diagnostic logging.

## 6. Performance Analysis

### 6.1 Communication Volume

The per-iteration communication volume at worst-case scale ($R = 16$ ranks, $T = 120$ stages — hypothetical maximum per DEC-009; production baseline is $T = 60$, $M = 192$ forward passes, $D_{\text{state}} = 2{,}080$) is approximately 587 MB ([Communication Patterns §3.1](./communication-patterns.md)):

| Operation                | Per-Iteration Volume | Frequency              |
| ------------------------ | -------------------: | ---------------------- |
| Trial point `allgatherv` |              ~206 MB | Once per iteration     |
| Cut `allgatherv`         |              ~381 MB | Once per stage (x 119) |
| Convergence `allreduce`  |                 32 B | Once per iteration     |
| **Total**                |          **~587 MB** |                        |

### 6.2 Coordinator Throughput Analysis

In the star topology, the coordinator is the bottleneck: it must receive data from all workers and then send the assembled result to all workers. The total data through the coordinator per iteration is:

**Allgatherv (cuts + trial points):**

$$
D_{\text{coord}} = \underbrace{(R-1) \times \bar{D}_{\text{send}}}_{\text{receive from workers}} + \underbrace{(R-1) \times D_{\text{total}}}_{\text{send to workers}}
$$

where $\bar{D}_{\text{send}}$ is the average per-rank contribution and $D_{\text{total}}$ is the assembled result size.

For trial point `allgatherv` ($D_{\text{total}} \approx 206$ MB, $R = 16$):

- Receive: $15 \times (206/16) \approx 193$ MB
- Send: $15 \times 206 \approx 3{,}090$ MB
- Total coordinator I/O: $\approx 3{,}283$ MB

For cut `allgatherv` (119 stages at worst-case $T = 120$; $D_{\text{total}} \approx 3.2$ MB per stage):

- Per stage: receive $\approx 3.0$ MB + send $\approx 48$ MB = 51 MB
- All stages: $119 \times 51 \approx 6{,}069$ MB
- Total coordinator I/O: $\approx 6{,}069$ MB

**Combined coordinator I/O per iteration:** $\approx 9{,}352$ MB.

### 6.3 Wire-Speed Analysis

On a **100 Gbps Ethernet** link (12.5 GB/s per direction, full-duplex):

| Metric                              |                         Value |
| ----------------------------------- | ----------------------------: |
| Coordinator I/O per iteration       |                     ~9,352 MB |
| Wire-speed transfer time            | $9{,}352 / 12{,}500 = 0.75$ s |
| Protocol overhead factor (est.)     |                          ~50% |
| Estimated wall-clock transfer time  |                       ~112 ms |
| Typical iteration wall-clock time   |                          ~5 s |
| **Communication overhead fraction** |                     **~2.2%** |

The 50% protocol overhead factor accounts for TCP/IP headers (~40 bytes per packet on a ~64 KB segment), kernel-to-user space copies, system call overhead, and Nagle/delayed-ACK interaction (mitigated by `TCP_NODELAY`). This factor is consistent with the estimate in [Communication Patterns §3.2](./communication-patterns.md).

**Note on full-duplex advantage:** The coordinator's receive phase (workers-to-coordinator) and send phase (coordinator-to-workers) can overlap at the TCP stack level; the serialized analysis above is conservative.

### 6.4 Comparison with MPI

| Metric                      | Ferrompi (InfiniBand HDR) | TCP (100 Gbps Ethernet) | Ratio         |
| --------------------------- | ------------------------: | ----------------------: | ------------- |
| Link bandwidth              |                   25 GB/s |               12.5 GB/s | 2x            |
| Protocol overhead           |                      ~50% |                    ~50% | comparable    |
| Topology                    | Tree/ring (MPI-optimized) |      Star (coordinator) | MPI advantage |
| Est. overhead per iteration |                    ~46 ms |                 ~112 ms | ~2.4x         |
| Fraction of 5s iteration    |                     ~0.9% |                   ~2.2% | both viable   |

The TCP backend is approximately 2-3x slower than MPI for pure communication, primarily due to the star topology requiring the coordinator to relay all data and the lower bandwidth of Ethernet versus InfiniBand. However, at **< 200 ms/iteration** overhead, which is **< 5% of a typical 5-second iteration**, the TCP backend is viable for production-scale SDDP. The LP solver dominates iteration time, and the communication overhead is well within acceptable bounds.

## 7. SharedMemoryProvider

The TCP backend implements `SharedMemoryProvider` using the `HeapFallback` strategy defined in [Communicator Trait §4.4](./communicator-trait.md). TCP-connected processes run in separate address spaces (potentially on different physical nodes), so true intra-node shared memory is not available. Each rank allocates its own `Vec<T>` for shared memory regions.

### 7.1 Implementation

The TCP backend reuses the `HeapRegion<T>` type defined by the local backend, as both backends share the same `HeapFallback` semantics:

```rust
#[cfg(feature = "tcp")]
impl SharedMemoryProvider for TcpBackend {
    type Region<T: CommData> = HeapRegion<T>;

    fn create_shared_region<T: CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, CommError> {
        Ok(HeapRegion {
            data: vec![T::default(); count],
        })
    }

    fn split_local(&self) -> Result<Box<dyn Communicator>, CommError> {
        // Each TCP rank is its own "node" for shared memory purposes.
        Ok(Box::new(LocalBackend))
    }

    fn is_leader(&self) -> bool {
        // Always true: HeapFallback requires every rank to populate
        // its own copy (see [Communicator Trait §4.4]).
        true
    }
}
```

The `HeapFallback` behavior summary for the TCP backend matches the canonical table in [Communicator Trait §4.4](./communicator-trait.md):

| Method                 | HeapFallback Behavior                                       |
| ---------------------- | ----------------------------------------------------------- |
| `create_shared_region` | Allocates `Vec<T>` with `count` elements (per-process copy) |
| `is_leader`            | Always returns `true` (every rank is its own leader)        |
| `split_local`          | Returns a single-rank communicator (rank 0 of size 1)       |
| `as_slice`             | Returns `&self.data[..]` (local heap memory)                |
| `as_mut_slice`         | Returns `&mut self.data[..]` (local heap memory)            |
| `fence`                | No-op (returns `Ok(())`)                                    |
| `Drop`                 | Drops inner `Vec<T>`                                        |

**Memory footprint:** With `HeapFallback`, every rank replicates shared data. For $R = 4$ ranks with ~20.8 MB of shareable data (opening tree + case data), the total footprint is ~83.2 MB versus ~20.8 MB with true shared memory -- a 4x overhead. This is acceptable for the TCP backend's target deployment scenarios (containers, Python orchestration), where memory is less constrained than on shared-node HPC clusters. See [Communicator Trait §4.4](./communicator-trait.md) for the full comparison.

## 8. Configuration

### 8.1 Environment Variables

The TCP backend is configured via environment variables following the `COBRE_` prefix convention from [Backend Registration and Selection §3.2](./backend-selection.md).

| Variable                 | Required | Default  | Description                                                                                                |
| ------------------------ | -------- | -------- | ---------------------------------------------------------------------------------------------------------- |
| `COBRE_TCP_COORDINATOR`  | Yes      | _(none)_ | Hostname or IP address of the coordinator (rank 0). Workers connect to this address during initialization. |
| `COBRE_TCP_PORT`         | No       | `29500`  | TCP port on which the coordinator listens. All ranks must use the same port value.                         |
| `COBRE_TCP_RANK`         | Yes      | _(none)_ | Rank index of this process (`0..COBRE_TCP_SIZE`). Rank 0 is the coordinator; all others are workers.       |
| `COBRE_TCP_SIZE`         | Yes      | _(none)_ | Total number of ranks in the communicator. Must be consistent across all processes.                        |
| `COBRE_TCP_TIMEOUT_SECS` | No       | `60`     | Timeout in seconds for collective operations and connection attempts. See SS5.1 for failure detection.     |

**Coordinator address resolution:** The `COBRE_TCP_COORDINATOR` value is resolved via standard DNS or IP address parsing. For container deployments, this is typically the coordinator container's hostname or service name (e.g., `cobre-rank0`, `10.0.0.1`). The coordinator process itself does not read `COBRE_TCP_COORDINATOR` -- it binds to `0.0.0.0:COBRE_TCP_PORT`.

**Rank assignment:** Ranks are assigned externally by the orchestrator (Python script, container scheduler, or shell script). Each process must be started with a unique `COBRE_TCP_RANK` in `0..COBRE_TCP_SIZE`. The coordinator must have `COBRE_TCP_RANK=0`.

### 8.2 Example Invocation

```bash
# Start coordinator (rank 0) on host cobre-coord
COBRE_COMM_BACKEND=tcp \
COBRE_TCP_RANK=0 \
COBRE_TCP_SIZE=4 \
COBRE_TCP_PORT=29500 \
cobre run /path/to/case &

# Start workers on other hosts
COBRE_COMM_BACKEND=tcp \
COBRE_TCP_COORDINATOR=cobre-coord \
COBRE_TCP_PORT=29500 \
COBRE_TCP_RANK=1 \
COBRE_TCP_SIZE=4 \
cobre run /path/to/case &

COBRE_COMM_BACKEND=tcp \
COBRE_TCP_COORDINATOR=cobre-coord \
COBRE_TCP_PORT=29500 \
COBRE_TCP_RANK=2 \
COBRE_TCP_SIZE=4 \
cobre run /path/to/case &

COBRE_COMM_BACKEND=tcp \
COBRE_TCP_COORDINATOR=cobre-coord \
COBRE_TCP_PORT=29500 \
COBRE_TCP_RANK=3 \
COBRE_TCP_SIZE=4 \
cobre run /path/to/case &
```

### 8.3 Feature Gating

The TCP backend is gated behind the `tcp` Cargo feature flag:

```toml
[features]
default = []
tcp = []  # No external dependencies -- uses std::net only

[dependencies]
# No tcp-specific dependencies
```

When `tcp` is not enabled, the `TcpBackend` type does not exist. The `create_communicator()` factory function ([Backend Registration and Selection §4](./backend-selection.md)) does not include a TCP branch, and `COBRE_COMM_BACKEND=tcp` produces an error listing available backends.

When `tcp` is enabled, the TCP backend is available for selection via the factory function or `COBRE_COMM_BACKEND=tcp`. In the auto-detection priority chain ([Backend Registration and Selection §2.2](./backend-selection.md)), the TCP backend is selected when `COBRE_TCP_COORDINATOR` is set and the `mpi` backend was not selected first.

### 8.4 Build Profile Integration

The TCP backend is included in the following build profiles from [Backend Registration and Selection §1.3](./backend-selection.md):

| Build Profile | Includes `tcp`? | Rationale                                                        |
| ------------- | :-------------: | ---------------------------------------------------------------- |
| CLI / HPC     |       No        | MPI is the production communication layer for cluster deployment |
| Python wheel  |       Yes       | TCP enables multi-process Python orchestration without MPI       |
| Test / CI     |       No        | Only `local` backend; no external dependencies                   |
| Development   |       Yes       | All backends compiled for testing                                |

## Cross-References

- [Communicator Trait §1](./communicator-trait.md) -- `Communicator` trait definition, `CommData`, `ReduceOp`, `CommError` type definitions implemented by this backend
- [Communicator Trait §2](./communicator-trait.md) -- Method contracts (preconditions, postconditions, determinism guarantees) that this backend preserves via the coordinator-mediated protocols in SS3
- [Communicator Trait §3](./communicator-trait.md) -- Generic parameterization pattern (`train<C: Communicator>`) enabling compile-time monomorphization of this backend
- [Communicator Trait §4](./communicator-trait.md) -- `SharedMemoryProvider` trait and `HeapFallback` semantics (§4.4) used by this backend for shared memory regions
- [Communicator Trait §4.4](./communicator-trait.md) -- `HeapFallback` canonical behavior table: per-process `Vec<T>`, `is_leader` always `true`, `fence` is no-op
- [Communication Patterns §1.1](./communication-patterns.md) -- Three collective operations per iteration (allgatherv x2, allreduce x1), plus initialization-only broadcast and barrier
- [Communication Patterns §1.2](./communication-patterns.md) -- No point-to-point messaging in SDDP; collectives only -- the design constraint that makes the coordinator pattern sufficient
- [Communication Patterns §2](./communication-patterns.md) -- Data payloads for trial points (§2.1), cuts (§2.2), and convergence statistics (§2.3) transmitted through the TCP backend
- [Communication Patterns §3](./communication-patterns.md) -- Communication volume analysis (~587 MB/iteration) and bandwidth requirements on 100 Gbps Ethernet
- [Communication Patterns §6.1](./communication-patterns.md) -- Deterministic rank-ordered receive invariant preserved by the coordinator's rank-ordered assembly in SS3.1
- [Communication Patterns §6.2](./communication-patterns.md) -- Floating-point reduction tolerance; the TCP backend's sequential reduction is stricter than MPI's implementation-defined tree
- [Backend Registration and Selection §1.2](./backend-selection.md) -- Feature flag matrix; `tcp` feature gates this backend with no external dependencies
- [Backend Registration and Selection §2.2](./backend-selection.md) -- Auto-detection priority chain; TCP is selected when `COBRE_TCP_COORDINATOR` is set and MPI was not detected
- [Backend Registration and Selection §3.1](./backend-selection.md) -- Per-backend environment variable table for the TCP backend
- [Backend Registration and Selection §4](./backend-selection.md) -- Factory pattern returning `TcpBackend` in the `CommBackend` enum for multi-feature builds
- [Backend Registration and Selection §5.1](./backend-selection.md) -- Programmatic backend selection via `TcpConfig` for library-mode callers
- [Solver Abstraction §10](../architecture/solver-abstraction.md) -- Compile-time selection pattern via generic parameters and Cargo feature flags; the architectural precedent for this backend's feature-gated design
- [Python Bindings §7](../interfaces/python-bindings.md) -- TCP backend as the enabler for distributed Python SDDP execution without MPI
- [Hybrid Parallelism §1.0a](./hybrid-parallelism.md) -- Single-process mode; the TCP backend provides a multi-process alternative for environments where MPI is unavailable
