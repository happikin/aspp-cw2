# CUDA Implementation Notes and Decision Log

This file tracks design decisions for `src/wave_cuda.cu` and the evidence behind them.
Use this as source material for the report's "Design and process" and "Results" sections.

## 1. Coursework Constraints (Relevant to Design)

- Modify only one backend file for submission (`wave_cuda.cu` for CUDA path).
- Keep correctness against CPU reference.
- Target single node with up to 8 GPUs, MPI decomposition retained.
- All GPU work and communication must complete inside `run(int n)`.

## 2. Baseline Understanding

- Existing codebase is MPI-first: each rank owns one 3D subdomain with halo cells.
- Time loop pattern is `halo_exchange -> step -> advance`.
- CPU implementation is reference for correctness checks.

## 3. Decision Log

## D1. Keep MPI decomposition as outer parallel model

- Decision:
  Keep existing per-rank subdomain model and integrate CUDA within each rank.
- Why:
  This matches provided architecture, minimizes risk, and aligns with correctness/marking flow.
- Tradeoff:
  Multi-rank behavior depends on halo communication overhead.
- Evidence:
  Multi-rank correctness checks passed (`Number of differences detected = 0`) for tested `-mpi` layouts.

## D2. Map rank to GPU by node-local rank

- Decision:
  `device = local_rank % num_devices`.
- Why:
  Robust for 1..N GPUs, simple, deterministic, and compatible with single-node multi-GPU runs.
- Tradeoff:
  On 1-GPU machines with `-np > 1`, ranks contend on one GPU (correctness OK, performance mixed).
- Evidence:
  `-np 4` on single GPU still produced correct results.

## D3. Move compute kernel to CUDA (initial correctness-first port)

- Decision:
  Port stencil+damping update to CUDA kernel while retaining host-based halo orchestration.
- Why:
  Fast path to functional GPU backend with low implementation risk.
- Tradeoff:
  Initial approach copied full `u.now` field device<->host each timestep, creating major overhead.
- Evidence:
  Correctness passed, but early timing showed avoidable transfer cost.

## D4. Replace full-field copies with halo-only staged exchange

- Decision:
  Pack only 6 halo faces on GPU, transfer only those faces to host pinned buffers, MPI exchange, unpack back to GPU halos.
- Why:
  Removes dominant transfer bottleneck from full-field `D2H/H2D` each step.
- Tradeoff:
  More code complexity (pack/unpack kernels, extra buffers).
- Evidence:
  Correctness remained zero-diff across tested cases, with large runtime improvement on local GPU tests.

## D5. Split compute into interior and boundary kernels

- Decision:
  Launch interior compute first, then perform halo exchange, then launch boundary compute after halo-ready event.
- Why:
  Boundary points require fresh halos; interior points do not. This enables communication/computation overlap.
- Tradeoff:
  Additional kernels and stream/event synchronization complexity.
- Evidence:
  Post-change correctness still zero-diff, and large-shape performance remained strong.

## 4. Validation Summary (Local Machine)

- Correctness checks passed (`ndiff = 0`) for representative cases:
  - `-np 1`, `-mpi 1,1,1`, `-shape 64/128/512`
  - `-np 2`, `-mpi 2,1,1`
  - `-np 4`, `-mpi 2,2,1` and `1,2,2`
  - restart case using checkpoint input
- Note:
  Local machine has one GPU; final benchmark conclusions must come from A100/H100 nodes.

## 5. Performance Notes to Capture in Report

- Compare at least these stages:
  1. Initial CUDA kernel with full-field host round-trip
  2. Halo-only exchange
  3. Interior/boundary overlap
- For each stage, record:
  - runtime per chunk
  - site-updates/second
  - scaling vs MPI ranks/devices
  - correctness status

## 6. Remaining Work (Planned)

- Detect/enable CUDA-aware MPI path on target cluster if available.
- Build reproducible benchmark matrix on A100 and H100:
  - sizes 256..2500
  - device/rank counts 1,2,4,8
  - multiple `-mpi` layouts per rank count
- Export JSON per run and generate plots/tables for report.

## 7. Update Template (Append for New Decisions)

Copy this block and append when making a new optimization:

```text
## D?. <short title>
- Decision:
- Why:
- Tradeoff:
- Evidence (command + key result):
- Keep / Revert:
```
