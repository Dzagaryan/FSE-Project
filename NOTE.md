# Development note

- A real shape bug was the old `TORCH_CHECK(x.dim() == 2)` gate: a one-dimensional
  input raised `RuntimeError` before the operation ran. Commit `bac5055`
  removed that gate. Commit `34ced98` removed the remaining dimension-specific
  size checks, completing support for arbitrary shapes. The current tests
  exercise one- and two-dimensional inputs.
- An integer input is unsupported. Calling `bifurcation_step` with integer `x`
  and float32 `r` raises `RuntimeError` in C++ because only float32 is accepted.
- For a reproducible large-input measurement, run
  `cd /tmp && /tmp/fse-smoke-try/bin/python -m cProfile -s cumulative /course/team-repo/tools/profile.py`
  after `./tools/smoke.sh` (adjust the repository path if needed). The workload makes ten calls on a tensor of
  1,000,000 values. Record the reported cumulative time for
  `bifurcation_step` here after running it in the course container.
