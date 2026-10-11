# Changelog

## 2026-10-11

- First version: the 6-bit MTP checkpoint (`TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP`, 147 GiB) on TensorFold v1.0.7's
  native Metal engine, greedy by default (the engine's only Flash Next decode), `--learn` available. A stdlib
  `python3 -m mac` orchestrator (settings and guards, a pinned engine build in a worktree, the checkpoint found in the
  Hugging Face cache or downloaded, a memory floor check, a detached server with a pid file and readiness on
  `/v1/models`, a launchd unit), the Spark recipe's quality suite and ruler run from a pinned clone, tests for the guards.
  Supersedes an unpublished 4-bit draft of 2026-10-09, whose checkpoint v1.0.2 served as gibberish and v1.0.7 refuses.
  Not yet measured.
