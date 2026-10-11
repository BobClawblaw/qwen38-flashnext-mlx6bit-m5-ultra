# Changelog

## 2026-10-11

- First version: the 6-bit MTP checkpoint (`TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP`, 147 GiB) on TensorFold v1.0.7's
  native Metal engine, greedy by default (the engine's only Flash Next decode), `--learn` available. A stdlib
  `python3 -m mac` orchestrator (settings and guards, a pinned engine build in a worktree, the checkpoint found in the
  Hugging Face cache or downloaded, a memory floor check, a detached server with a pid file and readiness on
  `/v1/models`, a launchd unit), the Spark recipe's quality suite and ruler run from a pinned clone, tests for the guards.
  Supersedes an unpublished 4-bit draft of 2026-10-09, whose checkpoint v1.0.2 served as gibberish and v1.0.7 refuses.
- Measured the same day (README, `evidence/`): one user 144.5 / 206.2 / 172.3 / 194.3 tok/s (prose / code / structured /
  list) with MTP drafts, 84 undrafted, every drafted reply byte-equal; prompts read at 3,600-4,400 tok/s to the full
  262k window with every needle found; the whole quality suite with thinking off (IFEval 134/150, GSM8K 242/250, MGSM
  225/240, MMLU 282/342, HumanEval 157/164, tools 59/60, JSON 30/30, long-context 55/55). `FZ_LANE=1 FZ_GDN=2` measured
  4-5% slower and is not set. Upstream PR #606's shared lane path, merged onto main and run here, serves four streams at
  1.5-1.6x the single stream's aggregate with exactness intact; it stays out of the recipe until it lands upstream.
  `tools/prompt_pass.py` added.
