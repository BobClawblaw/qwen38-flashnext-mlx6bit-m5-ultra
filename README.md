# Qwen3.8-Flash-Next MLX 6-bit on one M5 Ultra

Serve [TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP),
a uniform MLX affine 6-bit (group 32) conversion of Qwen3.8-Flash-Next with its native MTP draft block, on one Apple
M5 Ultra (256 GB) with the [TensorFold](https://github.com/ashhart/TensorFold) native Metal engine (v1.0.7): the
checkpoint's full 262,144-token window, MTP drafts, kept prompt states between turns, learned shared prefixes on disk,
an OpenAI-compatible API (chat, completions, Responses; Anthropic Messages too), a launchd unit.

Why 6-bit: it is the one Flash Next format the native Metal engine serves. Since v1.0.7 the engine refuses any other
affine width or group at load (`Flash Next runtime kernels require the supported 6-bit affine, group_size 32 format`),
and it is the format TensorFold qualifies on an M5 Ultra. The engine reads the checkpoint as published (`tensorfold-native
info` reports `format: 6-bit, group 32`, weights 147.25 GiB, memory floor 149.25 GiB) and builds its weight packs under
`~/.cache/tensorfold/packs` on the first start, so there is nothing to convert.

The recipe is the one-Mac sibling of
[qwen38-flashnext-int4mixed-2x-dgx-sparks](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks) and
of [glm53-flash-mlx4bit-m5-ultra](https://github.com/BobClawblaw/glm53-flash-mlx4bit-m5-ultra), built to the same
rules: a stdlib Python orchestrator, nothing copied, no harness dependencies. Its quality suite and decode ruler run
against this server from a pinned clone (`python3 -m mac suite|ruler`).

## Measured on this Mac

Mac Studio M5 Ultra, 256 GB, TensorFold v1.0.7 (`e7c523eb`), the checkpoint at `45032a19`, greedy, thinking off, one
request at a time, measured 2026-10-11. The model loads in 36-46 s with its packs built and the files in the page cache.

Decode with `bench/ruler.py`: 256-token replies, a distinct prompt per phase, median of 3 runs
([`evidence/2026-10-11-first-ruler/ruler-1user.json`](evidence/2026-10-11-first-ruler/ruler-1user.json)); two
back-to-back runs agreed within 1 tok/s:

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 144.5 | 206.2 | 172.3 | 194.3 | 0.10-0.20 s |

Tokens a second, MTP drafts on. The suite's `drafts` check measured the same prompts undrafted at 84 tok/s against 143
drafted, with every drafted reply byte-equal to its undrafted one. `FZ_LANE=1 FZ_GDN=2`, the speed-up recipe's decode
kernels, measured 4-5% slower on every phase here (137.7 / 199.8 / 165.2 / 187.3,
[`evidence/2026-10-11-fz-kernels/`](evidence/2026-10-11-fz-kernels/)) and is not set.

Prompt passes with [`tools/prompt_pass.py`](tools/prompt_pass.py): one request each, distinct text so the cache resumes
none of it, a needle near the start found in every reply
([`evidence/2026-10-11-prompt-pass/prompt_pass.json`](evidence/2026-10-11-prompt-pass/prompt_pass.json)):

| Prompt | Time to first token | Rate |
|---:|---:|---:|
| 4,108 tokens | 3.1 s | 1,330 tok/s |
| 16,794 | 4.0 s | 4,253 tok/s |
| 68,869 | 15.6 s | 4,411 tok/s |
| 139,000 | 34.1 s | 4,071 tok/s |
| 261,221 (the window, less the reply) | 71.7 s | 3,643 tok/s |

Quality with the Spark recipe's suite, thinking off, every check it has but vision (not served here):
[`evidence/2026-10-11-first-suite/`](evidence/2026-10-11-first-suite/report.md),
[`evidence/2026-10-11-suite-rest/`](evidence/2026-10-11-suite-rest/report.md),
[`evidence/2026-10-11-humaneval/`](evidence/2026-10-11-humaneval/report.md).

| Check | Result |
|---|---:|
| drafted equals undrafted (12 prompts) | 12/12 |
| concurrent: 4 streams together equal each alone (queued, one lane) | 12/12 |
| streaming equals non-streamed | 4/4 |
| multi-turn recall | 8/8 facts |
| stop strings | 3/3 |
| thinking on: reasoning_content, no think tags in content | 10/10 |
| copy, byte-identical | 2/2 |
| max_tokens exact | 3/3 |
| tool calls (exact function and arguments, optional arguments, parallel calls, no-call cases) | 59/60 |
| JSON mode, valid | 30/30 |
| IFEval (150 prompts, 24 instruction types) | 134 loose / 128 strict |
| GSM8K (250) | 242 (96.8%) |
| MGSM (8 languages, 30 each) | 225/240 (93.8%) |
| MMLU (57 subjects, 6 each) | 282/342 (82.5%) |
| HumanEval, pass@1 greedy (Docker sandbox) | 157/164 (95.7%) |
| repetition (8 long replies) | 0.21% repeated 4-grams, none hit max_tokens |
| long-context: needles to 224k, spread facts at 28k/57k/114k/225k | 55/55 |

The one tool miss: asked to "set a timer for 5 minutes", the model added a `label` argument the schema marks optional.
GLM-5.3-Flash on this Mac, with thinking on, scored IFEval 131, GSM8K 248, MMLU 308 of 342 on the same suite; with
thinking off, 96, 246 and 292. A thinking-on run of this recipe is not done yet.

### Four streams at once: upstream PR #606

Flash Next on Metal serves one reply at a time in v1.0.7; concurrent requests queue. The open upstream PR
[ashhart/TensorFold#606](https://github.com/ashhart/TensorFold/pull/606) (cwschroeder, continuous batching through the
shared lane core, `--parallel N`) was merged onto `main` `e7c523eb` (three small conflicts) and run on this Mac with
this checkpoint, the same ruler at 1 and 4 concurrent requests
([`evidence/2026-10-11-pr606-parallel/`](evidence/2026-10-11-pr606-parallel/)):

| Build | Streams | prose | code | structured | list | first token |
|---|---:|---:|---:|---:|---:|---:|
| #606, `--parallel 1` (the legacy driver), 1 user | 1 | 144.6 | 205.9 | 172.3 | 194.4 | 0.09-0.16 s |
| #606, `--parallel 4`, 1 user | 1 | 131.0 | 180.0 | 152.7 | 183.3 | 0.12-0.18 s |
| #606, `--parallel 4`, 4 users, each | 4 | 52.8 | 80.7 | 64.4 | 78.6 | 0.33-0.36 s |
| #606, `--parallel 4`, 4 users, aggregate | 4 | 213.8 | 321.8 | 254.2 | 314.8 | |

Tokens a second, 65,536-token window for the `--parallel 1` build and 16,384 for `--parallel 4`: at 65k the batching
engine refused four slots (`OverMemoryLimit`), since the engine's 70% cap on a 256 GB Mac leaves about 4 GiB past the
173 GiB in use once loaded. Four shared streams deliver about 1.5-1.6x the single stream's tokens, each user seeing a
first token in a third of a second instead of waiting its turn; the shared path costs a lone user 6-12%. On the shared
server the suite's exactness checks hold: four streams together equal each alone 12/12, drafted equals undrafted 12/12
(137 against 81 tok/s). Not in this
recipe: the PR is open, its author and ashhart are shaping it into the engine's default path with sampling, and this run
is a data point for that, not a release.

## Requirements

- One Apple Silicon Mac with 192 GB or more unified memory (the floor is 149.3 GiB plus the kept prompt states;
  qualified by TensorFold on an M5 Ultra), macOS with Xcode and its Metal toolchain, Zig 0.17.0 (the checkout's
  `.zig-version`), Python 3.11+ (the orchestrator is stdlib only), the `hf` CLI for the download
  (`pip install -U huggingface_hub`; the repository is public).
- Disk: the checkpoint, 147 GiB, plus the weight packs the engine writes under `~/.cache/tensorfold/packs`; the engine build.

## Quick start

```bash
git clone https://github.com/BobClawblaw/qwen38-flashnext-mlx6bit-m5-ultra.git && cd qwen38-flashnext-mlx6bit-m5-ultra
cp local.example.toml local.toml && $EDITOR local.toml   # port, paths (tensorfold_src, zig) if not the defaults
python3 -m mac validate            # the settings and the guards, touches nothing
python3 -m mac up                  # builds the engine once, finds or downloads the weights, waits for the API
python3 -m mac chat "Say hello in one sentence."
python3 -m mac status
PROFILE=reference python3 -m mac up   # --no-drafts: the exactness reference
python3 -m mac down
```

`up` builds `tensorfold-native` at the pinned commit in a worktree under `~/.cache/tensorfold-qwen38-mlx6bit/engine/`
(a few minutes), looks for the checkpoint at the pinned revision in the Hugging Face cache and downloads it otherwise,
checks the free memory against the engine's floor, starts the server detached with its log under `tf_cache`, and waits
until `/v1/models` lists `Qwen3.8-Flash-Next-MLX-6bit`. A start with the packs built and the files in the page cache
takes under a minute; the first one builds the packs.

Autostart: `launchd/com.tensorfold.qwen38-mlx6bit.plist` (install lines at its top).

## Settings

`recipe.toml` holds the defaults, `local.toml` your Mac's values, and any setting can be overridden by an environment
variable of the same name in upper case (`PORT=8090 python3 -m mac up`).

| Setting | Default | Notes |
|---|---|---|
| `profile` | `chat` | `chat`: MTP drafts; `reference`: `--no-drafts`, the plain reference for exactness checks |
| `context` | 262144 | the checkpoint's window; the Metal engine serves the plain rotary only, so no YaRN profile (the Spark recipe's 1M is CUDA) |
| `drafts` | true | the native MTP head's drafts, depth set by the engine's rule; text equal to undrafted by TensorFold's exactness gate |
| `thinking` | false | whether the chat template opens a reasoning block |
| `max_tokens` | 32768 | the reply cap when a request sets none |
| `temperature` | `"0"` | the engine decodes Flash Next greedily only and refuses a request that asks for a temperature; `"0"` makes the server's own default greedy so plain requests are served |
| `prompt_cache_gib` | -1 | kept prompt states: -1 the engine's default (what 70% of RAM leaves past the server, less 2 GiB), 0 off |
| `keep_warm` | 900 | seconds the GPU stays active after a request |
| `learn`, `learn_dir`, `learn_gib` | false, "", 0 | shared prefixes (a system prompt and its tools) kept on disk, so a restarted server resumes them (`--learn`, v1.0.7) |
| `compact_at` | `""` | context compaction: `auto` or a fraction (TensorFold's README) |
| `dashboard`, `api_key`, `request_log` | true, "", "" | `/dashboard` and `/stats`; a bearer key; `TENSORFOLD_REQUEST_LOG` file |
| `binary` | `""` | a prebuilt `tensorfold-native` to serve with instead of the pinned build |
| `memory_margin_gib` | 8 | `up` refuses when free plus inactive memory is under the floor plus this |
| `extra_args`, `extra_env` | | passed to the engine; flags the recipe owns, and flags of other families or platforms, are refused |

`python3 -m mac validate` refuses, before anything runs: a window above 262,144 or not a multiple of 256, zero-padded
or non-decimal integers, `learn_dir` without `learn`, a `binary` that is not a file, `extra_args` that re-set a flag the
recipe builds (`--port`, `--context`, `--no-drafts`, any prefix of them) or name a flag this engine has for other
families or CUDA only (`--parallel`, `--kv-dtype`, `--vision`, `--tool-system`, `--speed-up`), an `extra_env` entry that
is not `KEY=VALUE`.

## Quality suite and ruler

```bash
python3 -m mac ruler -- --runs 3 --json $PWD/evidence/my-run/ruler.json   # bench/ruler.py --url ... --model ... --concurrency 1
python3 -m mac suite -- --out $PWD/evidence/my-run                        # quality/suite.py; vision skipped unless --only
python3 -m mac suite -- --out $PWD/evidence/my-run --only drafts behaviour tools json
python3 -I tools/prompt_pass.py --model Qwen3.8-Flash-Next-MLX-6bit --json $PWD/evidence/my-run/prompt_pass.json
```

Both come from the Spark recipe at the commit `recipe.toml` pins (`spark_recipe_sha`), cloned under `.external/`
(git-ignored) and run with this server's URL and model name, so give them absolute output paths. Its `drafts` check
compares drafted replies with `"draft": false` ones; `concurrent` sends four streams, which this engine serves one at a
time, so it measures queueing, not sharing. HumanEval runs the model's code in a Docker sandbox: `--sandbox-image
python:3.12-slim` works; with Colima (`brew install docker colima`) set `TMPDIR` to a folder under your home, which is
what Colima shares with its VM. The suite downloads its datasets, so python.org's Python needs
`SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')`.

## Not supported

- One active reply at a time: Flash Next on Metal has one lane, so `--parallel` is not passed and concurrent requests
  queue. Two Macs at once is TensorFold's speed-up mode, a different recipe.
- Greedy decoding only: the native Flash Next engine has no sampler, and a request that sends a temperature is refused
  with a 400 before it is admitted.
- No images or video, no `--tool-system`, no grammars (`response_format`), no `--kv-dtype`: on Metal those are GLM's
  or the CUDA port's in v1.0.7.
- No YaRN: the checkpoint's 262,144 tokens is the window.
- No other precision: 4-bit (the `Vontra`/`TensorFold` 4-bit MTP checkpoint), the oQ family, 8-bit and mixed packs such
  as MTPLX's are refused by the engine at load. A faster mixed layout (4-bit experts, 8-bit attention) would be engine
  work upstream, not a recipe setting.

## Logs and evidence

`python3 -m mac logs [--tail N]` reads the server log under `tf_cache`; `python3 -m mac status` reads `/health`,
`/v1/models` and `/stats`. Every number this README will carry has a file under [`evidence/`](evidence/).

## License

MIT ([LICENSE](LICENSE)). The model is under the Qwen Community License 1.0 (its card); TensorFold is Apache-2.0.
