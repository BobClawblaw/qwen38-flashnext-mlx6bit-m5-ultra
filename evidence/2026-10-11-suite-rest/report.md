| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| ifeval | 134/150 | 89.3% | 455s | loose 134/150, strict 128/150, 150 covered prompts |
| gsm8k | 242/250 | 96.8% | 472s | misses at [12, 85, 93, 119, 163, 184, 234, 242] |
| mgsm | 225/240 | 93.8% | 726s | de 29/30, es 29/30, fr 26/30, ja 28/30, zh 28/30, ru 30/30, sw 28/30, bn 27/30 |
| mmlu | 282/342 | 82.5% | 847s | 57 subjects x 6; weakest college_mathematics 1/6, business_ethics 3/6, college_chemistry 3/6 |
| repetition | 8/8 | 100.0% | 52s | repeated 4-grams 0.21% mean, max 0.64%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 443s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 3277 tok/s |
| concurrent | 12/12 | 100.0% | 22s | 4 streams together vs alone |
| snapshot | 12/12 | 100.0% | 11s | replies written to /Users/xian/github/qwen38-flashnext-mlx6bit-m5-ultra/evidence/2026-10-11-suite-rest/replies.json |
