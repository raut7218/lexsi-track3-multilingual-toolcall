# Step 4 SFT report

SFT on 10,000 rebuilt MASSIVE examples raised the share of rows correct on our 55 functions by 44 to 57 points on each of the four test sets built from them, including the one in 4 languages outside training. It cut BFCL, the only test with unfamiliar functions, from 58.1% to 32.9% of rows correct. The BFCL loss comes from the training targets: no example has more than one call, and every argument value is a string.

Every number below comes from saved files in `GauravRaut/lexsi-track3-workspace`: `03_preds/sft/`, `03_preds/sft_dev/`, `03_preds/base/` and `04_models/sft/`. The tables come from `python code/tables.py sft` and `python code/tables.py base`.

## Runs

| Run | Job | What | Time | Cost at $0.40/h |
|---|---|---|---|---|
| SFT, first launch | `6ae97699c78863f9c631649b` | Reached `checkpoint-125`, then cancelled when HF credit ran out | n/a | n/a |
| SFT, relaunch | `6ae98cf30e0338d1bb7efb34` | Resumed from `checkpoint-125`, finished 625 steps | 4.78 h of training (17,191 s), plus model load | ~$1.91 for training alone |
| Dev evals | 3 jobs | `sft_dev/step{250,500,625}`, 2,300 dev rows each | not recorded here | not recorded here |
| Full eval | `6ae9e64bc78863f9c631cc6b` | `checkpoint-625`, 39 files, training prompt | 13:49–18:25 UTC, ~4.6 h | ~$1.84 |

Setup: Tiny Aya fp16, LoRA r16, effective batch 16, constant LR 2e-4 after 20 warmup steps, one pass over the first 1,000 rows of each of 10 languages (15.5% no-call). Peak GPU memory 11.41 GB.

**Checks passed:** the 100 code tests at the start of every job; 39 of 39 eval files with reply count equal to item count; no reply hit the 512-token cap on any SFT file; the resume continued from step 125 with no restart.

## Training loss

`trainer_state.json` at `checkpoint-625`, logged every 10 steps:

| Step | 10 | 50 | 100 | 200 | 250 | 300 | 400 | 500 | 600 | 620 |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss | 0.554 | 0.156 | 0.139 | 0.145 | 0.085 | 0.107 | 0.080 | 0.094 | 0.097 | 0.073 |

From step 100 on, loss stays between 0.073 and 0.163 and token accuracy between 93.6% and 97.8%. Loss falls fast and then flattens, so most of the format was learned in the first 100 steps (1,600 examples). The reported `train_loss` of 0.107 covers only the relaunch, steps 126–625.

## Checkpoint choice on dev

Dev score = share of all dev rows correct (call and no-call), mean of 10 languages. The rule was set before any results.

| Step | Examples seen | Dev score | AST strict | No-call correct | Lowest language | Highest language |
|---|---|---|---|---|---|---|
| 250 | 4,000 | 52.75 | 59.24 | 16.00 | ar-SA 46.0 | en-US 58.0 |
| 500 | 8,000 | 65.25 | 65.82 | 62.00 | vi-VN 60.5 | en-US 72.0 |
| 625 | 10,000 | **66.35** | 64.06 | 79.33 | vi-VN 59.0 | en-US 73.0 |

Step 625 beat step 500 by 1.10, which met the "ask before extending" rule. All of the gain came from declines (no-call 62.0 → 79.3), while AST strict fell 1.8. The user chose to stop and evaluate `checkpoint-625` (`decisions.md`, 2026-11-03).

## SFT against base

Training prompt for both. BFCL at 512 new tokens for both. "All rows correct" pools call and no-call rows. AST strict and no-call are means over files, as `tables.py` prints them.

| Test set | Rows | All rows correct, base → SFT | AST strict, base → SFT | No-call correct, base → SFT |
|---|---|---|---|---|
| Shortlist, 10 training languages | 5,880 | 8.6 → **65.8** | 8.7 → 64.5 | 7.8 → 73.4 |
| Full catalogue, 10 languages | 1,000 | 3.3 → **47.1** | 3.3 → 47.1 | n/a |
| English instructions, foreign requests | 1,000 | 3.0 → **47.2** | 2.0 → 40.1 | 8.7 → 87.3 |
| Held-out languages (sw, am, te, km) | 2,162 | 9.1 → **55.8** | 7.9 → 55.0 | 15.5 → 61.3 |
| BFCL en, unfamiliar functions | 1,240 | 58.1 → **32.9** | 60.3 → 23.4 | 34.2 → 54.6 |

### By language, shortlist (AST strict, 95% Wilson CI)

| Language | Base | SFT | SFT no-call |
|---|---|---|---|
| en-US | 9.6 (7.3–12.5) | 71.4 (67.3–75.2) | 79.5 |
| fr-FR | 9.0 (6.8–11.8) | 65.6 (61.3–69.6) | 77.3 |
| es-ES | 8.8 (6.6–11.6) | 66.0 (61.7–70.0) | 77.3 |
| zh-CN | 8.6 (6.4–11.4) | 65.8 (61.5–69.8) | 69.3 |
| ja-JP | 10.2 (7.8–13.2) | 66.0 (61.7–70.0) | 70.5 |
| ar-SA | 10.2 (7.8–13.2) | 58.8 (54.4–63.0) | 70.5 |
| hi-IN | 9.6 (7.3–12.5) | 63.2 (58.9–67.3) | 70.5 |
| id-ID | 7.0 (5.1–9.6) | 64.8 (60.5–68.9) | 75.0 |
| vi-VN | 6.4 (4.6–8.9) | 62.2 (57.9–66.3) | 70.5 |
| tr-TR | 7.6 (5.6–10.3) | 61.0 (56.7–65.2) | 73.9 |

### Held-out languages (AST strict, 95% Wilson CI)

| Language | Base | SFT | SFT no-call |
|---|---|---|---|
| sw-KE | 8.9 (6.7–11.8) | 51.7 (47.2–56.1) | 48.8 |
| am-ET | 8.7 (6.5–11.5) | 51.6 (47.2–56.0) | 65.1 |
| te-IN | 8.1 (6.0–10.8) | 59.1 (54.7–63.4) | 72.4 |
| km-KH | 5.8 (3.8–8.7) | 57.6 (52.5–62.6) | 58.8 |

These languages share our functions, so the claim is "outside our training", not "unseen by the model" (`decisions.md`, Step 1).

### BFCL at 512 new tokens

| Category | Base AST strict | SFT AST strict | Base right function | SFT right function |
|---|---|---|---|---|
| simple | 77.8 (73.4–81.6) | 44.8 (40.0–49.6) | 95.2 | 97.2 |
| multiple | 81.0 (75.0–85.8) | 44.0 (37.3–50.9) | 95.5 | 97.0 |
| parallel | 38.5 (32.0–45.4) | 1.0 (0.3–3.6) | 50.5 | 2.0 |
| parallel_multiple | 44.0 (37.3–50.9) | 4.0 (2.0–7.7) | 58.0 | 6.5 |
| irrelevance (no-call) | 34.2 | 54.6 | n/a | n/a |

The intervals are unpaired Wilson intervals. Every base-vs-SFT difference above is several times the interval width, so paired tests were not computed.

### Errors by pass, SFT (% of rows)

| Pass | Rows | correct | invalid_value | missing_param | over_call | unexpected_param | wrong_count | wrong_function |
|---|---|---|---|---|---|---|---|---|
| catalogue_training | 1000 | 47.1 | 6.1 | 6.5 | 0.0 | 12.4 | 9.5 | 18.4 |
| english_training | 1000 | 47.2 | 24.3 | 6.9 | 1.9 | 9.9 | 2.3 | 7.5 |
| heldout_training | 2162 | 55.8 | 8.8 | 4.9 | 5.8 | 12.4 | 1.2 | 10.9 |
| shortlist_training | 5880 | 65.8 | 6.5 | 5.7 | 4.0 | 10.4 | 1.2 | 6.4 |

For base, `unexpected_param` was 43.8–52.5% of rows on these passes (`baseline_report.md`). SFT cut it to 10–12%, which accounts for most of the gain. `english_training` keeps a high `invalid_value` (24.3%, base 8.3%); this was not looked into yet.

## Why BFCL dropped

**What the SFT targets contain** (first 1,000 rows of each of the 10 `sft_*` files, the exact training pool):

- **Calls per example:** 8,448 have one call, 1,552 have none, **0 have two or more**.
- **Value types:** every gold value is a list of text spans, and `prompts.target` writes the first one. So **every argument value the model was trained on is a string**: no numbers, booleans, lists or objects.
- **Arguments per call:** in en-US, of 842 calls, 260 have no arguments, 380 have one, and 49 have three or more.

**What changed in the BFCL replies:**

| | simple, base | simple, SFT | multiple, base | multiple, SFT |
|---|---|---|---|---|
| Arguments per call | 2.73 | 2.36 | 2.66 | 2.35 |
| String values | 51% | 76% | 52% | 74% |
| Integer values | 33% | 16% | 33% | 16% |
| Float + boolean values | 8% | 2% | 7% | 3% |
| `invalid_value` rows | 63 | 146 | 26 | 80 |
| `missing_param` rows | 2 | 64 | 3 | 23 |
| `wrong_count` rows | 14 | 0 | 2 | 0 |

The model still picks the right function (97% on simple and multiple) and still writes valid JSON (97.1% format valid). What it lost matches the training targets one for one:

1. **One call only:** on parallel and parallel_multiple, 186/200 and 185/200 rows are `wrong_count`.
2. **Strings for everything:** typed values (`"adults": "two"`, `"start_date": "March 10th, 2023"`) fail the value check.
3. **Fewer arguments:** required parameters get left out.

Length is ruled out: 0% of SFT BFCL replies hit the cap. A missing closing `]` explains only 3 of 13 unparseable simple replies, and the parser accepts most of those anyway.

## What this means

- **The project's main claim holds:** after SFT, Tiny Aya gets 47–66% of rows fully correct on the four test sets built from our tools, up from 3–9%. The gain carries over to the 4 languages outside training.
- **The cost is real and has a known cause.** The model learned MASSIVE's output style (one call, text-span values, few arguments) along with the task. This is a property of the data, so choosing an earlier checkpoint can at best reduce it: step 250 had already seen 4,000 such examples.
- **Fixing it needs examples with typed values and multiple calls,** either mixed into SFT or rewarded in GRPO. Both cost compute that the remaining HF credit (about $2.80 expected after this eval) does not cover. This is a decision for the user.
