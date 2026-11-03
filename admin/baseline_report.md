# Step 3 baseline report

Base Tiny Aya (`CohereLabs/tiny-aya-global`, fp16) on the full evaluation matrix.

**Run:** HF job `6ae87c710e0338d1bb7e07ae` on `t4-small`, at commit `e12f576`. It finished with status `COMPLETED` after 10.2 hours, about $4.08 at the listed rate. All 88 files and 28,444 items were scored. Replies and scores are in private dataset `GauravRaut/lexsi-track3-workspace`, under `03_preds/base/`. `python code/tables.py base` rebuilds every table below except the no-call split and the capped-reply counts, which were read from the saved replies.

AST is scored on call rows only, as in the paper. Group averages are the mean of the languages or categories, not pooled rows. Every percentage is from a saved run.

## Main results

| Pass | Files | Call rows | AST strict | AST range | Right function | Format | Strict format | No-call | Hit cap |
|---|---|---|---|---|---|---|---|---|---|
| Shortlist, training prompt | 10 | 5,000 | 8.7 | 6.4–10.2 | 79.0 | 99.8 | 93.8 | 7.8 | 2.8 |
| Shortlist, zero-shot | 10 | 5,000 | 7.1 | 5.4–8.6 | 50.9 | 69.8 | 67.8 | 36.2 | 0.7 |
| Shortlist, one-shot | 10 | 5,000 | 8.5 | 7.4–11.6 | 77.3 | 98.3 | 98.2 | 2.4 | 0.0 |
| English values, training prompt | 10 | 850 | 2.0 | 0.0–7.1 | 69.9 | 99.3 | 89.4 | 8.7 | 6.1 |
| Held-out, training prompt | 4 | 1,835 | 7.9 | 5.8–8.9 | 70.5 | 99.5 | 94.4 | 15.5 | 3.4 |
| Held-out, zero-shot | 4 | 1,835 | 6.5 | 5.2–7.5 | 40.5 | 61.3 | 57.9 | 43.9 | 0.9 |
| BFCL, training prompt | 5 | 1,000 | 47.1 | 11.5–81.0 | 56.8 | 98.9 | 75.8 | 35.0 | 18.1 |
| BFCL, training prompt, 512 tokens | 5 | 1,000 | **60.3** | 38.5–81.0 | 74.8 | 99.5 | 91.6 | 34.2 | 0.2 |
| BFCL, zero-shot | 5 | 1,000 | 35.2 | 18.0–47.5 | 45.8 | 54.6 | 56.2 | 36.2 | 1.6 |
| Catalogue, training prompt | 10 | 1,000 | 3.3 | 2.0–6.0 | 51.7 | 99.6 | 95.4 | n/a | 3.0 |
| Catalogue, zero-shot | 10 | 1,000 | 0.2 | 0.0–1.0 | 16.9 | 37.9 | 37.9 | n/a | 0.4 |
| Catalogue, one-shot | 10 | 1,000 | 7.2 | 6.0–9.0 | 64.0 | 98.7 | 98.7 | n/a | 0.0 |

## What the numbers say

1. **Tiny Aya often picks the right function but rarely gets the whole call right.** On the shortlist with the training prompt it picks the right function 79.0% of the time, yet only 8.7% of calls are fully correct.
2. **The main failure is extra arguments.** `unexpected_param` is 52.5% of all shortlist rows under the training prompt, and 44–52% in the other training-prompt and one-shot passes on MASSIVE data. Zero-shot is lower because broken syntax takes a share. In en-US, 320 of the 838 extra arguments on right-function replies were empty strings. BFCL's rule counts these as wrong, and we keep that rule.
3. **The base model almost never declines.** Only 7.2% of shortlist no-call rows under the training prompt got `[]` or plain text, and 92.2% got a call.
4. **Language isn't the main gap at this stage.** Shortlist AST under the training prompt runs from 6.4% (vi-VN) to 10.2% (ja-JP, ar-SA), and the 95% intervals all overlap. en-US, at 9.6%, isn't ahead. Held-out languages sit close to the same level, at 7.9%.
5. **BFCL single calls are far stronger than MASSIVE.** `simple` is 77.8% and `multiple` is 81.0%, against 9.6% for en-US on the shortlist. Both are English, so this gap comes from the task, not the language. It supports trying to teach the model restraint, but it doesn't show that SFT will work.
6. **The full catalogue is hard.** Listing all 55 functions drops right-function from 79.0% to 51.7% under the training prompt. Catalogue n is 100 per language, so one item is one point, and per-language figures are noisy.

## Two measurement problems

### The 128-token cap understated BFCL parallel scores, fixed by a rerun at 512

Under the training prompt, the base model writes pretty-printed JSON, so a list of calls ran past the 128-token cap. The parser then rescued only the first call in 169 of the 176 capped replies, so they failed on call count. All five BFCL categories were rerun under the training prompt at 512 new tokens, as pass `bfcl_training_512`, in HF job `6ae9728fc78863f9c6316290` at commit `dd9b21a`. The 128 files are kept. **BFCL in every later stage runs at 512, so base and trained models are compared at the same cap.**

| BFCL, training prompt | AST strict at 128 | AST strict at 512 | 95% CI at 512 | Hit cap at 128 | Hit cap at 512 | Strict format at 128 | Strict format at 512 |
|---|---|---|---|---|---|---|---|
| `simple` | 77.8 | 77.8 | 73.4–81.6 | 0.8 | 0.0 | 96.2 | 97.0 |
| `multiple` | 81.0 | 81.0 | 75.0–85.8 | 0.0 | 0.0 | 99.0 | 99.0 |
| `parallel` | 18.0 | **38.5** | 32.0–45.4 | 32.5 | 0.0 | 55.5 | 85.0 |
| `parallel_multiple` | 11.5 | **44.0** | 37.3–50.9 | 55.5 | 0.0 | 34.5 | 83.0 |
| `irrelevance` (no-call score) | 35.0 | 34.2 | n/a | 1.7 | 0.8 | 93.8 | 94.2 |

**The rerun is clean.** Every one of the 1,057 replies that had stopped under 128 tokens is byte-identical at 512. That holds because decoding is greedy and the batch order is the same, so the cap can only change where a reply ends. Only the other 183 replies, the ones capped at 128, could change. 176 of them are in the two parallel files. The longest finished reply was 424 tokens, so 512 is enough.

**Most of the low parallel score was the cap.** `parallel` roughly doubles and `parallel_multiple` almost quadruples. In `parallel_multiple`, `wrong_count` falls from 172 rows to 80. Without this rerun, an SFT model writing compact JSON would have gained up to 20.5 and 32.5 points on these two categories from reply length alone.

**`irrelevance` moves the other way, for the same reason.** Four replies hit the cap at 128. Two of them were cut-off calls that failed to parse, which BFCL's rule credits as correct declines. At 512 they finish as calls and count as over-calls, so the no-call score drops by 2 of 240 rows. The other two still hit 512, and both are repetition loops: one counts Monopoly players and the other repeats a list of survey answers.

Shortlist rows hit the 128 cap 2.8% of the time, and almost all their gold is a single call, so MASSIVE stays at 128.

### The zero-shot no-call score is mostly failed calls

BFCL's rule counts any reply with no parsed call as a correct no-call, so a broken attempt at a call gets credit. We follow that rule, but it makes zero-shot look far better at declining than it is. This split reads each reply on a no-call row. A failed call is an unparsed reply containing `name(` or starting with `[` or `{`, which is a heuristic.

| Pass | Rows | Called | Failed call, counted correct | Empty list or text | No-call score |
|---|---|---|---|---|---|
| Shortlist, training prompt | 880 | 92.2 | 0.6 | 7.2 | 7.8 |
| Shortlist, zero-shot | 880 | 63.7 | 35.3 | 0.9 | 36.2 |
| Shortlist, one-shot | 880 | 97.6 | 1.7 | 0.7 | 2.4 |
| Held-out, training prompt | 327 | 85.0 | 0.6 | 14.4 | 15.0 |
| Held-out, zero-shot | 327 | 56.6 | 42.8 | 0.6 | 43.4 |

The BFCL `irrelevance` rows show the same thing: under zero-shot, all 87 uncalled replies are failed calls, and none is a real decline. Held-out rows here are pooled, so their scores differ slightly from the language means in the main table, 15.5 and 43.9.

Zero-shot format failures themselves are real. Most break the requested `[f(a=1)]` form, usually `f({"a": 1})`, positional values, or a quoted string inside the list.

## Against the paper

The closest comparison is the catalogue pass against Tables 5 and 7 of MASSIVE-Agents, in AST percent over our 10 languages. Llama 3.2 3B is the closest model in size. Nova Premier is the highest of the 21 on these languages.

| Language | Zero-shot: Tiny Aya | Llama 3.2 3B | Nova Premier | One-shot: Tiny Aya | Llama 3.2 3B | Nova Premier |
|---|---|---|---|---|---|---|
| en-US | 0.0 | 18.4 | 57.4 | 8.0 | 14.2 | 63.7 |
| fr-FR | 0.0 | 7.9 | 47.1 | 6.0 | 8.5 | 54.0 |
| es-ES | 1.0 | 9.5 | 45.8 | 6.0 | 8.4 | 53.7 |
| zh-CN | 0.0 | 4.8 | 37.8 | 7.0 | 4.8 | 42.0 |
| ja-JP | 1.0 | 6.3 | 43.2 | 9.0 | 8.4 | 41.6 |
| ar-SA | 0.0 | 4.2 | 36.1 | 8.0 | 1.6 | 45.0 |
| hi-IN | 0.0 | 4.9 | 57.3 | 7.0 | 6.0 | 54.6 |
| id-ID | 0.0 | 2.7 | 43.6 | 7.0 | 5.3 | 47.9 |
| vi-VN | 0.0 | 3.7 | 38.8 | 7.0 | 2.1 | 47.9 |
| tr-TR | 0.0 | 3.7 | 31.2 | 7.0 | 2.1 | 32.8 |
| **Mean of these 10** | **0.2** | **6.6** | **43.8** | **7.2** | **6.1** | **48.3** |

**This is context, not a ranking.** Our catalogue uses a compact function list, because the paper's full JSON runs to about 9,400 tokens and Tiny Aya's limit is 8,192. It also uses our rebuilt data and only 100 rows per language. The one-shot gap of 7.2 against 6.1 is too small to read anything into, and the zero-shot gap comes mostly from format failure: 62.1% of rows were `invalid_syntax`. The shortlist is an easier task with five tools, so its numbers must not go in the same table.

## Errors by pass

Percent of all rows, including no-call rows.

| Pass | Correct | Invalid syntax | Invalid value | Missing param | Over-call | Unexpected param | Wrong count | Wrong function |
|---|---|---|---|---|---|---|---|---|
| Shortlist, training prompt | 8.6 | 0.1 | 3.6 | 3.7 | 13.8 | 52.5 | 4.0 | 13.7 |
| Shortlist, zero-shot | 11.4 | 25.6 | 1.9 | 0.9 | 9.5 | 34.5 | 4.1 | 11.9 |
| Shortlist, one-shot | 7.6 | 1.4 | 4.5 | 2.1 | 14.6 | 51.9 | 0.0 | 17.9 |
| English values | 3.0 | 0.6 | 8.3 | 3.8 | 13.7 | 45.6 | 5.6 | 19.4 |
| Held-out, training prompt | 9.1 | 0.4 | 3.1 | 6.2 | 12.9 | 44.2 | 5.5 | 18.7 |
| Held-out, zero-shot | 12.1 | 32.4 | 1.1 | 0.9 | 8.6 | 27.2 | 3.7 | 14.1 |
| Catalogue, training prompt | 3.3 | 0.4 | 4.6 | 0.0 | 0.0 | 43.8 | 15.1 | 32.8 |
| Catalogue, zero-shot | 0.2 | 62.1 | 0.1 | 0.0 | 0.0 | 16.6 | 1.6 | 19.4 |
| Catalogue, one-shot | 7.2 | 1.3 | 8.4 | 2.3 | 0.0 | 46.1 | 0.0 | 34.7 |

## By language

Shortlist with the training prompt, AST strict on 500 call rows per language, then held-out languages.

| Language | AST strict | 95% CI | AST verbatim | AST lenient | Right function | No-call | Hit cap |
|---|---|---|---|---|---|---|---|
| en-US | 9.6 | 7.3–12.5 | 7.8 | 10.4 | 80.8 | 8.0 | 0.5 |
| fr-FR | 9.0 | 6.8–11.8 | 7.4 | 10.0 | 84.2 | 2.3 | 2.4 |
| es-ES | 8.8 | 6.6–11.6 | 8.4 | 8.8 | 81.6 | 2.3 | 2.6 |
| zh-CN | 8.6 | 6.4–11.4 | 8.6 | 8.6 | 85.0 | 2.3 | 1.9 |
| ja-JP | 10.2 | 7.8–13.2 | 10.2 | 10.2 | 80.6 | 10.2 | 2.2 |
| ar-SA | 10.2 | 7.8–13.2 | 10.0 | 10.2 | 74.6 | 0.0 | 3.2 |
| hi-IN | 9.6 | 7.3–12.5 | 9.6 | 9.6 | 82.6 | 4.5 | 4.1 |
| id-ID | 7.0 | 5.1–9.6 | 6.4 | 7.6 | 73.0 | 25.0 | 1.7 |
| vi-VN | 6.4 | 4.6–8.9 | 4.8 | 6.4 | 69.6 | 19.3 | 4.4 |
| tr-TR | 7.6 | 5.6–10.3 | 6.6 | 8.2 | 78.0 | 4.5 | 4.8 |
| sw-KE (held out) | 8.9 | 6.7–11.8 | 8.3 | 8.9 | 66.9 | 9.3 | 1.9 |
| am-ET (held out) | 8.7 | 6.5–11.5 | 8.7 | 8.7 | 71.6 | 17.4 | 2.7 |
| te-IN (held out) | 8.1 | 6.0–10.8 | 8.1 | 8.1 | 80.6 | 10.3 | 3.4 |
| km-KH (held out) | 5.8 | 3.8–8.7 | 5.8 | 5.8 | 62.9 | 25.0 | 5.4 |

These intervals treat each language alone. The same items recur across languages, so pooled intervals need item-level resampling, which the SFT comparison will add.

## Time and budget

| Pass group | Items | Hours | Items/s |
|---|---|---|---|
| Shortlist, training prompt | 5,880 | 2.22 | 0.74 |
| Shortlist, zero-shot | 5,880 | 1.78 | 0.92 |
| Shortlist, one-shot | 5,880 | 1.42 | 1.15 |
| English values | 1,000 | 0.42 | 0.66 |
| Held-out, training prompt | 2,162 | 0.83 | 0.72 |
| Held-out, zero-shot | 2,162 | 0.71 | 0.85 |
| BFCL, training prompt | 1,240 | 0.40 | 0.85 |
| BFCL, zero-shot | 1,240 | 0.32 | 1.06 |
| Catalogue, training prompt | 1,000 | 0.73 | 0.38 |
| Catalogue, zero-shot | 1,000 | 0.59 | 0.47 |
| Catalogue, one-shot | 1,000 | 0.51 | 0.55 |

About $15.58 of credit remains, calculated from the $19.66 balance, which buys about 39 hours. The billing page wasn't re-read. At these speeds, the remaining plan needs about 21–24 hours, or about $8.50–$9.50:

| Work | Hours |
|---|---|
| SFT, 10,000 examples | 5.9 |
| GRPO and its test | 2.4–4.9 |
| Dev selection, 5 checkpoints of 2,000 items | 3.8 |
| Final evals, SFT and GRPO, 4.6 hours each | 9.2 |

SFT and GRPO times still rest on the short pilot, and pilot speed proved optimistic here, so expect some drift. A trained model that writes shorter replies would make evals faster.

## Open decisions

1. **BFCL cap:** closed 2026-11-03. The user chose to rerun all five categories at 512. Results are above.
2. **Step 4 SFT:** approved 2026-11-03, running.
