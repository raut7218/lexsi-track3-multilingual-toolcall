# Step 1 data report

Built 2026-11-02 on the T4 from commit `74f0bf8`, with 73 of 73 tests passing there. Source numbers are on Drive in `01_data/build_counts.json` and `01_data/report_numbers.json`. Items are in `01_data/items/` (42 files), and `functions.json` has sha256 `12cf91a9…0375f73`.

This is a MASSIVE-Agents-style rebuild, not the paper's dataset: its manual filter list and function descriptions aren't published.

## Result

| Check | Result |
|---|---|
| Source | MASSIVE 1.1, 16,521 rows per locale (train 11,514, dev 2,033, test 2,974), same ids, intents agree 100% |
| Functions and arguments | 55 functions, 211 arguments, 51 names (paper: 55, 200, 53) |
| Rows in all 10 languages after filters | Train 7,494, dev 1,315, test 1,937 |
| Built per language | SFT 3,000, GRPO 1,000, dev 200 (30 no-call), test 588 (88 no-call) |
| Integrity | 0 schema faults, 0 tool-list faults, 0 partition faults, 0 SFT/GRPO overlap, 0 test/dev overlap |
| Parallel across languages | Test ids and tool lists identical in all 10 |
| Gold values in the test query | 100% in every language |
| Function coverage | Test 55, dev 54, SFT 55, GRPO 42–48 (see decision 3) |

## Rows removed per filter

Filters run in the order shown, so a row counts under the first one it fails.

| Locale | Open-ended intent | Duplicate argument | Rare argument | Judgment | Kept train / dev / test |
|---|---|---|---|---|---|
| en-US | 952 | 291 | 426 | no judgments | 10,357 / 1,828 / 2,667 |
| fr-FR | 952 | 323 | 406 | 915 | 9,712 / 1,715 / 2,498 |
| es-ES | 952 | 330 | 414 | 64 | 10,298 / 1,814 / 2,649 |
| zh-CN | 952 | 254 | 426 | 320 | 10,157 / 1,795 / 2,617 |
| ja-JP | 952 | 277 | 425 | 157 | 10,250 / 1,814 / 2,646 |
| ar-SA | 952 | 208 | 426 | 2,316 | 8,799 / 1,549 / 2,271 |
| hi-IN | 952 | 334 | 419 | 824 | 9,778 / 1,716 / 2,498 |
| id-ID | 952 | 319 | 421 | 0 | 10,346 / 1,821 / 2,662 |
| vi-VN | 952 | 323 | 421 | 388 | 10,072 / 1,771 / 2,594 |
| tr-TR | 952 | 232 | 427 | 230 | 10,237 / 1,814 / 2,629 |

The judgment filter is uneven: it removes 14% of ar-SA and none of id-ID. id-ID's votes are real (919 distinct vote patterns), and almost all are top scores. Failures per sub-check, over all rows, which can overlap:

| Locale | Intent | Slots | Grammar | Spelling | Language |
|---|---|---|---|---|---|
| fr-FR | 52 | 894 | 105 | 3 | 2 |
| es-ES | 2 | 61 | 1 | 0 | 1 |
| zh-CN | 162 | 104 | 184 | 1 | 1 |
| ja-JP | 54 | 95 | 26 | 0 | 7 |
| ar-SA | 2,037 | 328 | 2,344 | 1 | 575 |
| hi-IN | 277 | 94 | 644 | 1 | 15 |
| id-ID | 0 | 0 | 0 | 0 | 0 |
| vi-VN | 203 | 296 | 39 | 0 | 6 |
| tr-TR | 67 | 140 | 113 | 6 | 8 |

## English alignment in the 4,000 train rows

| Locale | Aligned with en-US | English-value policy | No-call | Functions in GRPO |
|---|---|---|---|---|
| en-US | 4,000 | 999 | 614 | 48 |
| fr-FR | 3,942 | 1,002 | 603 | 47 |
| es-ES | 3,918 | 976 | 601 | 48 |
| zh-CN | 3,993 | 994 | 586 | 48 |
| ja-JP | 3,978 | 980 | 597 | 48 |
| ar-SA | 3,983 | 984 | 608 | 42 |
| hi-IN | 3,985 | 994 | 601 | 48 |
| id-ID | 3,969 | 960 | 610 | 48 |
| vi-VN | 3,979 | 984 | 601 | 48 |
| tr-TR | 3,964 | 978 | 597 | 48 |

## Context fit, in real Tiny Aya tokens

The model's limit is 8,192 positions. The chat template adds 366 tokens by itself. Each cell is the worst case over 10 languages, using every test item for 5 tools and the 20 longest queries per language for 55 tools.

| Prompt | 5 tools | 55 full JSON | 55 compact |
|---|---|---|---|
| BFCL zero-shot | 1,751 | **9,409 (does not fit)** | 3,153 |
| Custom one-shot | 1,755 | **9,413 (does not fit)** | 3,157 |
| Training format | 1,743 | **9,401 (does not fit)** | 3,145 |

- Median at 5 tools is about 1,320 tokens. The longest train prompt found, sampling 1 row in 40, is 2,231 tokens with 8 candidate tools.
- The full catalogue can't be used, so catalogue mode uses the compact form, as the plan expected.
- The longest gold answer is 75 tokens, so `MAX_NEW_TOKENS = 128` cuts none off.
- Step 0's memory and speed figures came from very short prompts. Real prompts are far longer, so those figures don't carry over. The Step 2 pilot has to re-measure both.

## OOD sets

| Set | Status | Detail |
|---|---|---|
| BFCL en | **Ready** | HF revision `61fc060`, loads in AuditKIT. simple 400, multiple 200, parallel 200, parallel_multiple 200, irrelevance 240 |
| MLCL zh/hi, Igbo | **Not obtainable so far** | The paper's only link is an anonymised review host. Its API returns `410 repository_expired` from Colab. No copy found on HF or GitHub, or on the ACL or Papers with Code pages. One aggregator page (awesomepapers.io) returned 429, so it is unchecked |
| Multilingual-BFCL vi | **Pending** | Still awaiting the authors' approval on 2026-11-02 |

As things stand, the only OOD set in hand is English. MLCL's paper also used `BFCL_v4_multiple`, while the BFCL files here are v3.

## Known limits

- `iot.hue_lighton` and `music.dislikeness` have only 1 and 2 rows in the shared test pool, so per-function results for them mean nothing. Four other functions have 8 or fewer rows.
- `functions.json` is on Drive, not in git. It rebuilds exactly from `data.build_all`.

## Decisions needed at this gate

1. **Judgment filter.** Keep the thresholds above? They cost ar-SA 14% and id-ID nothing. Recommended: keep. Test size is met comfortably, and the bias is reported rather than hidden.
2. **Rare-argument cutoff of 2%.** Gives 211 arguments against the paper's 200. Recommended: keep.
3. **SFT/GRPO split.** SFT takes the first 3,000 sampled rows, so rare functions run out before GRPO starts and GRPO sees only 42–48 functions. Shuffling before the split gives 55 in every language. Recommended: shuffle. This changes the built items, so it needs a rebuild of about 20 seconds.
4. **Multilingual OOD.** Plan decision 8 rules out machine translation, and rebuilding MLCL needs approval. Recommended: add a held-out-language test from MASSIVE's 42 unused locales, such as sw-KE, am-ET, te-IN and km-KH. Those rows are human-collected and parallel to the test ids. This tests unseen languages with the same schemas, so it does not replace new-schema OOD. Separately, the user could email the MLCL authors for the data.
