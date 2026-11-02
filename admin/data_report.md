# Step 1 data report

Built 2026-11-02 on the T4 from commit `bc9d9e2`, with 77 of 77 tests passing there. Source numbers are on Drive in `01_data/build_counts.json` and `01_data/report_numbers.json`. Items are in `01_data/items/` (46 files), and `functions.json` has sha256 `12cf91a9…0375f73`.

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
| Function coverage | Test 55, dev 54, SFT 55, GRPO 55 in every language |

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

| Locale | Aligned with en-US | English-value policy | No-call |
|---|---|---|---|
| en-US | 4,000 | 999 | 614 |
| fr-FR | 3,942 | 1,002 | 603 |
| es-ES | 3,918 | 976 | 601 |
| zh-CN | 3,993 | 994 | 586 |
| ja-JP | 3,978 | 980 | 597 |
| ar-SA | 3,983 | 984 | 608 |
| hi-IN | 3,985 | 994 | 601 |
| id-ID | 3,969 | 960 | 610 |
| vi-VN | 3,979 | 984 | 601 |
| tr-TR | 3,964 | 978 | 597 |

## Context fit, in real Tiny Aya tokens

The model's limit is 8,192 positions. The chat template adds 366 tokens by itself. Each cell is the worst case over 10 languages, using every test item for 5 tools and the 20 longest queries per language for 55 tools.

| Prompt | 5 tools | 55 full JSON | 55 compact |
|---|---|---|---|
| BFCL zero-shot | 1,751 | **9,409 (does not fit)** | 3,153 |
| Custom one-shot | 1,755 | **9,413 (does not fit)** | 3,157 |
| Training format | 1,743 | **9,401 (does not fit)** | 3,145 |

- Median at 5 tools is about 1,320 tokens. The longest train prompt found, sampling 1 row in 40, is 2,230 tokens.
- The full catalogue can't be used, so catalogue mode uses the compact form, as the plan expected.
- The longest gold answer is 75 tokens, so `MAX_NEW_TOKENS = 128` cuts none off.
- Step 0's memory and speed figures came from very short prompts. Real prompts are far longer, so those figures don't carry over. The Step 2 pilot has to re-measure both.

## OOD sets

| Set | Status | Detail |
|---|---|---|
| BFCL en | **Ready** | HF revision `61fc060`, loads in AuditKIT. simple 400, multiple 200, parallel 200, parallel_multiple 200, irrelevance 240 |
| MLCL zh/hi, Igbo | **Not obtainable so far** | The paper's only link is an anonymised review host. Its API returns `410 repository_expired` from Colab. No copy found on HF or GitHub, or on the ACL or Papers with Code pages. One aggregator page (awesomepapers.io) returned 429, so it is unchecked |
| Multilingual-BFCL vi | **Pending** | Still awaiting the authors' approval on 2026-11-02 |
| MASSIVE held-out languages | **Ready** | Swahili, Amharic, Telugu, Khmer. See below |

The only new-schema OOD set in hand is English. MLCL's paper also used `BFCL_v4_multiple`, while the BFCL files here are v3.

## Held-out languages

These use the 10-language test ids, tool lists and no-call assignment, and the same filters. A row that fails a filter in a held-out language is left out, so those sets are smaller, and comparisons with the 10 languages should use the shared ids.

| Locale | Items of 588 | No-call | Functions | Judgment removed | Worst prompt tokens |
|---|---|---|---|---|---|
| sw-KE | 570 | 86 | 55 | 257 | 1,753 |
| am-ET | 582 | 86 | 55 | 283 | 1,771 |
| te-IN | 581 | 87 | 55 | 255 | 1,761 |
| km-KH | 429 | 68 | 55 | 4,263 | 1,708 |

- Ids, order, tools and no-call flags match the main test set, and 100% of gold values appear in the query.
- km-KH loses 26% of its rows to the judgment filter, mostly on grammar (4,282 rows fail it), so it has 27% fewer items than the full set.
- All four languages are on Tiny Aya's own language list. They are new to our SFT and GRPO, not to the base model. The right claim is "languages outside our training", not "languages the model has never seen".
- They reuse the same 55 schemas, so they say nothing about new tools. BFCL en covers that, in English only.

## Known limits

- `iot.hue_lighton` and `music.dislikeness` have only 1 and 2 rows in the shared test pool, so per-function results for them mean nothing. Four other functions have 8 or fewer rows.
- `functions.json` is on Drive, not in git. It rebuilds exactly from `data.build_all`.

## Decisions made at this gate

The user accepted all four recommendations on 2026-11-02.

1. Judgment filter kept as above. Its uneven effect is reported.
2. Rare-argument cutoff kept at 2%.
3. Train rows are shuffled before the SFT/GRPO split, which raised GRPO from 42–48 functions to 55.
4. MASSIVE held-out languages added as the multilingual OOD test. MLCL is not rebuilt. Contacting its authors is left with the user.
