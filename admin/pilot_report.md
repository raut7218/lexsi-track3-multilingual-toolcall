# Step 2 pilot report

Run 2026-11-02 on the T4, up to commit `870aff0`, with 95 of 95 tests passing there. Source numbers are on Drive in `03_preds/pilot/` and `00_admin/scorer_crosscheck.json`.

These are pilot figures on small slices. No accuracy number here is a result, and none should be quoted.

## Checks

| Check | Result |
|---|---|
| Tests on the T4 | 95 of 95 |
| Scorer against BFCL's `ast_checker` (gorilla `6ea5797`, data `61fc060`), real replies | **800 of 800 agree**, after fixes |
| Same, deliberately broken gold answers | Agrees apart from 3 items, explained below |
| Prompt identity, SFT | 40 of 40 rows equal eval's prompt and the target |
| Prompt identity, GRPO | 32 of 32 real inputs equal eval's tokens, with a single BOS |
| Eval reproduces in a fresh process | 60 of 60 replies identical |
| Resume | Ran only the 30 unfinished of 60 |
| Non-finite values | 0 in SFT and in GRPO |
| GRPO continues a saved adapter | Yes |

## Bugs the pilot caught

Each of these would have damaged results without raising an error.

| Bug | Effect if missed | Fix |
|---|---|---|
| Same-named calls paired by sort order | Correct BFCL parallel answers scored wrong | Each gold call takes the first call that fully passes, as BFCL does |
| Ints accepted in a float array | Over-credit: all 3 real disagreements | Array items are checked by exact type |
| Schema `required` ignored | Over-credit when gold marks a required argument optional | The scorer enforces both |
| Nested gold compared literally | A reply copying gold's accepted-value lists passed | No exact-match shortcut for lists and dicts |
| TRL truncates SFT at 1,024 tokens by default | Prompts run to about 2,200 tokens, so training would cut replies off and learn from nothing | `max_length` set to 8,192 |
| AgentTune's tool rollout adds a second BOS | GRPO would train on different input from eval | No tools passed, see below |
| The same rollout generates a follow-up turn after each call | Correct JSON replies fail the format check, and time doubles | No tools passed, see below |

### Change to the GRPO setup

The plan used AgentTune's tool rollout with stub tools. GRPO now passes no tools, and the format text is already in the prompt. AgentTune's trainer then uses TRL's batched single-turn path. It is still GRPO through `create_agentic_trainer`, on exactly eval's prompts. AgentTune's `trajectories.jsonl` is no longer written, so `rewards.make_reward` logs every reply and its reward parts for the spot checks and the audit.

### BFCL gold that contradicts its own schema

The 11 remaining mutant disagreements come from 3 of the 1,000 BFCL call items. They are left as they are, because BFCL's own data is inconsistent:

- `simple_363`: gold names `find_closest`, and the tool is `restaurant_search.find_closest`. A correct reply can't score under our scorer, so this costs at most 1 of 1,240 items.
- `parallel_multiple_12`: gold lists `permeability`, which the function doesn't have.
- `parallel_multiple_26`: gold lists `type`, which the function doesn't have.

## Measured speed

| Stage | Setting | Speed | Peak GPU memory |
|---|---|---|---|
| Eval, 5 tools | Batch 1 | 0.35 items/s | 7.5 GB |
| Eval, 5 tools | **Batch 4, chosen** | **0.82 items/s**, replies identical to batch 1 on 60 of 60 | 7.9 GB |
| Eval, 5 tools | Batch 8 / 16 | 0.87 items/s, 58 and 59 of 60 identical | 9.0 / 11.2 GB |
| Eval, 5 tools | Batch 32 | Out of memory | |
| Eval, compact catalogue | Batch 4 | 0.34 items/s | 8.9 GB |
| Eval, BFCL | Batch 4 | 0.48–1.2 items/s by category and prompt | |
| SFT | **Batch 2, chosen** | **0.36 examples/s**, loss 0.39 to 0.23 over 320 examples | 11.4 GB |
| SFT | Batch 3 / 4 | 0.34 / 0.32 examples/s | 11.7 / 12.1 GB |
| GRPO | 1 prompt, 4 rollouts | **31.8 s per step** | 10.3 GB |

SFT gets no faster with a larger batch, so its time can only be cut by using fewer examples.

System RAM is also a limit. One kernel was killed at the 12 GB cap after holding raw data, tokenized sets and a trainer together. Long runs should each start in a clean kernel.

## GRPO risk

All four rollouts got the same reward in 5 of 8 pilot steps, even though no step had four identical replies. A tie gives zero advantage, so that step teaches nothing. This pilot used an untrained adapter. A model after SFT may be more consistent and tie more often, which would waste more of the step budget. Step 5 has to measure this on the real SFT adapter before committing to a full run.

AgentTune's default learning rate is `1e-6`. No learning rate has been tested yet, and Step 5 needs a short comparison first.

## Time budget

Estimates from the rates above. The BFCL irrelevance category wasn't timed, so it assumes the speed of `simple`.

| Work | As planned | Recommended |
|---|---|---|
| SFT | 30,000 examples, 23 h | **10,000 examples (1,000 per language), 7.7 h** |
| GRPO | 1,000 steps, 8.8 h | **500 steps, 4.4 h**, after a short check |
| Dev selection | Not sized | 3 SFT and 2 GRPO checkpoints at 41 min each, 3.4 h |
| MASSIVE eval, 5 passes | 10 h | 10 h, unchanged |
| Catalogue eval, 5 passes | 24 h | **100 per language, 4.1 h** |
| BFCL, 4 passes | 1.7 h | 1.7 h |
| Held-out languages, 4 passes | Not in the plan | 2.9 h |
| English policy subset, 3 passes | Not sized | 100 per language, 1 h |
| **Total** | **About 75 h** | **About 35 h** |

At 1,000 per language, SFT keeps all 55 functions in every language, with 15.5% no-call and 25.1% English policy. At 500 per language, one language drops to 54 functions.

## Decisions needed at this gate

1. **SFT size:** 10,000 examples, about 7.7 hours. Recommended. The full 30,000 would take 23 hours for an unknown gain.
2. **GRPO steps:** 500, about 4.4 hours, once Step 5 has checked ties and the learning rate on the real SFT adapter. Recommended. If most steps tie, the count should be revisited before spending the time.
3. **Catalogue sample:** 100 items per language, which the plan already names as the first cut. Recommended.
4. **Compute budget:** about 35 hours of T4 time from Nov 3 to Nov 12. The SFT and GRPO runs are each over 2 hours. The user's remaining Colab compute isn't visible from here.

## HF Jobs `t4-small` pilot

Job `6ae877ae0e0338d1bb7e00cf`, run 2026-11-02 at commit `39631e9`, cost $0.09 for 14 minutes. Source numbers are in private dataset `GauravRaut/lexsi-track3-workspace`, under `03_preds/hf_pilot/`. The same limits apply: no accuracy number here is a result.

| Check | Colab T4 | HF `t4-small` |
|---|---|---|
| Tests | 95 of 95 | 95 of 95 |
| Eval replies, batch 4, 60 items | Baseline | **60 of 60 identical** |
| Eval speed | 0.82 items/s | 1.02 items/s |
| SFT, 8 steps at batch 2 | 0.36 examples/s | 0.47 examples/s |
| GRPO, 4 steps | 31.8 s per step | 15.9 s per step |
| Peak GPU memory, eval / SFT / GRPO | 7.9 / 11.4 / 10.3 GB | 7.9 / 11.4 / 10.0 GB |
| Peak system RAM | Killed once at 12 GB | 13.0 GB of 16 GB |

GRPO's speed comes from 4 steps, against 8 on Colab, on a different adapter and item slice, so its budget is given at both speeds. Python was 3.13.18 here and 3.13.16 on Colab. The 60 replies are the base model only.

| Work | Hours at HF speed |
|---|---|
| SFT, 10,000 examples | 5.9 |
| GRPO, 50-step test and 500 steps | 2.4 to 4.9 |
| Dev selection, 5 checkpoints | 2.7 |
| Final evals, scaled by the eval speedup | 15.8 |
| **Total** | **27 to 29 hours, $11 to $12** |

