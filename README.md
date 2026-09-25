# Amazon ML Challenge 2026 — Business Entity Resolution: Strategic Analysis

**Problem:** Business Entity Resolution across 3 noisy sources (US, India, +France in test only)
**Metric:** F_0.5 (precision-weighted), macro-averaged per Source-1 entity, singletons included
**Constraint:** Final model ≤8B parameters, MIT/Apache 2.0 license, no external data lookups
**Window:** Sept 25, 12:00 AM IST → Sept 27, 11:59 PM IST (72 hours)

---

## 1. What This Problem Actually Is

Strip away the ML framing for a second: this is a **classic, well-studied Entity Resolution (ER) problem** — decades of database/IR research exist on exactly this shape of task (matching noisy business/product/person records across sources with no shared key). This is good news. You are not inventing an approach; you're **selecting and tuning from a known, mature pipeline**, which is far more tractable in 72 hours than a novel research problem.

The academic field splits ER into two stages, and your problem statement's own file structure (`candidate_pairs.tsv` then `matching_results.tsv`) mirrors this exactly:

1. **Blocking / Candidate Generation** — cheaply narrow billions of possible pairs (every Source-1 record × every Source-2/3 record) down to a small, high-recall candidate set
2. **Matching / Scoring** — a more expensive, precise model that decides, for each candidate pair, match or no-match

This two-stage shape is not incidental — it's *required by your output format*. You cannot skip blocking and brute-force compare every pair; with even modest source sizes, full cross-product comparison is computationally infeasible in the time you have, and the problem statement explicitly asks you to submit your blocking output separately for evaluation.

### Why this reframes your AWS/compute plan
Everything I told you earlier about GPU strategy assumed a possible VLM/image task. **This problem has zero images.** It's pure text (business names, addresses) plus categorical (country). This means:
- Your compute needs are dramatically lower than a VLM task
- Classical ML + string similarity + lightweight embeddings will very likely outperform or match a heavy LLM approach — and the research bears this out (see Section 4)
- Your $200 AWS credit is now *generous* for this problem, not a tight constraint — you could likely run this entire pipeline for well under $30–40 if you're disciplined
- CPU instances (`ml.m5.xlarge`, `ml.c5.xlarge`) may be entirely sufficient for blocking + classical matching; GPU only matters if you fine-tune a transformer/small LLM for the matching stage

---

## 2. Deconstructing the Problem Statement — Non-Obvious Details That Will Decide Your Score

These are easy to skim past but each one can make or break your F_0.5:

### 2.1 The metric rewards precision AND correctly saying "no match"
F_0.5 weights precision 2× over recall, and it's computed **per Source-1 entity, then macro-averaged**, including singletons (entities with zero true matches). This has a subtle but major implication:

> **A model that's too aggressive at matching will get destroyed by singleton false positives.** A Source-1 entity with no true match scores 1.0 if you predict empty, 0.0 if you predict *anything*, even one wrong low-confidence guess.

This means your matching model needs a **confident abstention mechanism** — a threshold or a "no-match" class — not just "top-1 nearest candidate." If you don't know how many singletons exist in the test set, treat this as a first-priority EDA question (Section 3).

### 2.2 France is a zero-shot generalization test
Training data only has US and India; test set adds France. The problem statement explicitly warns: **do not hard-code or filter by country**. This tells you the organizers are specifically testing whether your pipeline generalizes to an unseen address/name format, not whether you memorized US/India patterns. Practical implications:
- Any feature engineering that's US/India-specific (e.g., hardcoded regex for PIN codes, zip codes, specific abbreviation dictionaries) needs a **generic fallback path** for France
- String-similarity and embedding-based features generalize better across country/language than rule-based ones — this is a point in favor of a hybrid approach over a purely rule-based one
- Test your validation split's behavior on India specifically as a *proxy* for the France-generalization risk, since India is likely the most "different" of your two training countries from a Western-name-format baseline

### 2.3 Many-to-many, not one-to-one
"A Source 1 entity may match zero, one, or many records from Source 2 and Source 3." This isn't simple deduplication (1:1 pairs) — a single business can have multiple listings in Source 2/3 (e.g., different branches, or the same business duplicated with typos). Your matching stage needs to output a **set**, not a single best match — meaning you likely want a **threshold-based binary classifier per candidate pair** (does this pair match: yes/no) rather than a "pick the single best candidate" ranking approach.

### 2.4 candidate_pairs.tsv is scored for recall ceiling, not the leaderboard — but it still matters
It's not directly scored, but the problem statement is explicit: "a matched ID that never appeared as a candidate signals a pipeline bug." More importantly, **your blocking recall is a hard ceiling on your final F_0.5** — if the true match isn't in your candidates, no matching model can recover it. This is worth genuine engineering time, not an afterthought.

### 2.5 The 8B-parameter, MIT/Apache constraint is a strong signal
This explicitly permits (and somewhat nudges you toward) using a small open-weight LLM/language model as your matcher — but caps it, ruling out anything like a 70B model or closed API (GPT-4, Claude, etc. — also independently disallowed by the "no external lookups/services" rule, which likely extends to API-based LLMs given the fair-play framing). Section 4 covers what actually works at this scale.

---

## 3. First Moves — EDA Checklist (Do This First, Hours 0–4)

Before anyone writes a model, someone should answer these — they determine every downstream decision:

| Question | Why it matters |
|---|---|
| What fraction of Source-1 entities are singletons (no match) in training ground truth? | Sets your baseline for "predict nothing" and tells you how aggressively to threshold |
| What's the match multiplicity distribution? (how many entities match 1, 2, 3+ candidates) | Confirms many-to-many is common enough to design around, or rare edge case |
| How much missing data per field, per country? (esp. address components) | India/US likely differ a lot here — France will differ more |
| What's the country distribution in train vs. what you can infer about test? | Can't see France in train, but understanding the US/India split tells you how much of your tuning is "safe" vs. how much is guesswork for France |
| Are entity_id prefixes reliable (S1-/S2-/S3-) or is there any noise there? | Cheap sanity check, rules out a dumb bug early |
| What's the naive brute-force pair count (\|S1\| × (\|S2\|+\|S3\|))? | Tells you whether GPU-scale embedding search is even necessary, or if CPU-based blocking suffices |
| Spot-check 20–30 known true matches manually | Nothing beats eyeballing real examples — you'll immediately see the *kinds* of noise (transliteration? abbreviation? word order?) that dominate, which should drive your feature choices over generic string metrics |

---

## 4. Research-Backed Approach Options

I looked at both classical IR/database literature (this problem has ~20 years of prior work) and recent (2024–2026) LLM-based entity matching research, since your constraints (8B param cap, no external APIs) sit right at the boundary between "classical" and "LLM-assisted" approaches.

### 4.1 What the research actually shows (this matters for your architecture decision)

A few consistent findings across multiple papers I reviewed:
- **TF-IDF-based blocking remains a strong, cheap baseline.** A 2023 VLDB paper ("Sparkly") found a simple, well-tuned TF-IDF blocker to be "surprisingly strong" against much fancier learned blockers — don't dismiss this as too basic for your blocking stage.
- **Sentence embeddings (Sentence-BERT / MiniLM-style models) are the standard "advanced blocking" upgrade** — encode each record as one string, embed it, do approximate nearest-neighbor search (FAISS) to find candidates. This is fast, cheap, and CPU-feasible at moderate scale.
- **Fine-tuned small/discriminative models (not necessarily huge LLMs) are competitive with or beat much larger LLMs on entity matching.** One EDBT 2025 paper found fine-tuned small models "on par with Mixtral-8x7B" despite being orders of magnitude smaller — and cheaper/faster at inference by 2–4 orders of magnitude. This is directly relevant: your 8B cap is not a real handicap; the literature suggests you don't need anywhere near 8B parameters to do well.
- **A classic, well-cited approach (Ditto, and similar) treats matching as sentence-pair classification**: concatenate each record's fields into one string per side, feed the pair into a transformer (like a fine-tuned BERT/RoBERTa), classify match/no-match. This is simpler to implement well in 72 hours than a decoder-LLM prompting approach and has a long track record.

### 4.2 Three realistic architecture options for your team

**Option A — Classical/Lightweight (lowest risk, fastest to a working baseline)**
- Blocking: TF-IDF on normalized business_name (+ address tokens), or a rule-based key (e.g., first N characters of normalized name + country) combined with TF-IDF for recall
- Features: Jaccard, Levenshtein/edit distance, TF-IDF cosine similarity, token overlap, address component matches
- Matcher: Gradient-boosted tree (XGBoost/LightGBM) on the engineered similarity features, binary classify match/no-match per candidate pair, threshold tuned against your validation F_0.5
- Compute: CPU-only realistically sufficient. Trains in minutes.
- Risk: may plateau below a fine-tuned transformer's ceiling on genuinely hard fuzzy cases (heavy transliteration, word-order transpositions)

**Option B — Embedding + Transformer Matcher (balanced, and where I'd point a team with your skill mix)**
- Blocking: Sentence embeddings (e.g., a small multilingual sentence-transformer, since you need to generalize to France) + FAISS approximate nearest-neighbor for candidate generation, possibly unioned with a TF-IDF/rule-based blocker for extra recall safety
- Matcher: Fine-tune a small pretrained encoder (something in the 100M–1B range — e.g., a multilingual MiniLM/DeBERTa-style model, well under your 8B cap) as a sentence-pair classifier on concatenated record fields
- Also feed in the classical similarity features (Option A's features) as auxiliary inputs to a small classifier head, or ensemble with an XGBoost model on those features — ensembling a classical model with a learned embedding model is a well-documented way to catch different error types
- Compute: light GPU helpful for fine-tuning (a few hours on `ml.g4dn.xlarge`), but not heavy — this is a small model, not a 7-8B one
- Risk: more moving parts, more that can break under time pressure — needs careful validation-set discipline

**Option C — Small Open-Weight LLM as Matcher (uses your full 8B allowance, higher risk/reward)**
- Blocking: same as Option B (embeddings + ANN)
- Matcher: fine-tune (via LoRA/QLoRA — parameter-efficient, not full fine-tune) a small open model like **Qwen3-8B, Llama-3.1-8B-Instruct, or DeepSeek-R1-Distill-Qwen-7B** (all confirmed open-weight, several MIT/Apache-licensed) to output match/no-match given a formatted pair of records as a prompt
- The research is genuinely mixed here: fine-tuned LLMs *can* edge out smaller discriminative models on some benchmarks, but the gap is often small, and inference cost/time is much higher per pair — a real concern if your candidate set is large and you have a 3-submission-per-day cap to work within
- Compute: needs GPU fine-tuning (QLoRA on `ml.g5.xlarge`/`ml.g4dn.2xlarge`), meaningfully more setup and iteration time than Options A/B
- Risk: highest complexity-to-benefit ratio for a 72-hour window with a team that (per your earlier answers) has DL/VLM strength but this is a different skill (LLM fine-tuning for classification, not vision) — worth doing only if Option B is done early and you have time/compute to spare

### 4.3 My actual recommendation
Given your team profile (strong across DL, classical ML, and engineering) and the 72-hour/precision-sensitive nature of F_0.5:

**Build Option A first — treat it as your submission-ready floor, not throwaway code.** It's fast, avoids compute risk entirely, and gets you on the leaderboard within the first several hours. Then push toward **Option B** as your main effort, since your team's skill mix supports it and the literature backs it as a strong middle ground. Only reach for **Option C** if B is solid, submitted, and validated well before the deadline, and you have people/compute free to experiment — treat it as a stretch goal, not the plan.

This mirrors exactly what I flagged in your AWS handbook: never be leaderboard-empty, get a working (if simple) submission early, then improve.

---

## 5. Blocking Strategy — Deserves Real Attention

Since blocking recall caps your entire pipeline's ceiling, treat this seriously:

1. **Normalize aggressively before anything else**: lowercase, strip punctuation, expand known abbreviations both directions (Corp ↔ Corporation, Pvt ↔ Private, Ltd ↔ Limited, Rd ↔ Road, St ↔ Street, & ↔ and), strip legal suffixes into a separate flag/feature rather than deleting information
2. **Use multiple blocking keys and union them** — don't rely on one blocking signal. Reasonable candidates:
   - TF-IDF cosine similarity on normalized business_name (top-K per Source-1 entity)
   - TF-IDF/token overlap on business_address
   - Sentence embedding cosine similarity (catches semantic-level similarity TF-IDF misses, e.g., transliteration variants)
   - Country as a hard or soft filter (soft, since France in test has no training examples to calibrate against — don't over-trust country matching alone)
3. **Measure blocking recall on your own validation split before building the matcher** — if you can't recover, say, 95%+ of true matches in your candidate set, fix blocking before investing in a fancier matcher; no matcher can find a pair that was never generated as a candidate
4. **Watch your candidate set size** — too generous a top-K blows up matcher inference time (and possibly cost if you're paying for GPU inference); too tight and you cap recall. Tune top-K against your validation recall/size tradeoff.

---

## 6. Realistic Risk List

| Risk | Why it matters here specifically | Mitigation |
|---|---|---|
| Over-matching hurts you more than under-matching | F_0.5's precision weighting + singleton scoring | Tune your match threshold conservatively; validate against F_0.5 directly, not just F1 or accuracy |
| France generalization | Zero training examples for this country | Avoid country-specific hardcoding; validate held-out India performance as a generalization proxy |
| Blocking recall ceiling | Two-stage pipeline structure | Measure blocking recall explicitly before matcher work begins |
| Wasting compute on Option C prematurely | 8B LLM fine-tuning is the most time/compute-expensive path | Sequence A → B → (C if time allows), per Section 4.3 |
| Submission cap (5/day) | Can't spam-tune against the public leaderboard | Rely on your own validation F_0.5 as primary signal; treat leaderboard submissions as periodic checkpoints, not a tuning loop |
| Methodology document quality | Explicitly reviewed for top teams, no page limit mentioned as a downside | Don't treat it as an afterthought — document your blocking strategy and architecture decisions as you go, not at hour 71 |
| Reproducibility of final package | Explicitly audited ("code/business_entity_resolution/", README, requirements.txt) | Keep environment/dependencies pinned from day 1; don't let this become a last-minute scramble |

---

## 7. Immediate Next Steps for Your Team

1. Run the EDA checklist (Section 3) — assign this to 1–2 people right away, results should be shared with the whole team within the first few hours
2. Build Option A end-to-end (blocking → matching → valid submission format) as fast as possible — validate output against `utils/validate_submission.py` immediately, don't wait
3. Get one submission onto the leaderboard early, even if mediocre
4. Move to Option B once A is submitted and validated
5. Revisit AWS compute needs now that this is confirmed to be a text/tabular problem — CPU instances are very likely sufficient for A and much of B; reserve GPU budget for B's transformer fine-tuning and any Option C exploration
