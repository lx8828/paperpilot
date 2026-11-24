# Oracle-KG upper-bound case study (2026-10-19)

> Scope: Q4 engineering evidence.  This is **not** a real KG implementation.
> It tests whether the existing semantic retriever produces relation-level
> hallucinations on which an ideal, complete knowledge graph could act.

## Setup

- Corpus: 50 papers from `assets/papers/`.
- Chunks: 1,021 current chunks with cached `bge-m3` vectors.
- Retrieval: cosine similarity, top-12 chunks across the 50 papers.
- Answerer: the production L3 two-step answer path (`facts -> answer`).
- Model: `deepseek-chat`.
- The existing `validator.gate` was disabled to observe generation-side
  confusion rather than whether the current checker catches it.
- Twenty targeted questions were run in two rounds.  Ten expected a positive
  relation; ten deliberately assigned that relation to a wrong but highly
  similar paper.

Raw runs (local run records; they contain paper excerpts, so per repo policy
they are not distributed — the files sit next to this report under `qa/`):

- `oracle_kg_probe_50_round2.json`
- `oracle_kg_probe_50_round3.json`

## Observed failures

All three observed failures are variants of the same high-risk case:
`SkillMAS` and `Skill-MAS` are different papers with very similar names.

| # | Question | Semantic-RAG output | Correct attribution |
|---|---|---|---|
| 1 | Does `SkillMAS` decouple experience retention from parameter updates? | **Yes**, citing the “third path / decouples experience retention from parametric updates” evidence | This is a core mechanism of **Skill-MAS**, not SkillMAS |
| 2 | Does `SkillMAS` model high-level orchestration as an evolvable Meta-Skill? | **Yes**, and explicitly renamed the target to “Skill-MAS” inside the answer | The evolvable Meta-Skill mechanism belongs to **Skill-MAS** |
| 3 | Is `Skill-MAS` evaluated on ALFWorld, Lifelong Agent Bench OS, and τ-Bench Retail? | **Yes**, then used **SkillMAS** numbers (94.0 / 76.7 / 70.2) | Those benchmarks belong to **SkillMAS**; Skill-MAS uses four different complex benchmarks |

The three errors are not simple omissions.  They attach a real mechanism or
evaluation result to the wrong entity, producing a confident answer that
looks fully grounded because the retrieved evidence itself is real.

## Existing validator result

The production validator was then run on the three saved outputs with the
correct context count (`n_entries=12`):

| Case | `validator.check` | `gate` action |
|---|---|---|
| SkillMAS / decoupling | `ok=true`, no issues | **pass** |
| SkillMAS / Meta-Skill | `ok=true`, no issues | **pass** |
| Skill-MAS / ALFWorld | `ok=true`, no issues | **pass** |

This is the important gap.  The validator successfully checks whether the
answer is supported relative to its cited text, but it does not check whether
the cited text belongs to the paper node named in the question.  Here the
evidence is real, so the answer passes factuality checking while still being
bound to the wrong entity.

Raw validator run (local record, not distributed; under `qa/`):

- `validator_on_kg_failures.json`

## Hard multi-hop failure

A second probe asked for **one paper** that both:

```text
uses an evolvable Meta-Skill
AND
is evaluated on HotpotQA / LoCoMo / GAIA
```

No such paper exists.  The answer incorrectly selected `MASkills`.  The
answer cited only chunks from `2609.02094v1.pdf` (MASkills):

- evidence that MASkills evaluates on HotpotQA, LoCoMo, and GAIA;
- evidence that MASkills edits its skill space with refinement/induction/
  consolidation/pruning operators.

It then inferred that this was an `evolvable Meta-Skill`.  The first half is
true for MASkills, but `Meta-Skill` is the mechanism of the different paper
`Skill-MAS`.  A simple check requiring all citations to come from the chosen
paper therefore **does not catch it**.

The production validator returned:

```text
ok = true
issues = []
gate action = pass
```

An Oracle graph would reject the conjunction because no single paper node
has both:

```text
paper --core_mechanism--> evolvable Meta-Skill
paper --evaluated_on----> HotpotQA/LoCoMo/GAIA
```

This is a stronger example than cross-paper citation confusion: every cited
text is from the selected paper, but the synthesized relation is still false.

## MID findings are not blocked by default

A further hard question about `GenRM` and outcome-based filtering produced a
self-contradictory ending and two unsupported `MID` issues.  The validator
detected them, but the production gate still returned `pass` because only
`HIGH` issues trigger repair/fallback by default.  Therefore, detection must
not be equated with correction.

## Oracle graph used for the upper bound

The manually specified ideal graph would contain at least:

```text
SkillMAS  --core_mechanism--> bounded skill evolution
SkillMAS  --evaluated_on----> ALFWorld
SkillMAS  --evaluated_on----> Lifelong Agent Bench OS Task
SkillMAS  --evaluated_on----> tau-Bench Retail

Skill-MAS --core_mechanism--> evolvable Meta-Skill
Skill-MAS --decouples-------> experience retention and parameter updates
Skill-MAS --evaluated_on----> DeepResearchBench
Skill-MAS --evaluated_on----> HLE-Math
Skill-MAS --evaluated_on----> BrowseComp-Plus
Skill-MAS --evaluated_on----> VitaBench
```

Each node and edge would carry a paper id, chunk id, page, and verbatim
evidence quote.

## Ideal-KG checks

1. Query `SkillMAS --decouples--> experience/parameter updates`

   No such path exists.  The relation exists only from `Skill-MAS`.
   Ideal action: reject as unsupported or correct the entity attribution.

2. Query `SkillMAS --core_mechanism--> evolvable Meta-Skill`

   No such path exists.  The retrieved evidence is grounded but belongs to a
   different paper node.  Ideal action: entity-link error, reject the claim.

3. Query `Skill-MAS --evaluated_on--> ALFWorld`

   No such path exists.  ALFWorld is linked to `SkillMAS`; Skill-MAS has a
   different benchmark set.  Ideal action: detect the cross-paper binding
   error and reject the false premise.

Thus, under the Oracle assumption of complete nodes, correct entity linking,
typed relations, and provenance, all three observed hallucinations can be
identified and blocked.

## Control observations

The other 17 targeted questions were answered correctly, including swaps
among:

- COMAD and MASkills;
- ToolGen and UniToolCall;
- DTDR and ToolDreamer;
- A-MEM and Infini Memory;
- HiMem and Weighted Memory Tree;
- Mirror-Critique and VRR.

This matters for honest reporting: the semantic baseline is not generally
broken.  The result is a targeted case study showing that KG constraints can
address a specific residual failure mode, not evidence that KG universally
improves answer accuracy.

## Allowed report claim

> On 20 targeted relational questions over a 50-paper corpus, ordinary
> semantic RAG produced three confident entity-binding errors.  A complete
> Oracle KG with typed `core_mechanism` and `evaluated_on` relations could
> detect all three by finding a missing path, a conflicting path, or a path
> attached to a different paper node.  This establishes an upper bound for
> relation-level hallucination correction, not the performance of an
> automatically constructed KG.

## Limitations

- The corpus is small and the questions were deliberately targeted at
  near-name entities.
- The graph was manually specified; extraction, entity linking, graph
  completeness, and query generation were not evaluated.
- The existing validator was disabled for the baseline run.
- The result does not establish statistical superiority over RAG or
  full-context answering.
- Missing graph edges must trigger full-text verification; absence of an edge
  cannot be treated as proof that the relation is false.

## Reproduction

```powershell
$env:PYTHONIOENCODING='utf-8'
./.venv/Scripts/python.exe -u qa/oracle_kg_probe_50.py
```

The current script contains the third-round case list.  Earlier raw runs are
preserved in the JSON files listed above.
