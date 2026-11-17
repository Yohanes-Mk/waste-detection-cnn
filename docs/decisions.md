# Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-11-16 | New standalone public-pattern repo `waste-detection-cnn`, MIT license, instead of folding into `deep-learning-lab` | Matches the student's own convention of giving showcase-worthy work its own named repo (cf. `Sign-Speech-...`, `Realtime-surveillance-system`, `ResearchMate`); this is a distinct 10-point individual assignment, unrelated to the group Project already tracked in `deep-learning-lab`. |
| 2026-11-16 | Keep tooling lightweight (`requirements.txt` + a few scripts + a notebook) rather than mirroring the heavier reference repo (`joolaoye/csci441-project`: uv/pyproject.toml/YAML configs/pytest) | That reference repo's scope fits a bigger, longer-running research project; this is a 10-point assignment and the PRD itself says folder structure is "proposed organization, not a requirement to generate empty boilerplate." |
| 2026-11-16 | Build E0 (baseline) first; treat E1 (augmentation) as a stretch goal added only if time remains | PRD §12 explicitly flags E1 as "recommended scientific enhancement, not professor mandate" when the deadline is tight; exact due date is still unconfirmed (see open item below). |
| 2026-11-16 | AI-authorship/disclosure split deferred by the student ("do everything I ask for, I'll worry about that when the time comes") | Noted here rather than resolved — flagging because the syllabus requires submitted work to reflect the student's own understanding/voice with disclosed AI use (`docs/assignment.md`). This will need an actual answer before final submission, not just before this repo exists. |
| 2026-11-16 | Due date not yet confirmed in D2L; proceeding with Phase 0 research regardless | Student said to start now rather than wait. Nothing time-sensitive is blocked by this — Phase 0 (literature review, protocol) doesn't depend on the deadline. Needed before any submission-timeline planning (e.g. deciding E1 go/no-go). |
| 2026-11-16 | Proposed refinement to RQ1 (narrow the augmentation claim to small/rare-class performance) — see `docs/literature_notes.md` | Generic "does augmentation help" is a near-tutorial-level ablation (TACO is a one-click Ultralytics dataset entry); narrowing it to TACO's documented small-object/class-imbalance weakness makes it specific without requiring a new experiment. **Proposed, not yet approved — Gate 0.** |

## Open items needing the student's decision (not facts an agent can look up)
- Confirm the Research Assignment's actual due date in D2L.
- Approve, reject, or counter-propose the refined RQ1 in `docs/literature_notes.md` (Gate 0).
- Decide, before final submission, how AI involvement will be disclosed in the report and how much of the submitted narrative/code will be rewritten in the student's own words (syllabus requirement, not optional).
- Repo visibility: currently **private** (recommended default while the assignment is open/ungraded); flip to public afterward if still wanted for portfolio use.
