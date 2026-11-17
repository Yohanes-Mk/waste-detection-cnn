# Waste Detection CNN

Research assignment for **CSCI 441/541 — Neural Network and Deep Learning** (St. Cloud State University, Fall 2026). Worth 10 points. This is an **individual** assignment, separate from the course's group Project.

**Status: Phase 0 (planning / literature review). No dataset has been downloaded and no model has been trained yet.**

## Assignment

> Research the implementation of convolutional neural networks to do object detection/segmentation/classification: identify a domain, identify a research focus, use or build a dataset, train a CNN on it, and test the trained model on new unseen data with a written report.

Full text: see `docs/assignment.md`. Full due date is **not yet confirmed — check D2L**.

## Project framing

- **Domain:** outdoor litter/waste detection (motivated by, but not implementing, an autonomous cleanup robot — robotics integration is explicitly out of scope for this assignment).
- **Research focus:** object detection (bounding boxes), not classification or segmentation. Rationale in `PRD.md` §11 / final report once written.
- **Candidate dataset:** [TACO](http://tacodataset.org/) (Trash Annotations in Context) — not yet downloaded; see `data/README.md`.
- **Candidate model:** a lightweight pretrained CNN detector (e.g. YOLO11n) fine-tuned on TACO — not yet selected/implemented; see `docs/decisions.md`.

See `PRD.md` for the full planning document and `docs/protocol_phase0.md` for the current one-page experimental protocol proposal, which is pending student approval (Gate 0) before any dataset download or training begins.

## Repo layout

```
README.md               this file
PRD.md                  full planning document
docs/
  assignment.md         verbatim copy of the professor's assignment
  literature_notes.md   Phase 0 literature scan + RQ assessment
  decisions.md          decision log (what was decided, when, why)
  agent_progress.md     phase-by-phase status tracker
  protocol_phase0.md    one-page experimental protocol proposal (pending approval)
data/
  README.md             dataset provenance/license notes (filled in once a dataset is approved)
```
`configs/`, `src/`, `notebooks/`, `outputs/`, `report/` will be added once Phase 1+ begins — intentionally not scaffolded empty.

## AI use disclosure

This repository was developed with assistance from an AI coding agent (Claude) for planning, scaffolding, literature research, and code drafting. Per the course's AI use policy, the student author reviews, understands, and takes ownership of all submitted reasoning and code; AI involvement is disclosed in the final report per instructor policy.
