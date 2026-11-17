# Phase 0 literature scan

Scope: find enough credible sources to judge whether RQ1 (as drafted in the PRD) is measurable, non-trivial, and adequately distinct from a standard tutorial exercise, per the PRD's "mandatory early research challenge." This is a quick scan (web search + abstracts), not an exhaustive related-work survey — a fuller version belongs in the eventual report.

## Sources reviewed

1. **Proença & Simões (2020), "TACO: Trash Annotations in Context for Litter Detection,"** arXiv:2003.06975. https://arxiv.org/abs/2003.06975
   The dataset's own paper: 1,500 images, 4,784 annotations, hierarchical taxonomy, crowdsourced from diverse real-world outdoor scenes (beaches, streets, parks). Reports instance segmentation with Mask R-CNN as the original baseline. Confirms the dataset's known weaknesses: small size, long-tailed/imbalanced categories, many single-instance classes.

2. **Hernandez (2022), "Searching for Uncollected Litter with Computer Vision,"** arXiv:2211.14743. https://arxiv.org/pdf/2211.14743v1
   Directly frames litter detection as the perception front-end for a search/cleanup task without claiming a deployed robot — closely matches this project's "motivating application, not a promised deliverable" framing.

3. **"Real time intelligent garbage monitoring and efficient collection using YOLOv8 and YOLOv5..."** (2025), *Scientific Reports*. https://www.nature.com/articles/s41598-025-99885-x
   Applies YOLOv5/YOLOv8 to real-time litter/garbage detection for collection systems, including resource-constrained hardware (Raspberry Pi) — supports that a lightweight YOLO-family detector is a reasonable, literature-supported choice for this problem, not an oversized tool for the job.

4. **"Robust and Label-Efficient Deep Waste Detection"** (2025), arXiv:2508.18799. https://arxiv.org/pdf/2508.18799
   Recent (2025) work specifically on *robustness* and label-efficiency in waste detection — relevant to the augmentation/generalization half of RQ1 and H1; indicates this is still an open, active research angle rather than a solved problem.

5. **Object detection and classification for TACO litter dataset using Faster R-CNN** (CalState MS project). https://scholarworks.calstate.edu/concern/projects/w6634d332
   A worked example of TACO + a two-stage detector (Faster R-CNN) rather than YOLO — useful as the "on-paper alternative model" comparison required later in PRD §6, and as evidence that TACO + a single generic detector, run once, is a well-trodden student-project pattern.

## Is RQ1 measurable?

Yes. mAP@0.5 / mAP@0.5:0.95, precision/recall at a fixed IoU threshold, and per-class AP are all standard, literature-reported metrics for exactly this dataset and task (sources 1, 3, 5 all report some subset of these). Nothing about RQ1 is unmeasurable with this assignment's time and compute budget.

## Is RQ1 adequately distinct from a standard tutorial?

**Partially — the generic framing is not distinct enough; a narrower framing is.** Two findings push this conclusion:

- Ultralytics has an official hosted dataset entry for TACO on its own platform (`platform.ultralytics.com/.../taco-trash-annotations-in-context`), i.e. "fine-tune YOLO on TACO" is literally a one-click tutorial path, not a novel combination.
- "Does data augmentation improve generalization" is one of the most generic ablations in all of deep learning — source 1–5 collectively treat it as background knowledge, not a contribution. A 10-point assignment doesn't need novelty, but the PRD's own §2 explicitly asks the agent to flag exactly this kind of genericness before locking it in.

What *is* still genuinely informative, supported by the dataset's own documented weaknesses (source 1) and the open robustness angle (source 4), is narrowing the augmentation question to a specific, checkable sub-claim instead of a blanket "does augmentation help."

## Proposed refinement (not a silent replacement — pending your approval)

**Keep RQ1's core shape (baseline vs. augmented fine-tuning of a lightweight detector), but narrow the "does it help" claim to where TACO's own documented weakness lives:**

> **RQ1 (refined):** For a lightweight CNN-based detector fine-tuned on TACO, does training-time augmentation improve detection performance disproportionately on small-footprint/rare litter categories (TACO's documented weak spot) compared to common, large-footprint categories — rather than improving all categories uniformly?

This keeps E0/E1 exactly as planned (no extra experiment needed), just changes *how the same two runs get sliced and discussed* — break mAP out by object-size bucket and by rare- vs. common-class, instead of only reporting one pooled number. Cheap to add, meaningfully more specific, and directly tied to a documented dataset property instead of a generic claim.

### Alternative #2 (more scope change, offered per PRD requirement — not recommended as the default)

> **Alt-RQ:** How does collapsing TACO's ~60 fine-grained categories into a compact taxonomy (e.g. one unified `litter` class vs. 3–5 material-based groups) affect detection accuracy and training stability for a lightweight CNN detector, given TACO's severe class imbalance?

This replaces the augmentation comparison with a class-taxonomy comparison. It's a legitimate, literature-supported angle (source 1 flags class imbalance directly) but is a bigger pivot — it changes what E0 vs. E1 means and would need the class-taxonomy decision (PRD §4) resolved first. Not recommending this as the default; flagging it only because the PRD requires offering it.

## Recommendation

Go with the **refined RQ1** above. It requires no new experiment, no scope expansion, and no change to the chosen dataset/model — just a more specific evaluation lens on the same two planned runs. This is proposed, not yet approved; see `docs/decisions.md` and `docs/protocol_phase0.md`.
