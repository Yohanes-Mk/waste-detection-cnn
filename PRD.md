# Product & Research Requirements Document (PRD)
## Vision-Based Waste Detection for Autonomous Robotic Systems

**Version:** 0.1 — research planning baseline
**Course:** CSCI 441/541, Neural Network and Deep Learning
**Assignment:** Research Assignment (10 points)
**Status:** Planning; no experiments or results have been completed
**Due date:** **TBD — verify in D2L**
**Owner:** Student researcher
**Coding collaborator:** VS Code AI agent, subject to academic integrity requirements

> **Scope note:** This project is **waste/litter object detection**, NOT industrial defect detection. A mobile robot is the motivating application, not a promised deliverable. The course assignment is about CNN implementation, training, and inference on unseen images.

## 1. Source of truth and academic constraints

### Required by the uploaded professor's assignment
- [ ] Identify a research domain.
- [ ] Identify and justify **one** research focus: object detection, segmentation, or classification.
- [ ] Construct a dataset or use an existing dataset (e.g., GitHub/Kaggle/repository).
- [ ] Research and implement a technique to **train a convolutional neural network** on that dataset.
- [ ] Test the **trained model** using **new, unseen data** and report model output.
- [ ] Submit a report stating domain; explaining selected focus; describing dataset count, size, dimensions, etc.; including CNN training code (recommended) and optionally run screenshots; discussing results on unseen data.
- [ ] Verify deadline in D2L. The assignment is worth 10 points.

### Related course syllabus constraints
- The syllabus allows AI tools for limited academic support (e.g., idea generation, outlines, early revisions). **Submitted work must reflect the student's own understanding and reasoning**, and AI usage must be disclosed. Confirm with instructor whether AI-agent-generated code is permitted for this assignment before relying on it in a graded submission.
- Do not fabricate model runs, metrics, screenshots, annotations, results, citations, or experimental observations.

### Recommended additions (NOT explicitly required by the professor)
- Separate train/validation/test data, track random seeds and dependencies, compare a baseline against one controlled intervention, and evaluate precision/recall/mAP.
- Preserve reproducibility, failure-case analysis, and a clear README.
- Optional local inference demo if time remains; robot integration is **out of scope**.

## 2. Problem, hypothesis, and research questions

### Problem
Waste on sidewalks, parks, and other outdoor environments varies in size, shape, visual appearance, lighting, and background clutter. An autonomous cleanup robot first needs to identify and localize candidate litter objects before it can navigate toward or pick them up. This project studies the visual detection component only.

### Primary research question (RQ1)
**How accurately can a lightweight CNN-based object detector localize common litter objects in real-world scenes, and does training-time augmentation improve generalization to unseen images?**

### Initial hypothesis (H1)
**An augmented training configuration will improve held-out detection robustness relative to an otherwise identical non-augmented fine-tuning baseline, particularly for small objects and variable lighting.** This is a hypothesis, not an asserted result.

### Secondary questions (optional; time permitting)
- RQ2: Which litter categories and object sizes have the highest false-negative rates?
- RQ3: What inference speed and model size trade-offs arise when using a compact detector?

### Mandatory early research challenge for the agent
Before finalizing RQ1, find 2–4 related research papers or credible technical studies. Determine whether RQ1 is measurable and adequately distinct from a standard tutorial. Suggest up to two **better-scoped** alternative RQs if supported by evidence. Explain trade-offs. **Do not silently replace RQ1 or expand scope; request student approval.**

## 3. Project boundaries

**IN:** litter image dataset, annotation validation, class mapping, preprocessing, CNN model selection, fine-tuning, controlled comparison, evaluation on unseen images, annotated inference examples, reproducible report.

**OUT:** robotic arms, SLAM, route planning, physical waste pickup, custom hardware, live camera deployment, full multi-model benchmark, claims of safety-critical reliability. These may be future work.

### Research quality bar
This is an experiment, not a demo. No conclusion may be made without traceable training and evaluation outputs. Use one well-controlled experiment rather than several superficial ones.

## 4. Dataset plan and provenance

### Candidate: TACO (Trash Annotations in Context)
- Official source: https://github.com/pedropro/TACO
- Project website: http://tacodataset.org/
- Alternate mirrored listing: https://huggingface.co/datasets/pictograph/taco
- Approximate reference scale for one mirror: **1,500 images, 4,784 annotations, 60 categories**. **Do not assume these are the exact counts of the selected download.** Compute and report actual counts and dataset version.
- Annotations may be COCO-style bounding boxes/segmentations. Use bounding boxes for the selected *object detection* task.
- **Licensing check required:** annotations and individual images may have different licenses. Do not claim that all images are freely reusable commercially; preserve attribution and image-level rights. Educational/research use is the intended use.
- **Dataset challenges:** sparse rare classes, imbalanced categories, small objects, inconsistent contexts, diverse source images.

### Dataset intake checklist
- [ ] Confirm download source, commit/version, date, checksums or manifest, and permitted use.
- [ ] Save a source/attribution file, `data/README.md`, and downloadable reproduction instructions; do not commit large raw data by default.
- [ ] Parse metadata; record number of images, annotation instances, distinct classes, image size distribution (min/median/max width and height), missing/corrupt files, and empty annotations.
- [ ] Visualize at least 20 random annotation overlays plus representative edge cases.
- [ ] Count instances **and images per class**; identify rare classes and inconsistent labels.
- [ ] Propose a compact class taxonomy (possibly `litter` as one detection class or 3–5 broad classes); justify it using actual class frequencies. **Ask student approval before merging/dropping classes.**
- [ ] Check whether similar frames, near-duplicates, or source groupings could leak between splits.
- [ ] Create fixed train/validation/test split with saved image identifiers; ensure all versions and augmentations of the same original remain in one split. Favor group-aware splitting if source groups can be established.
- [ ] Suggested **initial** split: 70/15/15, subject to dataset audit and class coverage. Do not assert split sizes before generating them.
- [ ] Test set is locked before experimenting; do not use it to choose preprocessing, model, epochs, confidence threshold, or augmentations.

## 5. Data processing pipeline (explicit, inspectable stages)

`source images + annotations → validation/audit → class mapping → leak-resistant split → annotation conversion → training-only transforms → train → validation → locked test → error analysis → report`

### 5.1 Data validation
- [ ] Confirm annotation boxes are valid, in bounds, nonzero area; repair only deterministic mistakes with logged changes.
- [ ] Confirm image loading works and that dimensions match annotation coordinate systems.
- [ ] Create before/after example overlays and an audit log.

### 5.2 Preprocessing
- [ ] Convert COCO annotations to selected model input format with unit tests for coordinate conversion.
- [ ] Standardize image orientation / handle EXIF safely. Keep original files immutable.
- [ ] Resize/letterbox to model input size (candidate: 640×640) **while correctly transforming boxes**.
- [ ] Normalize only as expected by the chosen model framework (avoid double normalization).
- [ ] Check for tiny objects made invisible by downscaling.
- [ ] Keep preprocessing identical across experimental arms except the one intervention being tested.

### 5.3 Augmentation — experimental intervention
- [ ] Baseline: minimal standard preprocessing; avoid untracked framework defaults that introduce hidden augmentation.
- [ ] Treatment: justified **training-only** augmentation, e.g., brightness/contrast variation, moderate scale/crop, horizontal flip when valid; configure exact values in a versioned YAML.
- [ ] Show examples with boxes transformed correctly; reject transformations that corrupt the label.
- [ ] Never augment validation/test images to inflate the primary evaluation. Synthetic robustness stress-testing, if used, is a separately labeled analysis.

### Data quality acceptance gate
- [ ] Dataset manifest and summary statistics generated.
- [ ] At least 20 annotation overlays reviewed.
- [ ] Class taxonomy and split strategy documented/approved.
- [ ] No obvious data leakage; conversion sanity tests pass.
- [ ] Training and evaluation configurations fully traceable.

## 6. CNN training and controlled comparison

### Candidate architecture
**YOLO11n** (Ultralytics) fine-tuned for litter detection is the provisional baseline. It includes CNN-based visual feature extraction. Reference: https://docs.ultralytics.com/models/yolo11 and https://docs.ultralytics.com/tasks/detect . Alternatives may be considered if compute, license, or educational clarity supports a change.

### Model selection decision
- [ ] Compare proposed model with at least one feasible alternative **on paper** (e.g., YOLO11n versus a small Faster R-CNN with CNN backbone); compare compute, complexity, measurable outputs, and assignment fit.
- [ ] Select ONE primary model and obtain student approval before implementation.
- [ ] Explain backbone, detection head, bounding-box prediction, pretrained weights, and what layers/parameters will be fine-tuned.
- [ ] Verify model/framework license for intended use; note any restrictions in README.

### Experiments
| ID | Configuration | Purpose | Required? |
|---|---|---|---|
| E0 | Fine-tuned CNN detector with standard preprocessing and controlled defaults | Baseline | Yes |
| E1 | **Same** initialization, split, epochs, and evaluation; add specified training augmentations | Test H1 | Yes |
| E2 | Optional lightweight model-size or image-resolution variation | Follow-up only if time | No |

Training parameters to record: seed, train/val/test identifiers, class taxonomy, model architecture/checkpoint, pretrained weight origin, learning rate, optimizer, batch size, epochs, early stopping, input size, augmentations, hardware, runtime, library versions, checkpoint selection criteria.

- [ ] Run a small smoke test (e.g., 1 epoch/mini-subset) before full training.
- [ ] Train E0 and E1 using only train data; select checkpoints on validation metrics.
- [ ] Save checkpoints, loss/metric curves, configuration files, and stdout logs.
- [ ] If compute allows, repeat with multiple fixed seeds and report variation. If only one run is feasible, explicitly acknowledge that limitation.
- [ ] Training may be conducted on a local GPU or Google Colab; record actual hardware. Never imply the agent executed a run unless run logs exist.
- [ ] If training fails, capture error and propose the **smallest reversible** repair; do not conceal failures or fabricate success.

## 7. Evaluation and unseen-image requirement

### Primary measures
- **mAP@0.5** and **mAP@0.5:0.95** for localization quality.
- Precision and recall at a documented confidence/IoU threshold.
- Per-class results if class support is sufficient; sample counts alongside metrics.
- Inference latency with measured hardware and input size (supporting metric, not a primary scientific result).

### Evaluation protocol
- [ ] Choose the best checkpoint using **validation only**.
- [ ] Run final E0 and E1 on the locked test split; produce side-by-side numeric table and graphical comparison.
- [ ] Run inference on additional **new unseen images** not used in training, validation, or parameter tuning; identify image provenance and any known distribution shift.
- [ ] Save at least 5–10 labeled prediction samples (successes, false positives, false negatives) where ground truth exists; distinguish visual examples from metric-bearing evaluation data.
- [ ] Discuss difficult backgrounds, small objects, occlusion, skewed class frequency, and qualitative failure cases.
- [ ] State whether H1 was supported, unsupported, or inconclusive using the recorded results; avoid causal overclaims from a single run.

### Important distinction
Do **not** call a public test image 'unseen' if it was used for model selection or previously appeared in training. If using a generic pretrained checkpoint, disclose its pretraining dataset and potential overlap/limitations; your **fine-tuning** holdout must be untouched.

## 8. Functional and non-functional requirements

### Functional
- [ ] `audit_dataset`: analyze images and labels; output reproducible counts, image dimension statistics, and visual checks.
- [ ] `prepare_dataset`: validate annotations, map classes, split, convert to detector format.
- [ ] `train`: run one specified experiment config and save metadata/results.
- [ ] `evaluate`: compute held-out metrics and export machine-readable predictions.
- [ ] `predict`: annotate specified new image(s) using saved checkpoint.
- [ ] `make_report_assets`: generate tables, charts, and selected error cases from real experiment outputs.

### Non-functional
- [ ] Deterministic seeds and saved configs when supported.
- [ ] No hard-coded private credentials or personal file paths.
- [ ] Source data immutable; avoid committing large models/datasets without intent.
- [ ] Fail loudly on missing dataset paths or corrupted annotations.
- [ ] Code understandable to student; each nontrivial script documents its purpose, inputs, outputs, and key logic.
- [ ] Tests for class mapping, coordinate conversion, split integrity, and output file existence.

## 9. Repository layout

Folders are a **proposed organization**, not a requirement to generate empty boilerplate. Start with the smallest useful structure, scaled to a 10-point individual assignment rather than the full research-lab layout below.

```text
waste-detection-cnn/
├── README.md
├── PRD.md                         # this document
├── LICENSE
├── requirements.txt
├── docs/
│   ├── literature_notes.md
│   ├── decisions.md
│   ├── agent_progress.md
│   └── protocol_phase0.md
├── data/
│   └── README.md                  # provenance, license, download steps
├── configs/
├── src/
├── notebooks/
├── outputs/
└── report/
```

## 10. Agent execution rules — staged checklist

### Global rules (the agent MUST follow)
1. **READ THIS PRD FIRST**, then the assignment requirements above and repo state; summarize proposed next step.
2. Work **one phase at a time**. After each phase, produce evidence, checklist status, unresolved issues, and a concise student decision request when needed.
3. Default to **researching** model/dataset alternatives, not silently substituting them. Every major design change requires a written rationale and student approval.
4. Explain important code blocks and their intent so the student understands and can defend the work.
5. Distinguish **proposed**, **implemented**, **executed**, **measured**, and **approved**. Do not mark a checkbox done without actual evidence.
6. Do not write a fabricated final paper or fake citations. Summarize real observations for the student's own analysis and writing.
7. Do not leak or commit datasets or credentials; respect data licenses.
8. No additional robotic-navigation or hardware work until all graded assignment requirements are satisfied and the student explicitly authorizes extension.
9. Record decisions in `docs/decisions.md` and progress in `docs/agent_progress.md`. For each stage, save the exact outputs needed to reproduce it.
10. If uncertain, suggest 1–2 alternatives and request a targeted decision instead of making an irreversible change.

### Phase 0 — Understand requirements and formulate RQ
- [ ] Read assignment and PRD; restate the professor's five steps and report items.
- [ ] Inspect current repo and environment.
- [ ] Conduct a focused literature scan and assess RQ1 versus up to two better alternatives.
- [ ] Draft research hypothesis, experiment comparison, variables and success measures.
- [ ] **Gate 0:** Student approves research question, scope, and experiment approach.

### Phase 1 — Dataset investigation and acquisition
### Phase 2 — Data preprocessing
### Phase 3 — Architecture selection and baseline
### Phase 4 — Controlled augmentation experiment
### Phase 5 — Locked evaluation and new images
### Phase 6 — Report package and student review
### Phase 7 — Portfolio extension (OPTIONAL, after grading deliverables)

(Full phase detail preserved from original planning doc; see `docs/agent_progress.md` for live status.)

## 11. Assignment report blueprint (mapped exactly to professor's wording)

| Report section | Professor's requirement | Expected evidence |
|---|---|---|
| 1. Identified domain | 'Your response to step 1' | Environmental robotics / litter detection; motivation |
| 2. Research focus and reasoning | 'Discuss your research focus and reasoning for its selection' | Object detection; why bounding boxes rather than classification/segmentation |
| 3. Dataset description | 'Number of data points, size, dimensions, etc.' | Actual image and annotation counts, dimensions, class count/distribution, dataset origin, split |
| 4. CNN training technique and code | Recommended code and output/snapshot | Chosen CNN architecture, preprocessing, learning setup, config, code references and real loss/validation plots |
| 5. Unseen-data results | 'Discuss the results that your model produced for any unseen data' | Test metrics, annotated genuinely unseen examples, errors, interpretation |

## 12. Definition of done

### Minimum acceptable (course assignment)
- [ ] Domain and object detection focus explicitly justified.
- [ ] Real dataset used; image and annotation statistics documented.
- [ ] CNN actually trained/fine-tuned, with real code and run evidence.
- [ ] Trained model tested on unseen data, outputs saved and interpreted.
- [ ] Report contains all five professor-required elements.
- [ ] Deadline and AI-use rules verified; student reviews final submission.

## 13. References to verify and cite in the eventual report

1. Uploaded *CSCI 441/541 Research Assignment* — authoritative source for the five assignment steps and required report content.
2. Uploaded *CSCI 441/541 Neural Network and Deep Learning* syllabus — AI policy and course context.
3. TACO official repository: https://github.com/pedropro/TACO
4. TACO dataset listing and class information: https://huggingface.co/datasets/pictograph/taco
5. Ultralytics YOLO11 model documentation: https://docs.ultralytics.com/models/yolo11
6. Ultralytics object detection documentation: https://docs.ultralytics.com/tasks/detect

See `docs/literature_notes.md` for the Phase 0 literature scan (added after initial planning).
