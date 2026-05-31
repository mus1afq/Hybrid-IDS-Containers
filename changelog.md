# Changelog

The repository has a single commit because the final submission required a clean upload to the institutional archive. This document records the actual build history.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Dates are approximate.

---

## [1.0.0] — May 2026 — Final submission

### Added
- End-to-end test runner (`tests/run_all_tests.sh`) covering both pipelines, fusion, and metric regeneration.
- Runtime evaluation across six Docker scenarios with full demo scripts in `demo/`.
- N-window sensitivity analysis showing peak F1 at N=15s.
- Full results plots: confusion matrices, ROC curves, N-window F1 curve.

### Changed
- Consolidated path management into `src/common/config.py` to eliminate hardcoded paths across the codebase.
- Renamed legacy scripts to follow `verb_object_modifier.py` convention.

### Fixed
- Schema alignment mismatch between training and runtime network feature vectors. Runtime captures used a different column order that silently degraded RF predictions before the fix.
- Syscall feature extraction now handles partial sysdig event lines instead of dropping the window.

---

## [0.9.0] — April 2026 — Runtime evaluation

### Added
- Six demo scenarios: nginx baseline, Python REST baseline, reverse shell, cryptominer, privilege escalation, exfiltration.
- `run_and_capture.sh` orchestrating Docker workload launch, sysdig capture, and tshark capture in parallel.
- Per-scenario evaluation script producing the runtime confusion matrix.

### Findings
- Runtime accuracy held at 66.7% across the six scenarios. Two missed detections traced to distribution shift between CHIDS training data and modern attacker syscall patterns.
- Documented as the central limitation of the project rather than buried.

---

## [0.8.0] — March 2026 — Fusion layer

### Added
- Deterministic late-fusion rule combining syscall and network probability streams.
- Threshold tuning script sweeping p_sys and p_net in 0.05 steps on the validation split.
- Synthetic hybrid evaluation producing F1 of 0.898.

### Decisions
- Chose deterministic rule over a learned meta-classifier. Rationale: explainability for the dissertation, no risk of overfitting a third model to a small validation set.
- Syscall threshold set at 0.60 for the "high confidence override" path.
- Network threshold set at 0.50 combined with a syscall floor of 0.10 to filter bursty-traffic false positives observed on benign workloads.

---

## [0.7.0] — February 2026 — Network model

### Added
- Random Forest classifier trained on Bot-IoT network flow features.
- Feature alignment between training schema and runtime tshark output.
- Network-only evaluation producing F1 of 0.841 on the synthetic test set.

### Decisions
- Selected Random Forest after benchmarking against Logistic Regression, SVM, and XGBoost on the same split. XGBoost gave a marginal F1 improvement but introduced a dependency that was not justified by the gain.

---

## [0.6.0] — January 2026 — Syscall model

### Added
- Logistic Regression classifier trained on CHIDS syscall feature vectors (124 `evt_*` columns plus `total_events`).
- StandardScaler persisted alongside the model for inference-time consistency.
- Syscall-only evaluation producing F1 of 0.719.

### Decisions
- Logistic Regression chosen for interpretability. Coefficients map directly to syscall categories, which fed into the dissertation discussion section.
- 125-feature schema fixed at this point and treated as a hard contract for the runtime extractor.

---

## [0.5.0] — December 2025 — Data pipeline

### Added
- CHIDS preprocessing pipeline: cleaning, label normalisation, train/validation/test splits.
- Bot-IoT preprocessing pipeline: feature subsetting, class rebalancing for tractability.
- Synthetic hybrid evaluation set (n=580) pairing syscall and network samples per label.

### Notes
- The synthetic set is small and balanced by design. This is acknowledged as a limitation throughout the dissertation and README.

---

## [0.4.0] — November 2025 — Capture pipeline

### Added
- sysdig container launch script capturing host-namespace syscall events from target Docker containers.
- tshark capture wrapper writing per-container `.pcap` files.
- Raw-to-feature extractors for both telemetry streams.

### Decisions
- sysdig chosen over Falco for direct event traces. Falco's higher-level abstractions did not map cleanly onto the CHIDS feature schema.

---

## [0.3.0] — November 2025 — Architecture

### Added
- Two-stage architecture document covering data flow, model boundaries, and fusion contract.
- Repository skeleton with separated `training`, `inference`, and `evaluation` modules per modality.

### Decisions
- Late fusion over early fusion. Each modality keeps its own feature representation and classifier; the fusion layer only sees probabilities.

---

## [0.2.0] — October 2025 — Literature review

### Completed
- Survey of container-aware IDS literature covering syscall, network, and hybrid approaches.
- Comparison of CHIDS, ADFA-LD, and Bot-IoT as candidate datasets.
- Justification of dataset selection: CHIDS for container-specific syscall behaviour, Bot-IoT for IoT-adjacent network traffic patterns representative of microservice noise.

---

## [0.1.0] — September 2025 — Scoping

### Completed
- Problem statement and research question agreed with supervisor.
- Ethics form submitted and approved (no human subjects; public datasets only).
- Project plan and Gantt chart produced.

### Research question
> Does cross-domain late fusion of syscall and network telemetry improve runtime intrusion detection over single-modality approaches in containerised environments?

---

## Notes on the single-commit repository

The submission process at Birmingham City University requires a final clean artefact. The development history lived in a private working repository across the academic year. This changelog reconstructs the meaningful milestones for anyone reviewing the public repository.

Future projects in this repository will use conventional commit history from day one.
