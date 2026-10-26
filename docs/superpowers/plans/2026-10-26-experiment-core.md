# Experiment Core Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement inline.

**Goal:** A runnable, modular experiment repository for code RLVR/SFT transfer and math retention.

**Architecture:** Lightweight Python experiment package composes TOML recipes and
launches pinned Open Instruct in its own environment. Data, rewards, evaluation,
diagnostics and reports remain independent modules.

**Tech Stack:** Python >=3.11 standard library, optional Hugging Face datasets,
Open Instruct (Python 3.12), Docker verifier workers.

**Spec:** `docs/superpowers/specs/2026-10-26-experiment-core-design.md`

## Global Constraints

- No GPU runs or model-weight downloads during setup.
- Keep the code modular and human-readable; avoid unnecessary tests.
- No automatic uploads or external job submission.

## Review Focus

- Malformed/nested code-test formats must fail clearly, never reward empty tests.
- Prompt duplicates must not cross training and evaluation partitions.
- Generation/scoring failures must not silently inflate reported accuracy.
- Paired comparisons must reject different protocols or example populations.
- Launch plans must preserve pinning and disable upstream publication defaults.

## Task 1: Experiment configuration and provenance

- [x] Create package metadata, four experiment configs, RLVR/SFT recipes and hardware profiles.
- [x] Add revision lock, upstream checkout/bootstrap, static argument validation and run manifests.
- [x] Verify planning never imports training dependencies or launches processes.

## Task 2: Dataset and reward boundary

- [x] Normalize Dolci rows, split/deduplicate by prompt hash and export RLVR/SFT JSONL.
- [x] Add benchmark preparation and source/split hashes.
- [x] Implement transparent fractional rewards and Docker code verifier API with local traces.
- [x] Check leakage and reward boundary cases using CPU fixtures.

## Task 3: Training observability and evaluation

- [x] Mirror upstream scalar metrics to JSONL, retain stdout and rollout traces.
- [x] Provide generation and grading commands with complete prediction provenance.
- [x] Compute pass@k, paired gains/forgetting and bootstrap confidence intervals.
- [x] Exercise an offline end-to-end demo and reject incompatible comparisons.

## Task 4: Human workflow and verification

- [x] Document setup, experimental controls, hardware overrides, resume/checkpoints and extension points.
- [x] Run focused CPU checks, lint, CLI demo and static upstream compatibility checks.
- [x] Review the completed implementation and state unverified GPU boundaries explicitly.

## Execution notes

Implemented inline in the requested empty workspace. Six configs include two broader-cohort RLVR arms in addition to the four matched method-comparison arms. The user authorized setup and CPU-only verification; GPU and Docker integration limits are documented in `docs/verification.md`.

A fresh reviewer inspected the implementation. Important findings addressed: stderr separation and fail-closed verifier integration; CPU Ray transport verified separately. Static flags now target the selected entrypoint; common tokenizer eligibility is recorded; scientific-package versions are explicit in the Docker image. No deferred review findings beyond the documented unavailable runtime boundaries.
