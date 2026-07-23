# Phase 4 — Historical environment recovery

## Status

A minimal CPU execution path has been recovered for the archived region-similarity implementation. This is a compatibility reconstruction, not proof of the exact original machine environment or reproduction of the published results.

## Verified compatibility environment

The GitHub Actions smoke test uses:

- Ubuntu 22.04 CPU runner
- Python 3.8
- PyTorch 1.8.1 CPU
- torchvision 0.9.1 CPU
- FastAI 2.3.1
- FastCore 1.3.20
- spaCy 2.3.1
- NumPy 1.20.3
- SciPy 1.6.3

The complete dependency set is stored in `environment/legacy-requirements.txt`. The original repository did not include a lockfile, so these versions were selected from the period of the notebooks and verified through GitHub Actions.

## Verified execution path

The command below has been verified on CPU:

```bash
python scripts/smoke_test.py
```

It:

1. constructs a reduced one-layer historical ARViT instance;
2. divides synthetic images into 16 × 16 regions;
3. calculates per-region Gram matrices and the normalized historical distance tensor;
4. completes a forward pass and verifies logits and attention dimensions;
5. evaluates the historical `ARViT_Loss`;
6. performs backward propagation and one SGD optimizer step.

The following command has also been verified:

```bash
python scripts/smoke_test.py --imagenette
```

It downloads Imagenette-160 and repeats the one-batch training smoke test using real images.

## Installation

```bash
python -m pip install pip==23.3.2 setuptools==68.2.2 wheel==0.41.3
python -m pip install -r environment/legacy-requirements.txt
```

## Historical launch scripts

The archived launch scripts are not used by the smoke harness. They retain assumptions including:

- CUDA and NCCL distributed execution;
- FastAI distributed and FP16 training;
- machine-specific `Path.home() / 'Luiz/...'` data and output paths;
- datasets and serialized learners that are not included in the repository.

These assumptions are preserved and will be addressed during later modernization.

## Checkpoint status

Run:

```bash
python scripts/check_checkpoints.py
```

The entries under `pretrained_models/` are historical Git LFS pointers. The binary objects are not present in the migrated repository, so checkpoint loading cannot currently be verified.

## Scope limitations

The successful smoke uses `gm_patch=16`, where the region grid matches the 16 × 16 attention-token grid. This phase does not validate the published 32 × 32-region path. Phase 2 identified that the historical code does not implement the required attention-map reduction for that configuration.

This phase also does not establish:

- exact reproduction of the original CUDA environment;
- full pretraining or fine-tuning;
- published accuracy reproduction;
- checkpoint equivalence;
- correction of the MSE, bias, reduction, or scaling discrepancies identified in Phase 2.

No archived model, regularizer, loss, or launch-script behavior was changed.
