# Eastern-Wear Video Dataset

This repository is the dataset foundation for a future live video clothing-appearance visualization product for Pakistani/eastern fashion.

## Current status

**Phase: dataset + private local prototype.** The repository defines the collection protocol, rights gates, metadata contract, annotation direction, and a body-aware camera-overlay approximation. It does not contain approved real participant media and is not ready for model training.

The experimental real try-on integration is documented in [`docs/REAL_TRYON_STATUS.md`](docs/REAL_TRYON_STATUS.md). The FASHN VTON source is present locally, but model weights, the backend endpoint, and live-camera generation are not complete.

The intended pilot covers adult women's stitched Pakistani straight-cut suits/kameez with trousers. Moving dupattas and other garment categories are outside the initial target. The future system is intended to visualize appearance, not predict size, measurements, tailoring, stretch, or physical fit.

## Intended visual try-on outcome

The future customer experience should show the shopper how the **exact selected suit** may appear on them: its colors, print, embroidery, neckline, sleeves, silhouette, natural drape, and appearance during ordinary movement. The purpose is to help someone judge whether the design looks good on them visually. It must not claim that the displayed result proves the correct size, comfort, measurements, tailoring, stretch, or real-world physical fit.

## Private visual check

The local prototype uses browser pose tracking to automatically scale, rotate, and place the supplied garment cutout on the user's camera feed. It does not upload or save camera frames. It is still only a visual approximation, not exact cloth simulation, size prediction, or physical-fit validation. Run it with the instructions in [`prototype/README.md`](prototype/README.md).

## What belongs in the dataset

An approved wearing-take record connects all of the following:

1. A product/garment photo set for one exact garment design.
2. A reference photo or short clip of a consenting adult in ordinary neutral clothing.
3. An original video of that same person wearing that exact garment.
4. Stable opaque IDs, checksums, technical metadata, provenance, and rights approval.

A similar-looking outfit, an unrelated wearer, or an AI-generated view is not verified paired ground truth. Unpaired or synthetic material must remain in a separately identified collection.

## Repository rules

- Keep original media and consent records on access-controlled local storage.
- Do not commit faces, videos, consent forms, names, contact details, signatures, addresses, secrets, or unauthorized product media.
- Never overwrite or delete originals during inspection or preprocessing.
- Treat automated quality checks as review flags; human review decides whether a sample is acceptable.
- Keep identity-to-contact mapping outside manifests and outside this repository.
- Record commercial model-training permission before a source can enter an approved manifest.

See:

- [`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md) — schema and validation rules.
- [`docs/CAPTURE_PROTOCOL.md`](docs/CAPTURE_PROTOCOL.md) — repeatable capture instructions.
- [`docs/CONSENT_AND_RIGHTS.md`](docs/CONSENT_AND_RIGHTS.md) — permission and withdrawal requirements.
- [`docs/ANNOTATION_GUIDE.md`](docs/ANNOTATION_GUIDE.md) — labels and review workflow.
- [`docs/DATA_DECISIONS.md`](docs/DATA_DECISIONS.md) — decisions and their rationale.
- [`docs/LOCAL_SETUP.md`](docs/LOCAL_SETUP.md) — verified local FFmpeg setup.
- [`docs/OWNER_APPROVAL_CHECKLIST.md`](docs/OWNER_APPROVAL_CHECKLIST.md) — owner decisions required before real capture.
- [`data/manifests/example_clip.jsonl`](data/manifests/example_clip.jsonl) — fictional schema example only.

The local inspection, rights-aware validation, and preview foundations are implemented. No real data should be collected into this repository until the rights and capture process has been reviewed.

## First inspection tool

The first read-only CLI is implemented. It scans a directory, hashes files, identifies duplicate checksums, and extracts selected technical video metadata through `ffprobe` when FFmpeg is installed. It never edits or copies source media.

After installing this repository locally, run a dry-run report with:

```powershell
$env:PYTHONPATH = 'src'
python -m dataset inspect --input <source-directory> --dry-run --output inspection.json
```

If `ffprobe` is not on `PATH`, video records are retained but flagged as requiring review. The command does not approve files for the dataset and does not replace rights verification or human QA.

Manifest validation is also implemented:

```powershell
python -m dataset validate --manifest <manifest.jsonl> --dataset-root . --output validation.json
```

Use `--require-approved` as the stricter release gate, together with `--rights-registry`. It rejects non-fictional records without verified rights, an active restricted rights record, required permissions, source checksums, passed QA, and complete existing file links. The fictional example is intentionally reported as documentation-only and is never eligible for approval.

Preview generation is available for verified candidate/approved records:

```powershell
python -m dataset preview --manifest <manifest.jsonl> --dataset-root . --output-root data/derived/previews --sample-fps 3 --report-output data/reports/preview.json
```

Each derived frame records the source presentation timestamp and source frame number when available. Rerunning with the same source checksum and sample rate skips existing output rather than overwriting it.

Coverage and grouped splitting are also implemented:

```powershell
python -m dataset report --manifest <manifest.jsonl> --dataset-root . --output data/reports/coverage.json
python -m dataset split --manifest <manifest.jsonl> --dataset-root . --output-manifest data/splits/v0.1.0.jsonl --seed 42 --report-output data/reports/split.json
```

The split tool keeps person IDs, garment families, sessions, and their takes together. A tiny pilot may produce a warning instead of meaningful validation/test coverage; it must not be described as statistically reliable.
