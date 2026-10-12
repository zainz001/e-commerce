# AGENTS.md — Eastern-Wear Video Dataset Project

> **Current phase: DATASET + PRIVATE LOCAL PROTOTYPES + EXPERIMENTAL TRY-ON INTEGRATION.** This repository is for collecting, organizing, annotating, validating, and versioning a lawful dataset, plus private local visual prototypes and an experimental image-based virtual try-on integration for the future **live video clothing-appearance visualization** model. Do not claim production virtual try-on, physical fit, or sizing while the integration is incomplete. Do not build the marketplace in this repository.

## 1. Project intent: read this before making changes

The eventual product is a Pakistani/eastern-fashion marketplace where a shopper chooses an actual garment and sees **how that specific design may look on them**, eventually through a live camera while moving. Suppliers will later manage their own storefronts, but that is **not part of this repository's current work**.

**The AI is for visual appearance, NOT size or physical-fit prediction.** It should eventually preserve:

- The person's recognizable appearance, face, hair, skin tone, general body appearance, and pose.
- The **actual product's** colors, prints, embroidery, neckline, sleeves, silhouette, and other important visible details.
- Visual continuity across movement: no random pattern changes, texture flicker, drifting garments, or disappearing sleeves.

It does **not** promise accurate garment size, exact tailoring, body measurements, fabric stretch, or real-world fit. Never describe the dataset or future model as solving those problems.

The owner has trained and deployed a separate **snooker** computer-vision model in the past. **There is no existing clothing model or clothing dataset to assume or reuse.** Start this project from a clean, documented dataset process.

### Current technical strategy

Use **Python 3.11+** to orchestrate dataset operations, **FFmpeg/ffprobe** for robust video inspection and extraction, and optional **OpenCV or PyAV** when Python needs direct frame access. Use a locally run annotation tool such as **CVAT** if it becomes useful. Make the pipeline runnable locally and reproducible; **no paid AI API, hosted dataset service, GPU, backend, or frontend is required in this phase**. A GPU may be useful later for optional annotation assistance, not for the initial dataset pipeline.

A later stage may train in PyTorch and investigate ONNX/WebGPU or another edge runtime, but **do not design the dataset around an unproven model architecture**. Preserve high-quality originals and flexible metadata so multiple future architectures can use the dataset.

## 2. Hard scope boundaries

### In scope NOW

1. Define a small, well-specified data collection pilot.
2. Record source permissions/consent and traceable provenance.
3. Ingest and checksum original garment photos, person reference photos, and **real videos of consenting adults wearing the photographed garments**.
4. Inspect video metadata; flag bad/corrupt material.
5. Generate non-destructive previews, contact sheets, timestamps, and optional extracted frames.
6. Create and validate manifests connecting **garment + person reference + actual wearer video**.
7. Design an annotation schema; begin only annotations justified by a concrete dataset need.
8. Create reproducible train/validation/test split tooling when sample counts support leakage-safe splits.
9. Produce data-quality and coverage reports.
10. Document every decision and keep this AGENTS.md current.
11. Run the private local camera prototype with a user-supplied or otherwise rights-cleared garment asset for body-aware visual checks only.
12. Integrate an appropriately licensed research virtual try-on checkpoint for private still-image experiments before attempting live video.

### Out of scope until explicitly approved

- Commercial model training, production inference, ONNX/ncnn/WebGPU export, generative live-camera rendering, and garment-generation code. Experimental local still-image inference is allowed only as documented, rights-labeled prototype work; continuous live rendering remains gated on successful still-image validation.
- NestJS/Django/other commerce APIs; supplier dashboards; subscriptions; checkout; the e-commerce website or app.
- Sizing/fitting estimation, measurements, body-shape scoring, or unrealistic claims about physically accurate cloth simulation.
- Building large-scale distributed data infrastructure or depending on external paid services.
- Scraping social media, brand sites, catalogues, or videos without clear permission and commercial training rights.

If a task would cross these boundaries, describe the dependency and implement only the dataset-related prerequisite unless the owner explicitly expands scope.

## 2a. Private local prototype boundary

The `prototype/` page is a private local camera/calibration tool, not a trained live try-on system. The experimental FASHN integration is a separate still-image path and must not be described as live or physically accurate until validated. Camera permission may be requested only after the user clicks the start button; camera-on alone must not save or upload frames. The page now has an explicit **Start full-video recording / Stop & save full recording** control. It records a complete browser WebM plus JSON frame-timing metadata only after that explicit recording action; no recording is uploaded automatically. Prototype assets must be labeled with their rights status; unverified third-party or AI-assisted assets are private-test-only and cannot enter approved training or commercial manifests. The upright pose-guided garment guide is only a temporary visibility fallback and must not be presented as fitted clothing.

## 2b. Current handoff state — 2026-09-29

- The user wants a true personalized video try-on: capture a whole moving video, use the temporal body/pose/occlusion evidence to build a private visual calibration profile, and later render the exact suit as if worn. This is not a request for a single still or a clipboard-like overlay.
- `prototype/app.js` supports camera preview, an explicit full-video recorder, reference/wearing stage selection, observed video-frame timing metadata, and local `.webm`/`.json` downloads.
- `prototype/local_server.py` serves the page and exposes a loopback-only FASHN still-image endpoint. It keeps request images in memory and does not upload or save them.
- The FASHN checkpoint and DWPose weights exist under ignored local storage. The Windows Application Control allow rule permits the local PyTorch runtime. CUDA may be used for private research/training and local evaluation; the future API/frontend deployment must not depend on server CUDA and will require a validated portable export. No generated FASHN try-on image or model training has completed.
- No real calibration recording has been confirmed in the user’s Downloads. A camera stream alone is not evidence that frames were saved; the user must explicitly start and stop the recorder.
- `node --check prototype/app.js`, `python -m py_compile prototype/local_server.py`, and `python -m pytest -q` pass; the test suite currently reports 14 passed.
- Next smallest task: capture and save one short reference video and one exact-garment wearing video, then inspect/import them with the existing dataset tools without committing personal media. Run and review the first local FASHN still-image output before evaluating temporal rendering. Do not claim live rendering, exact physical fit, sizing, or training success.

## 3. Narrow first dataset target

**Pilot category:** adult **women's stitched Pakistani straight-cut suits / kameez with trousers**, initially **without a moving dupatta**. Record dupatta presence if captured, but do not quietly mix it into the initial target. Broaden to other cuts, men's clothing, unstitched fabric visualization, and moving dupattas only after the pilot proves the data process.

**Primary supervised sample:**

- One legitimately sourced **product/garment photo set** for a specific garment ID.
- One reference photo or optional short reference clip of a consenting adult **in ordinary neutral clothing**, where their identity and visible appearance are clear.
- One or more **original, unaltered videos of that same person actually wearing that exact garment**; these are observed target sequences, not AI-generated guesses.
- Stable IDs and metadata proving which person, garment, session, rights record, and video belong together.

A product photo paired with an unrelated person wearing a *similar* garment is **not paired ground truth**. Unpaired or synthetically generated data can be stored in clearly separate collections with explicit provenance, but must never be silently labeled as verified real pairs.

**Pilot suggestion, not a production-size promise:** start with a small consented group (for example, 5–10 adults and a manageable set of garments) to test capture and ingest quality. Do not call the dataset ready for training just because those numbers were met.

## 4. Capture protocol — propose and document before collecting

Create `docs/CAPTURE_PROTOCOL.md` with a repeatable protocol. Prefer original footage captured with the participants' explicit permission:

- Capture the **real product** separately: clear front, back, and close-ups of embroidery/print, neckline, and sleeves. Keep its true colors and product/SKU linkage. Consider a flat-lay and a mannequin photo where feasible.
- Capture the same participant's **reference** image/short clip in comfortable everyday neutral clothing. No underwear or revealing baseline photos are required. Keep hair/face visible if the participant agrees.
- Capture that participant **actually wearing the exact product**, ideally with static camera, whole intended garment visible, uncluttered background, adequate even light, and good focus.
- Record movement in separate, labeled takes: front and neutral pose; gentle left/right turns; optional arms lifting/crossing; approach/step back; a short continuous turn if safe. Keep natural speed and include stationary intervals.
- A practical initial target is **1080p, approximately 25–30 fps, 10–20 seconds per take** if the capture device supports it. Preserve original FPS and resolution when those differ; never silently fabricate frames or resample raw footage.
- Capture garment ID, anonymized person ID, session/take ID, camera orientation, lighting/background group, visible occlusion, and whether the full intended garment stays in frame.
- Obtain explicit consent for the actual collection, model-training purpose, storage, quality review, any commercial use, and any planned sharing. **Do not record minors for the initial pilot.** Give participants a clear withdrawal/contact process.
- Avoid visible bystanders and unintentionally captured personal information. Turn off/location-strip unnecessary EXIF/geotags in derived material while retaining technical metadata needed for QA.

**No invented ground truth:** a single product photograph cannot establish how a garment looks from behind or how it moves. Missing back/side views must be recorded as missing or collected, not generated and mislabeled as observations.

## 5. Proposed repository layout

This is the desired direction, **not a claim that these directories already exist**. Inspect the actual repository before creating anything and adapt minimally:

```text
.
├── AGENTS.md
├── README.md
├── .gitignore
├── pyproject.toml                 # when Python tooling is created
├── config/
│   └── dataset.yaml              # capture rules, extraction settings, schema version
├── docs/
│   ├── DATASET_SPEC.md            # source of truth for schema and labels
│   ├── CAPTURE_PROTOCOL.md
│   ├── CONSENT_AND_RIGHTS.md
│   ├── ANNOTATION_GUIDE.md
│   ├── DATA_DECISIONS.md          # decisions and why
│   └── CHANGELOG.md              # changes another Codex session must know
├── src/
│   └── dataset/                   # Python importable package when implemented
├── scripts/                       # small CLI entrypoints only
├── tests/                         # synthetic non-personal fixtures
├── data/                          # NEVER commit original personal data
│   ├── raw/
│   │   ├── garments/              # original product photos by garment ID
│   │   ├── references/            # protected person reference media
│   │   └── worn_videos/           # protected original video by take ID
│   ├── derived/
│   │   ├── previews/
│   │   ├── sampled_frames/
│   │   └── contact_sheets/
│   ├── annotations/
│   ├── manifests/
│   ├── splits/
│   └── reports/
└── private/                        # NEVER commit; rights files, access-controlled
    └── consent_records/
```

Place raw media and consent records on access-controlled local storage. The public Git repository should contain **code, schemas, instructions, and synthetic tests**, not participant faces, footage, consent forms, or unauthorized product media. Configure `.gitignore` accordingly. Preserve an encrypted/controlled backup separately if required by the owner's retention plan.

## 6. Dataset IDs, manifest schema, and provenance

Use opaque IDs unrelated to real names or contact information. Keep any identity-to-contact mapping **separate, restricted, and outside committed manifests**. Use JSONL manifests initially for easy review and scripted validation; convert to Parquet later only if volumes justify it.

The core record is one **wearing take**, not one frame. Example record (illustrative schema, not evidence that the files exist):

```json
{
  "schema_version": "0.1.0",
  "clip_id": "clip_000001",
  "person_id": "person_0001",
  "garment_id": "garment_0001",
  "garment_family_id": "family_0001",
  "session_id": "session_0001",
  "take_id": "take_01",
  "category": "womens_stitched_straight_cut_suit",
  "garment_front_path": "data/raw/garments/garment_0001/front.jpg",
  "garment_back_path": null,
  "reference_image_path": "data/raw/references/person_0001/session_0001/front.jpg",
  "worn_video_path": "data/raw/worn_videos/clip_000001.mp4",
  "original_sha256": "<sha256-of-actual-source-file>",
  "width": null,
  "height": null,
  "fps_num": null,
  "fps_den": null,
  "duration_seconds": null,
  "frame_count": null,
  "pose_tags": ["front", "left_turn", "right_turn"],
  "dupatta_present": false,
  "lighting_group": "indoor_soft_01",
  "occlusion_tags": [],
  "source_type": "self_recorded_with_consent",
  "rights_record_id": "rights_000001",
  "rights_status": "verified",
  "annotation_status": "not_started",
  "qa_status": "pending",
  "split": null
}
```

`<sha256-of-actual-source-file>` is a documentation placeholder; code must compute it. Only populate measurements after inspecting real files. Use null for unknown values, never fake measurements or inferred consent. Store file paths relative to the repository's dataset root; reject path traversal, missing files, invalid IDs, and broken cross-references.

Maintain a restricted rights/provenance registry containing source, owner/participant authorization, allowed uses (especially **commercial training**), restrictions, and withdrawal status. Each derivative must trace back to source `clip_id`/`garment_id` and original checksum. On withdrawal or rights expiry, quarantine the source, find all derived copies, and flag any affected dataset versions for remediation; do not promise automatic removal from already-trained models without a separately designed process.

## 7. Video ingest and preprocessing — Python + FFmpeg

Build **small, idempotent, CLI-driven** tools. Prefer standard Python, `subprocess` with argument arrays (`shell=False`), `pathlib`, `hashlib`, `json`, and minimal extra dependencies. FFmpeg/ffprobe are command-line dependencies; use optional OpenCV or PyAV only where they provide clear value. Use `pytest` for tests.

Required ingestion capabilities:

1. Read an input directory **without changing source media**.
2. Verify approved provenance/rights before accepting source footage into the usable dataset.
3. Use `ffprobe` with machine-readable JSON to record codec, width, height, original frame-rate rational, duration, orientation/rotation, and relevant decode issues.
4. Calculate SHA-256; flag byte-for-byte duplicates and suspected visually duplicate takes for review.
5. Flag corrupt/unreadable videos, missing garment/reference links, incomplete coverage, and severely blurred or clipped material.
6. Generate short previews and contact sheets for human QA. Produce optional timestamped 2–5 fps **preview samples** without discarding full-rate originals. Training temporal sampling is a **later model-specific decision**.
7. Normalize image formats/color handling only in `data/derived/`, explicitly recording transforms, source checksum, software version, and timestamps.
8. Write structured reports rather than silently deleting, fixing, replacing, or accepting bad samples.
9. Support safe dry-runs, progress logging, retries, interrupted runs, and reruns without generating duplicate dataset records.

For every sampled video frame, store **source video ID + original presentation timestamp + original source frame number if available**. Preserve variable-frame-rate timing where relevant. Never infer elapsed time solely from a nominal FPS when timestamps are available.

`ffprobe` reference: https://ffmpeg.org/ffprobe.html

## 8. Annotation strategy — collect now, annotate deliberately

The priority is clean **person-reference / product-photo / real worn-video alignment**. Don't spend months labeling structures an eventual model may not need. Write `docs/ANNOTATION_GUIDE.md` and an annotation schema before starting manual work.

Possible labels, introduced only when justified:

- Garment class and components: kameez/shirt, trousers; dupatta **only when that category becomes part of the scope**.
- Product-identity attributes: visible print/embroidery detail region, dominant colors, front/back view availability.
- Pose/motion tags at clip or time-range level: front, three-quarter, side, turning, arm raised, walking/stepping.
- Occlusion and quality tags: hand covering garment, hair over collar, out of frame, motion blur, poor lighting.
- **Optional** keyframe person/garment masks, garment-part masks, or body keypoints if an experiment proves they are needed.

For video annotations, support a keyframe-plus-interpolation workflow with human review; never assume automated tracks or interpolated masks are ground truth. If needed, evaluate local **CVAT**, which supports video tracks, keyframes, and various shapes/masks. Export annotations in a documented, versioned format and retain the native project export when possible.

CVAT video-format reference: https://docs.cvat.ai/docs/manual/advanced/formats/format-cvat/

Do not build face recognition, biometric identification, skin-tone scoring, attractiveness scoring, or body-shape scoring. `person_id` links voluntary capture sessions only.

## 9. Splitting and leakage prevention

**Never randomly split frames from the same video.** Keep all derivatives of one clip in the same split. Also keep all takes from the same session together. The intended generalization test is to people and garments the model has **not** seen.

- Plan disjoint **person IDs** and **garment families/design IDs** between training, validation, and test when enough data exists; group near-identical colorways/design variants together to reduce garment leakage.
- Group same-shoot repeated takes, similar backgrounds, near-duplicate videos, and synthetic derivatives with the appropriate source group.
- For a tiny pilot, acknowledge that strict person- and garment-disjoint train/val/test sets may be too small for meaningful evaluation. **Do not create fake statistical confidence**; report coverage and collect more varied source data first.
- Store split assignments as versioned manifests and make split generation deterministic from an explicitly recorded seed and grouping policy.
- If future researchers need separate evaluations for unseen person vs unseen garment, explicitly create those protocols rather than mixing them into one ambiguous score.

## 10. Automated QA and human review

Quality checks should produce **flags for review**, not automatically delete personal footage or overwrite originals.

Automated QA should check: rights present; required three-part pairing; readable image/video; dimensions/FPS/duration; sha256; duplicate IDs; missing assets; out-of-frame/blank segments; sampled blur/exposure indicators; timestamp order; manifest schema; invalid or inconsistent labels; split leakage.

Human review is essential for: whether product images match the **exact** worn outfit, embroidery/print visibility, true product colors, safe/consented capture, meaningful movement, camera framing, severe occlusion, and whether the reference/target images represent the same participant.

Report per-category and per-person **coverage counts** without revealing personal identities, plus a list of insufficiently represented poses/lighting/garments to guide the next shoot. Automated measurements are screening aids, not proof of truthful visual fidelity.

## 11. Implementation order for Codex/agents

**Do not implement everything in one large change.** Inspect the repository and complete the smallest useful task in this order:

1. Create `README.md`, `docs/DATASET_SPEC.md`, `docs/CAPTURE_PROTOCOL.md`, `docs/CONSENT_AND_RIGHTS.md`, and protective `.gitignore`. Clarify consent and commercial-use requirements before any real data collection.
2. Agree on the minimal metadata schema and local layout; create one **synthetic/example record**, clearly marked fictional.
3. Implement source-file inspection + SHA-256 + `ffprobe` JSON metadata; test on small synthetic media.
4. Implement rights-aware manifest generation and schema/path validation. Nothing enters the approved manifest with unknown rights.
5. Implement non-destructive preview/contact-sheet and timestamped frame extraction.
6. Capture/import a **small real, consented** garment–person–video pilot; produce QA and coverage reports.
7. Add minimal manual annotation and export support only when the dataset pilot identifies a real need.
8. Build grouped split logic once sufficient independent people and garment families exist; verify leakage tests.
9. Freeze and document a dataset version so the next project phase can reproduce the exact inputs.

Suggested future CLI names (NOT pre-existing commands; implement only as each milestone needs them):

```text
python -m dataset inspect --input <source-directory> --dry-run
python -m dataset ingest --input <source-directory> --manifest data/manifests/clips.jsonl
python -m dataset validate --manifest data/manifests/clips.jsonl
python -m dataset preview --manifest data/manifests/clips.jsonl --sample-fps 3
python -m dataset report --manifest data/manifests/clips.jsonl
python -m dataset split --manifest data/manifests/clips.jsonl --policy person-and-garment-disjoint --seed 42
```

Never show these proposed commands in a README as executable until the corresponding command exists and has a passing smoke test. Never claim a dataset is ready just because scripts run.

## 12. Required engineering practices

- Use small, typed, documented Python modules and simple configuration instead of one giant notebook or script. Notebooks are optional for exploration only; promote accepted processing logic into tested code.
- Pin dependency versions when adding them, record FFmpeg version in processing reports, and document local setup for Windows/PowerShell as well as a portable CLI path when practical.
- Test manifest parsing, missing/bad rights, corrupted video, duplicate checksums, cross-reference integrity, extraction timestamp mapping, repeated-ingest idempotency, and leakage policies with **non-personal synthetic fixtures**.
- Make ingestion dry-run by default until the owner approves paths and rights handling; never delete or destructively modify original recordings.
- Keep all generated media outside Git. Never commit names, emails, consent signatures, personal video, exact capture address, secrets, or access tokens.
- If using external datasets, record each dataset's actual license and **verify suitability for commercial training** before importing. A research-paper download is not automatic commercial permission.
- Do not secretly upload data or use external inference, telemetry, hosted annotation, or cloud storage.
- At the end of **every change**, update `docs/CHANGELOG.md` with what changed, how to run it, tests and results, unresolved problems, and the next smallest task. Update `docs/DATA_DECISIONS.md` for schema/capture/split changes. Keep `AGENTS.md` aligned when the phase or non-negotiable rules change.
- If an assumption lacks evidence from the repository or real captured data, clearly mark it as **proposed / not validated**; never claim model quality, FPS, or commercial viability before it is measured.

## 13. Dataset phase completion gate

Do not propose beginning the marketplace or full model training merely because a folder of videos exists. A usable dataset pilot must have:

- Explicit rights/consent records and traceable source IDs for every approved item.
- Exact-garment product images linked to same-person reference media and **real** wearing videos.
- Unchanged source videos with verified checksums, technical metadata, and timestamps.
- Human-reviewed sample quality, documented exclusions, and useful pose/garment coverage.
- Deterministic, tested tooling to inspect, validate, and regenerate derived previews/manifests.
- An honest note about data gaps and whether any proposed split is actually leakage-safe.
- Versioned schema, capture protocol, annotation decisions, rights policy, and handoff documentation.

**The next project phase is a separate owner decision.** The ultimate technical gate for the future model is visual appearance preservation and convincing motion on ordinary customer devices, **not** physical fitting accuracy.
