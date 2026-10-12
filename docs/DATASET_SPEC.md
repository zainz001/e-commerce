# Dataset specification

Status: proposed foundation, schema version `0.1.0`.

This document defines the minimum record needed to connect an exact garment, a consenting person reference, and an original video of that person wearing the exact garment. It describes observed visual data; it does not describe size or physical fit.

## Intended output

The dataset should support a future visual try-on experience: showing the exact selected suit on the same person with recognizable appearance, product details, natural-looking drape, and continuity during movement. “Looks good on me” is a visual judgment about design and appearance. It is not a size recommendation or evidence of physical fit.

## Scope

The pilot category is `womens_stitched_straight_cut_suit`: a stitched kameez/shirt with trousers, initially without a moving dupatta. A captured dupatta must be recorded, but the take is outside the initial target if it materially changes the task.

The core record is one wearing take. All sampled frames, previews, annotations, and reports derived from a take must retain its `clip_id` and original checksum.

## Required identity and linkage fields

| Field | Requirement |
| --- | --- |
| `schema_version` | Exact schema version, currently `0.1.0`. |
| `record_status` | `fictional_example`, `candidate`, `approved`, or `withdrawn`. |
| `clip_id` | Opaque unique ID for the wearing take, for example `clip_000001`. |
| `person_id` | Opaque voluntary-capture ID; never a name or contact identifier. |
| `garment_id` | Opaque ID for the exact product/design. |
| `garment_family_id` | Group for the same design and near-identical variants, used for leakage-safe splits. |
| `session_id` | Opaque capture-session ID; all same-session takes stay together. |
| `take_id` | Stable take ID within a session. |
| `category` | Controlled category value. |
| `garment_front_path` | Relative path to the source product front image. |
| `garment_back_path` | Relative path or `null` when not collected; never invent a back view. |
| `reference_image_path` | Relative path to the participant reference image or clip. |
| `worn_video_path` | Relative path to the unchanged original wearing video. |
| `rights_record_id` | Restricted rights/consent registry reference. |
| `rights_status` | Must be `verified` before approval; `unknown`, `pending`, `restricted`, and `withdrawn` are not approved. |

## Technical metadata

`original_sha256`, `width`, `height`, `fps_num`, `fps_den`, `duration_seconds`, and `frame_count` are populated from the actual source after inspection. Until measured, they must be `null`; placeholders or inferred measurements are invalid.

Video inspection must also preserve codec, pixel format when available, rotation/orientation, variable-frame-rate information, decode warnings, and source presentation timestamps in a structured report.

## Labels and QA fields

- `pose_tags`: controlled clip-level or take-level tags such as `front`, `three_quarter`, `left_turn`, `right_turn`, `side`, `arms_raised`, and `walking`.
- `dupatta_present`: boolean observation; it is not an approval decision.
- `lighting_group`: capture grouping, not a quality score.
- `occlusion_tags`: controlled observations such as `hand_over_garment`, `hair_over_collar`, `out_of_frame`, `motion_blur`, and `poor_lighting`.
- `source_type`: how the source was obtained, such as `self_recorded_with_consent`.
- `annotation_status`: `not_started`, `in_progress`, `reviewed`, or `rejected`.
- `qa_status`: `pending`, `passed`, `flagged`, or `rejected`.
- `split`: `null` until deterministic grouped splitting is implemented; then `train`, `validation`, or `test`.

## Validation rules

The future validator must reject records with:

- missing or duplicated IDs;
- absolute paths, path traversal, or paths outside the dataset root;
- missing referenced files;
- non-verified rights;
- a missing person/reference/garment/video linkage;
- invalid enum values or malformed JSONL;
- missing checksums after source inspection;
- split leakage across a person, garment family, session, near-duplicate group, or derived clip.

The validator must report all failures without changing or deleting source media.

## Provenance and derivatives

Every derived frame, preview, contact sheet, annotation, and report must reference the source `clip_id` or `garment_id`, original checksum, source timestamp/frame number where applicable, transformation description, tool/software version, and creation time. Derivatives are not independent ground truth.
