# Data decisions

## 2026-09-27 — Initial dataset foundation

- **Phase:** dataset plus a private local pose-guided visual approximation; marketplace, production model training/inference, and fitting/measurement claims remain out of scope.
- **Pilot category:** adult women's stitched Pakistani straight-cut suits/kameez with trousers.
- **Visual try-on goal:** show how the exact selected suit appears on the same person, including design details, natural drape, silhouette, and ordinary movement, so the shopper can judge visual appeal.
- **Fit-language boundary:** “looks good on me” is a visual appearance judgment; the dataset must not claim correct size, comfort, measurements, tailoring, stretch, or physical fit.
- **Dupatta policy:** record presence when captured, but do not mix moving-dupatta takes into the initial target without an explicit scope decision.
- **Primary unit:** one wearing take, linked to one exact garment and one participant reference.
- **Ground truth policy:** only real footage of the same consenting participant wearing the exact garment is verified paired data. Similar garments and generated views are separate collections.
- **Privacy policy:** use opaque IDs; keep contact mappings and consent evidence outside committed manifests.
- **Rights gate:** commercial model-training permission must be verified before approval.
- **Source policy:** preserve originals and checksums; all preprocessing is non-destructive and traceable.
- **Split policy:** keep clips, sessions, people, garment families, near-duplicates, and derivatives grouped. Strict person/garment-disjoint evaluation is deferred until sample counts support it.
- **Implementation order:** documentation and synthetic example first; source inspection/checksum/`ffprobe` tooling next; rights-aware validation and previews after that.
- **Prototype boundary:** the private camera page uses browser pose landmarks to approximate garment scale, rotation, and placement. It must not be described as a trained try-on model, exact cloth simulation, a size/fit predictor, or approved training data.
- **Real try-on decision:** begin with an offline still-image virtual try-on checkpoint using a person image plus an exact garment image. A person video is a validation input, not a general training set. Live video is deferred until still-image fidelity, rights, and runtime performance are verified.

These are proposed pilot decisions and must be updated when real capture evidence or owner-approved scope changes justify a revision.

## 2026-09-29 — Training versus deployment runtime

- CUDA is allowed for private model training, checkpoint evaluation, and other measured research work when it improves iteration speed.
- The future customer-facing API/frontend runtime must not depend on the training machine's CUDA installation. It should use a validated portable deployment path, such as ONNX/ncnn/WebGPU or another measured runtime selected after export experiments.
- A portable export is not yet validated, and this decision does not authorize production deployment or claims of live video try-on.

## 2026-09-29 — Owner approval recorded

- The owner approved the visual try-on objective: the selected exact suit should appear naturally worn by the person, with its visible design details, drape, silhouette, and movement preserved as faithfully as the validated method allows.
- “Fitting” is interpreted as visual wearing appearance only. The project still must not claim measured size, comfort, tailoring accuracy, stretch behavior, or physical fit.
- The pilot scope, adult-participant requirement, commercial-training rights requirement, garment-permission requirement, and privacy boundaries were approved.
- Restricted consent-record ownership, storage, retention, withdrawal contact, and access reviewers remain an operational prerequisite before real media enters an approved manifest.

## 2026-09-29 — Full-video calibration capture

- The initial personalization/calibration artifact is a **whole moving video**, not a single frame. The sequence can contain many frames and must retain timing information for later temporal analysis.
- The browser prototype now separates two explicitly selected capture stages: a person reference in neutral clothing and the same person wearing the exact garment.
- Camera startup alone does not save data. The participant must click **Start full-video recording** and then **Stop & save full recording**. The browser downloads a local WebM and JSON frame-timing sidecar; no camera data is uploaded automatically.
- A captured video is a calibration/data candidate, not automatic proof of measurements, physical fit, or model quality. A private visual profile may be investigated later; general model training requires rights review, QA, and more varied people/garments.
- No real calibration videos have been confirmed in the repository or Downloads as of this decision.
