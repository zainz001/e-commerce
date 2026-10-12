# Annotation guide

Status: proposed; manual annotation begins only after the pilot identifies a concrete need.

## Priority

First review the alignment and quality that make a sample useful:

1. Is the product photo the exact garment worn in the video?
2. Is the reference image the same consenting participant?
3. Are the source rights verified for the intended use?
4. Is the garment visible enough through the take's movement?
5. Are colors, print, embroidery, neckline, sleeves, and silhouette reviewable?

Do not spend effort labeling structures that are not justified by a dataset experiment.

## Controlled labels

Use clip-level or time-range labels where possible:

- pose/motion: `front`, `three_quarter`, `side`, `left_turn`, `right_turn`, `arms_raised`, `walking`, `stationary`;
- occlusion/quality: `hand_over_garment`, `hair_over_collar`, `out_of_frame`, `motion_blur`, `poor_lighting`, `background_interference`;
- product visibility: visible front/back view, print/embroidery visibility, and relevant garment detail regions.

Masks, keypoints, or garment-part annotations are optional and require a documented experiment showing why they are needed. If video tracks are used, retain keyframes and human review; interpolation is an aid, not ground truth.

## Annotation record requirements

Each annotation export must include schema version, source `clip_id`/`garment_id`, original checksum, annotator/reviewer pseudonymous IDs, timestamps or source frame references, label vocabulary version, and review status. Retain the native annotation-tool export when one is used.

## Prohibited labels

Do not annotate identity recognition, biometrics, skin-tone scores, attractiveness, body-shape scores, measurements, or physical fit. The dataset represents visual appearance observations, not a promise of real-world sizing or tailoring.
