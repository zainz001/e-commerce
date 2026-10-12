# Consent and rights policy

Status: proposed minimum gate for the pilot. This is an operational policy, not legal advice; obtain appropriate local legal review before collection or commercial use.

## Approval requirement

Nothing enters an approved manifest unless its restricted rights record confirms:

- the participant knowingly consented to the actual capture and storage;
- the participant consented to the intended model-training/development use;
- commercial training and any planned commercial product use are explicitly covered when those uses are intended;
- the garment/product images and any third-party media have documented permission and license terms;
- retention, access, sharing restrictions, and a withdrawal contact are recorded;
- the source is not a minor, an unconsenting bystander, or an unauthorized scrape.

`rights_status: verified` means the restricted record has been checked. It does not mean that permission is permanent or unlimited.

## Restricted registry

Store the rights registry and consent evidence under access-controlled `private/consent_records/` or an equivalent protected location. Do not put names, phone numbers, email addresses, signatures, identity documents, or contact mappings in committed JSONL manifests. Public manifests use only opaque IDs and `rights_record_id`.

Each rights record should identify, at minimum:

- source and responsible owner;
- participant/garment references using restricted identifiers;
- permission scope, including commercial model training if applicable;
- allowed derivatives and sharing restrictions;
- capture and authorization dates, expiry if any, and reviewer;
- withdrawal instructions and current withdrawal status.

The machine-readable restricted registry should provide `rights_record_id`, `withdrawal_status`, `participant_consent` flags for `capture`, `storage`, `quality_review`, `model_training`, and `commercial_training`, plus `garment_permission` flags for `product_images` and `commercial_use`. Keep any names, contact details, signatures, or identity mappings outside the registry file used by tooling.

The validator can cross-check a manifest against this registry with `--rights-registry`. Approved validation requires the registry, an active withdrawal status, all required permissions, and explicit commercial-training permission.

## Withdrawal and expiry

When permission is withdrawn or expires:

1. Mark the rights record withdrawn/expired and stop new processing.
2. Quarantine the original and identify all manifests, derivatives, previews, frames, annotations, reports, and backups that came from it.
3. Mark affected dataset versions for remediation and prevent re-ingestion.
4. Record what was removed or retained and why under restricted access.

Do not promise automatic removal from an already-trained model; that requires a separate technical and governance process.

## Safety boundaries

Do not build face recognition, biometric identification, attractiveness scoring, skin-tone scoring, or body-shape scoring. `person_id` is only a voluntary capture-session link. Avoid bystanders and unnecessary personal information, and provide a practical way for a participant to ask questions or withdraw.
