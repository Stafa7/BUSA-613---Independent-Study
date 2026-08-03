# Dataset Acquisition Manifest

Acquisition date: 2026-06-25  
Storage location: `data/raw/`

This manifest records the official dataset repositories downloaded for the independent study. The repository contents are preserved unchanged. Subsequent cleaning and transformation must be written outside these directories.

## CoAID

- Official source: <https://github.com/cuilimeng/CoAID>
- Local path: `data/raw/coaid/`
- Git commit: `d238224346781255e1e7e6ed8bc410c2b2e6329e`
- Nearest release: `V0.4-2-gd238224`
- Downloaded size: approximately 18 MB, including Git metadata
- Repository status after download: clean
- README SHA-256: `a62c5599257391782c3cd634a36772594bf75ed53d7a4d8b71e4fbf83f22e5f7`

### Available data

The repository contains four dated collection directories: `05-01-2020`, `07-01-2020`, `09-01-2020`, and `11-01-2020`.

Observed CSV records, excluding headers:

| Record type | Count |
|---|---:|
| Reliable news | 4,532 |
| Unreliable news | 925 |
| Reliable claims | 490 |
| Unreliable claims | 28 |
| Tweet-ID links | 160,875 |
| Reply-ID links | 135,877 |

The exact usable counts may change after parsing, duplicate removal, missing-text checks, and label-provenance review.

## FakeHealth

- Official source: <https://github.com/EnyanDai/FakeHealth>
- Local path: `data/raw/fakehealth/`
- Git commit: `ec9379de8f8f13af8c436dd6dd9bfaddacd2df30`
- Downloaded size: approximately 68 MB, including Git metadata
- Repository status after download: clean
- README SHA-256: `9b93fcd234f4b8ece5911c8a6cdcfea8a422e645c8caefcf8211b32d681dbf67`

### Available data

| Component | HealthStory | HealthRelease |
|---|---:|---:|
| Downloaded article-content files | 1,638 | 599 |
| Expert review records | 1,690 | 606 |
| Items with engagement-ID mappings | 1,690 | 606 |
| Tweet IDs | 384,073 | 47,338 |
| Retweet IDs | 120,709 | 16,959 |
| Reply IDs | 27,601 | 1,575 |

Review-file checksums:

- `HealthStory.json`: `4d9db2ec053888234313e8ef9a8c999fe9b57cbc3dea8c0181c217827256fd7c`
- `HealthRelease.json`: `f0c800154e29f1df8599b91c337ae362641083aa44e3c93f3a75186b861dcd1f`

The repository includes article text, metadata, expert reviews, ratings, review criteria, and engagement identifiers. The engagement mappings contain platform IDs rather than complete post objects.

## Acquisition decision

The optional FakeHealth Zenodo archive was downloaded because the project now preserves all publicly available FakeHealth components locally. It is approximately 2.2 GB and primarily supplements the repository with follower and following identifiers. Those network records do not provide hydrated post content and do not change the core no-live-API study design.

Optional Zenodo archive metadata:

- Record: <https://zenodo.org/records/3862989>
- File: `FakeHealth.zip`
- Local archive path: `data/raw/fakehealth_zenodo/FakeHealth.zip`
- Local extraction path: `data/raw/fakehealth_zenodo/extracted/`
- Expected size: `2,205,934,463` bytes
- Expected MD5: `1dd710f663694096ba604144ad7e4930`
- Verification status: passed
- Extracted size: approximately 4.9 GB
- User follower JSON files: `255,235`
- User following JSON files: `255,229`
- Total user-network JSON files: `510,464`

The project will begin with the official repository data already downloaded. It will not depend on the X API or live rehydration. Engagement-feature feasibility will be assessed from locally available fields and identifier counts during Phase 3 of the execution plan.

## Next checks

1. Parse each dataset and reconcile records against this manifest.
2. Create field-level data dictionaries.
3. Audit text availability, duplicates, missingness, labels, sources, and join keys.
4. Determine whether engagement identifier counts alone support meaningful features without API access.
5. Complete the cross-dataset label and text-unit compatibility assessment.
