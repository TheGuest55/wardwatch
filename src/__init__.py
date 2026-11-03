"""WardWatch Phase 2 analysis package: feature extraction, labelling, dataset build, data checks.

Runs inside the pinned `wardwatch-analysis` image (see services/analysis/Dockerfile), launched by
scripts/analysis.sh. Reads captured sessions under data/raw/sessions/<id>/ (mounted read-only) and
writes only under data/processed/ and docs/."""
