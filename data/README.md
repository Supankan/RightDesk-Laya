# Data workspace

- `raw/`: downloaded source files, kept unchanged.
- `processed/`: prepared records and split assignments.
- `manifests/`: versioned source metadata and transformation provenance.

Raw, processed, and generated manifest contents are ignored by Git. Manifests record source revision,
license, checksums, and transformation rules. Later-stage fields such as queue
mapping and calibration split remain explicitly null until those stages run.
Store labels separately from model inputs.

Phase 2 downloaded ABCD v1.1 locally and prepared 10,042 tickets. Phase 3 prepared
8,722 active cases across four partitions, with 1,320 documented duplicate
exclusions and all 1,004 official test cases retained. Queue mapping v1 is
provisional. Phase 7 uses the reserved calibration partition in three disjoint
roles for fitting, threshold selection, and validation; the final test stays
unevaluated. Both generations of data and provenance remain available. The manifests
are portable provenance; ignored data files must be downloaded/prepared again
on another machine. See `docs/data-plan.md` for commands and boundaries.
