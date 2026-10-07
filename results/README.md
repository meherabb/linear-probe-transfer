# Complete result tables

`all_tables/` contains the 123 CSV tables embedded in the executed notebook, totaling 57,047 data rows. This includes the per-pair tables that are not part of the smaller curated CSV export. The notebook records a SHA-256 checksum for the compressed table payload; `scripts/export_embedded_tables.py` verifies that checksum during recovery.

Tables beginning with `legacy_v2_` are retained for provenance and are labeled as pilot outputs in the notebook. Do not combine them with the fresh v3 paper-profile results.
