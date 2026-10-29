# Lab reports published with APA paper v2

The APA v2 paper (`docs/APA_PAPER_V2_DRAFT.md`, Appendix A) cites reports from the project's lab fork. That is a local
experimental clone, never pushed. This commit publishes those reports at the paths the paper cites.

- **Provenance.** `artifacts/apa_v2_lab_reports_manifest.json` lists each file with:
  - its source lab branch and commit;
  - the sha256 of the original;
  - the sha256 of the published copy;
  - every substitution made.
- **Redaction for the public repository.** Six of the 184 files were changed, by substitution only:
  - a personal first name in process notes became "the author";
  - local scratch paths became `<local scratch>`.

  No number, result or verdict was changed. The other 178 files are byte-identical to the lab originals.
- **Scope.** These are working records: orders, ledgers, reports and receipts, written as the work happened. They
  include failed gates, refuted predictions and process errors, kept as recorded.
