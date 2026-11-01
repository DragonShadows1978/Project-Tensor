# GN1-REV2 redaction note

The remote is public (L21). In `hostile_rev2_a.json`, `hostile_rev2_b.json`, `hostile_rev2_c.json` and
`hostile_rev2_d.json`, the installed runner's absolute path (under the user's home directory) was replaced by
`forge-gpu-run(installed)` after the runs. Nothing else was changed. The scripts resolve the runner with
`command -v forge-gpu-run` and contain no home path.
