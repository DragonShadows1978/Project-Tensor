# Lead harness completeness, before any GPU measurement
Add forward h/g1 and h_tf32/g1_tf32 cases so every selective-training forward
family is exercised in L-G1/L-G2, not solely source-compared. Twenty-four small
cases per capture remain far below the registered 300s budget estimate.
L-G4 verifies from stored thresholds that all visible pairs are selected,
including the exact singleton. Worker imports pin the exact base/arm so SHA.
An EXIT verifier writes RED gate results even on failed/timeout commands.
No GPU execution or tolerance measurement has occurred. No algorithm change.
