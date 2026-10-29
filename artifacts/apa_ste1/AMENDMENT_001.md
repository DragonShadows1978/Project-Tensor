# Source census/build completeness amendment (registration unchanged)
The base deterministic-support .cu files and deterministic_gather.h were absent
from the delivered worktree. Restore exact bytes from the base engine's source;
receipts/restored_source.json records provenance and hashes. No new algorithm.
The actual census also includes bk3 and TF32 key-owner kernels; STE covers both.
Variant c is an explicit deletion diagnostic unavailable to the train setter:
keep its existing zero selected dK/dV, add only the unselected dK contribution
when STE is on. Its dQ and zero dV stay unchanged; refine-all remains equal.
GPU correctness calibrates c against this explicit ablation reference.
Prior art: inherited Project-Tensor diagnostic c (2026); no new algorithm.
