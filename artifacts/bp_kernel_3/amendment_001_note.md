# Preparation amendment 001 — canonical symlink normalization

Before any GPU gates. Original registration.json is unchanged.
Initial CPU tests: 55 passed in 0.85s. Kernel dry-run completed at
`dry_kernel_737075b5eb6a475aa4dabe20e1415671/receipt.json`.
Census registration failed before creating any census registration:

```
ValueError: '/home/vader/ForgeRealm-cold/GRAPA-Native-LLM/checkpoints/grapa_mla_w12288_r015_bf16_train.ckpt' is not in the subpath of '/mnt/ForgeRealm/GRAPA-Native-LLM'
```

Reason: parent ROOT uses Path.resolve(); canonical GRAPA is a symlink.
Fix: use CHECKPOINT = GRAPA / repo-relative checkpoint path in our census
harness; parent metadata reader sees the same underlying file. No canonical
source edits. Add source-only, hash-chained immutable amendments to kernel
registration verification; receipts bind both base and current chain hashes.
Protocol keys, gates and predictions cannot be overridden by amendments.

Prior art: SHA-256 (NIST 2001), hash chains (Haber and Stornetta 1991), taken;
ours: source-only preparation amendment schema. Unverified — lead to check:
"Haber Stornetta 1991 hash chain". No native correctness claim.
