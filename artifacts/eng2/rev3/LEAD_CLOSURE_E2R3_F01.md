# Lead closure of E2R3-F01 (MINOR, consumer census completeness)

**The finding:** A3's file selection missed 16 changed-entry call sites, reached through object fields or namespace
helpers. REV3's independent census of all 2,452 tracked Python files finds 498 explicit-dtype sites, 49 of them
unresolved, against A3's 482 and 48. REV3 confirmed no newly refused site.

**Closure.** No product or test code is involved; the defect is in A3's census tool. Following the precedent for
ENG1-A2 and ENG2-A2, the lead closes it as follows:
- **The census of record** is now REV3's `Q3_consumer_final.json` (498 sites, 49 unresolved, 0 newly refused). It
  supersedes A3's `CONSUMER_CENSUS.json` for the promotion's consumer claim.
- **The one new unresolved site,** GRAPA `grapa/tripwire.py:655`, was read at GRAPA HEAD with `git show`:
  - the call is `self.tc.zeros(tuple(t.shape), dtype=self.dtype(t))`;
  - `dtype(t)` is `return str(t.dtype)` (`:642-643`), and `t` is a `tensor_cuda` tensor from `like`;
  - a tensor's `.dtype` is always one of the six supported names. This is the same reading as A3's Note A, so the
    site is supported.
- The other 15 omitted sites pass supported string literals (REV3).

**Result:** 0 consumer call sites would be newly refused by ENG2. All 49 unresolved sites read as passing a supported
dtype. That is a reading of the code, not a mechanical trace, and it is stated as such in the promotion record.

**Status:** closed. Under the 11-03 rule, this round of fixes gets no further review.
