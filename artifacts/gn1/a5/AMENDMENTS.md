# Execution amendments

No change to the immutable order. A4→A5 path/order/label relocation and CPU transport adaptation are mechanical D1 work; functional tooling changes are D2–D5.

N4 inspection correction: do not parse a deliberately malformed prior receipt as if it were a new completion receipt. Retain its read error in the control record; the target remains a JSON refusal without traceback. Add raw_exit diagnostic to D3 receipt; normalized exit remains authoritative.
