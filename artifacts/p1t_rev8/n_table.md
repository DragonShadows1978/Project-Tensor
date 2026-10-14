| Item | Verdict | Evidence |
|---|---|---|
| N1 | CORRECTLY NOT A DEFECT | Actual hooks close before any normal return; an arbitrary replacement hook returning without close is outside that guarantee. C:1640-1642; A:417-421; native-runner control. |
| N2 | CORRECTLY NOT A DEFECT | Explicit unregister/_clear/os._exit deliberately removes documented protection. CI:292-294; SI:264-266; CPython atexit docs. |
| N3 | CORRECTLY NOT A DEFECT | Unreadable diagnostic value selects documented default and never runs conversions. PA:222-279; C:1528-1543; raw-entry census. |
| N4 | CORRECTLY NOT A DEFECT | Preserving KeyboardInterrupt for atexit to print is documented behavior after quiescence. C:1674; A:452; interrupted-hook CPU staging. |
