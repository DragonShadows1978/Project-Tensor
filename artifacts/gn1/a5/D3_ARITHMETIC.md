# D3 kill-grace ordering

Source inspection / reasoning, confirmed by the real installed-runner N2b/N2c receipts. Runner source SHA-256: ad6c0a22edc3baf8fd513f1123e6d460266a7ede77b91c496b3ea38e8724e107.

The staged environment explicitly fixes grace=10 s and poll=5 s. The runner sends TERM, waits the full grace, sends group KILL, and waits for its child. One complete registered poll interval plus grace is 5 + 10 = 15 s, strictly below the parent outer KILL delay of 20 s. The immediate child check sleeps at most0.02 s (line204); even adding it conservatively gives15.02 <20. The script expires at outer+40 s; the parent outer KILL is at outer+20 s, preserving20 s for receipt publication and shell margin. Source scheduling or filesystem stalls are not bounded by this arithmetic; the finite interruption controls establish the observed outcomes.

```text
115:         args.grace = float(os.environ.get("FORGE_GPU_KILL_GRACE", "10"))
123:     if args.margin < args.grace:
124:         raise UsageError("--margin must cover the kill grace")
132:             args.poll = float(os.environ.get("FORGE_GPU_POLL", "5"))
173:     old = {sig: signal.signal(sig, caught) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
183:             if requested or time.monotonic() >= deadline:
184:                 rc = 128 + requested if requested else 124
185:                 outcome = "ran" if requested else "window-closed" if window_closed else "timeout"
195:             if time.monotonic() >= next_poll:
201:                 next_poll = now + poll
204:             time.sleep(min(0.02, max(0, deadline - time.monotonic())))
205:         try:
206:             os.killpg(child.pid, signal.SIGTERM)
209:         grace_end = time.monotonic() + grace
210:         while time.monotonic() < grace_end:
211:             child.poll()
212:             time.sleep(min(0.02, max(0, grace_end - time.monotonic())))
213:         try:
214:             os.killpg(child.pid, signal.SIGKILL)
215:         except ProcessLookupError:
217:         child.wait()
218:         return (128 + requested if requested else rc), outcome
```

N2b_D3 and N2c_D3: exit124, cause outer-timeout, no live child at parent return. The supplemental G4_p116.json records each actual PID from its child start marker immediately after parent return; both are absent. The original obedient-child helper did not record a PID, so the supplement closes that instrumentation gap. Raw_SIGKILL_normalisation deliberately sets grace25 (outside D3), recording raw-9 ->137 and outer-timeout.

Prior art: GNU coreutils timeout(1), POSIX signal exit convention, existing runner TERM/grace/process-group KILL, and REV4 N2b/N2c (2026). No novel supervision algorithm claimed. GNU history unverified — lead to check: coreutils timeout kill-after.
