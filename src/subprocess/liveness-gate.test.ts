/**
 * Liveness Gate tests — P119
 *
 * Structural tests verify the gate API exists and compiles correctly.
 * Integration tests require a real Claude CLI and are marked TODO.
 *
 * Run: npm test
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { SessionPoolRouter } from "./router.js";

// ---------------------------------------------------------------------------
// Structural / API-surface tests (no real CLI required)
// ---------------------------------------------------------------------------

describe("liveness gate — structural", () => {
  it("SessionPoolRouter has __testing_forceStdinClose method", () => {
    const router = new SessionPoolRouter({ opusSize: 0, sonnetSize: 0 });
    assert.strictEqual(typeof router.__testing_forceStdinClose, "function");
  });

  it("stats() returns livenessEvictions counter initialised to 0", () => {
    const router = new SessionPoolRouter({ opusSize: 0, sonnetSize: 0 });
    const s = router.stats();
    assert.ok("livenessEvictions" in s, "livenessEvictions should be in PoolStats");
    assert.strictEqual(s.livenessEvictions, 0);
  });

  it("__testing_forceStdinClose is a no-op for unknown session keys", () => {
    const router = new SessionPoolRouter({ opusSize: 0, sonnetSize: 0 });
    // Should not throw
    router.__testing_forceStdinClose("nonexistent-session-key");
  });
});

// ---------------------------------------------------------------------------
// Integration tests — require real Claude CLI
// ---------------------------------------------------------------------------
// TODO: The full integration flow below requires the Claude CLI binary and
// real credentials.  Run manually in a development environment after verifying
// that `claude` is installed and authenticated.
//
// Outline of the integration test (not run in CI):
//
//   1. Instantiate router with small pool sizes (opusSize: 1, sonnetSize: 1)
//   2. Call router.execute() with a fresh session key to claim a process;
//      record result1.pid
//   3. Wait for the emitter "done" event to confirm the process is idle
//   4. Call router.__testing_forceStdinClose(sessionKey) to simulate dead stdin
//   5. Call router.execute() again with the same session key
//   6. Wait for emitter "done" event; record result2.pid
//   7. Assert:
//      - result2 succeeded (no error event)
//      - result2.pid !== result1.pid   (evicted + fresh process claimed)
//      - router.stats().livenessEvictions === 1
//      - total elapsed < 15_000 ms
