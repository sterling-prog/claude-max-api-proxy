/**
 * Output watchdog tests
 *
 * Uses fake timers and __testing_armWatchdog — no real CLI required.
 *
 * Run: npm test
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mock } from "node:test";
import { SessionPoolRouter } from "./router.js";

describe("output watchdog", () => {
  before(() => {
    // Short timeout so fake-clock ticks are small
    process.env.POOL_WATCHDOG_TIMEOUT_MS = "500";
  });

  after(() => {
    delete process.env.POOL_WATCHDOG_TIMEOUT_MS;
    mock.timers.reset();
  });

  it("stats() exposes watchdogEvictions counter initialised to 0", () => {
    const router = new SessionPoolRouter({ opusSize: 0, sonnetSize: 0 });
    const s = router.stats();
    assert.ok("watchdogEvictions" in s, "watchdogEvictions must be in PoolStats");
    assert.strictEqual(s.watchdogEvictions, 0);
  });

  it("POOL_WATCHDOG_TIMEOUT_MS=0 disables watchdog — no eviction fires", (_, done) => {
    const saved = process.env.POOL_WATCHDOG_TIMEOUT_MS;
    process.env.POOL_WATCHDOG_TIMEOUT_MS = "0";
    const router = new SessionPoolRouter({ opusSize: 0, sonnetSize: 0, maxTotalProcesses: 0 });
    process.env.POOL_WATCHDOG_TIMEOUT_MS = saved!;

    const emitter = router.__testing_armWatchdog("disabled-session");
    const errors: Error[] = [];
    emitter.on("error", (err: Error) => errors.push(err));

    // Nothing should fire even after several ticks
    setTimeout(() => {
      assert.strictEqual(errors.length, 0, "disabled watchdog must not evict");
      assert.strictEqual(router.stats().watchdogEvictions, 0);
      done();
    }, 100);
  });

  it("busy proc with no stdout is evicted with 503 after timeout", (_, done) => {
    mock.timers.enable({ apis: ["setTimeout"], now: Date.now() });

    // maxTotalProcesses:0 prevents killAndRespawn from spawning a real proc
    const router = new SessionPoolRouter({ opusSize: 0, sonnetSize: 0, maxTotalProcesses: 0 });
    const emitter = router.__testing_armWatchdog("test-session");

    const errors: Array<Error & { statusCode?: number }> = [];
    emitter.on("error", (err: Error & { statusCode?: number }) => errors.push(err));

    mock.timers.tick(600); // advance past 500ms watchdog

    setImmediate(() => {
      assert.strictEqual(errors.length, 1, "exactly one eviction error");
      assert.strictEqual(errors[0].statusCode, 503, "eviction error must be 503");
      assert.ok(router.stats().watchdogEvictions >= 1, "watchdogEvictions must increment");
      mock.timers.reset();
      done();
    });
  });

  it("stdout activity resets the watchdog — no eviction if output arrives before deadline", (_, done) => {
    mock.timers.enable({ apis: ["setTimeout"], now: Date.now() });

    const router = new SessionPoolRouter({ opusSize: 0, sonnetSize: 0, maxTotalProcesses: 0 });
    const emitter = router.__testing_armWatchdog("reset-session");

    const errors: Error[] = [];
    emitter.on("error", (err: Error) => errors.push(err));

    // Advance to 400ms — within the 500ms window, no eviction yet
    mock.timers.tick(400);
    // Simulate stdout chunk: resets the timer to another 500ms from now
    router.__testing_resetWatchdog("reset-session");
    // Advance 400ms more — 800ms total, but only 400ms since the reset; still within window
    mock.timers.tick(400);

    setImmediate(() => {
      assert.strictEqual(errors.length, 0, "no eviction within the reset window");
      assert.strictEqual(router.stats().watchdogEvictions, 0);
      mock.timers.reset();
      done();
    });
  });
});
