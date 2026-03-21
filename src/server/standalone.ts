#!/usr/bin/env node
/**
 * Standalone server — session-pooled Claude Max API proxy
 *
 * Usage:
 *   npm run start
 *   node dist/server/standalone.js [port]
 *
 * Environment variables:
 *   POOL_OPUS_SIZE              Warm opus processes (default: 6)
 *   POOL_SONNET_SIZE            Warm sonnet processes (default: 4)
 *   POOL_MAX_REQUESTS_PER_PROCESS  Context accumulation threshold (default: 50)
 *   MAX_TOTAL_PROCESSES         Hard cap on locked + warm processes (default: 30)
 *   SWEEP_HOUR                  Hour in ET for nightly sweep (default: 3)
 *   SWEEP_IDLE_THRESHOLD_MS     Idle time before sweep recycles (default: 7200000)
 *   POOL_REQUEST_QUEUE_DEPTH    Per-process queue depth (default: 3)
 *   POOL_REQUEST_TIMEOUT_MS     Per-request timeout ms (default: 300000)
 */

import cron from "node-cron";
import { startServer } from "./index.js";
import { verifyClaude, verifyAuth } from "../subprocess/manager.js";
import { SessionPoolRouter } from "../subprocess/router.js";
import { setPoolRouter } from "./routes.js";
import type { RouterConfig } from "../subprocess/router.js";

const DEFAULT_PORT = 3456;

function parseEnvInt(name: string, defaultVal: number): number {
  const v = parseInt(process.env[name] || "", 10);
  return isNaN(v) ? defaultVal : v;
}

async function main(): Promise<void> {
  console.log("Claude Code CLI Provider — Session-Pooled Server");
  console.log("=================================================\n");

  const port = parseInt(process.argv[2] || String(DEFAULT_PORT), 10);
  if (isNaN(port) || port < 1 || port > 65535) {
    console.error(`Invalid port: ${process.argv[2]}`);
    process.exit(1);
  }

  // Verify Claude CLI
  console.log("Checking Claude CLI...");
  const cliCheck = await verifyClaude();
  if (!cliCheck.ok) {
    console.error(`Error: ${cliCheck.error}`);
    process.exit(1);
  }
  console.log(`  Claude CLI: ${cliCheck.version || "OK"}`);

  // Auth check (warn, don't exit — M1 invariant: server starts even if auth fails at startup)
  console.log("Checking authentication...");
  const authCheck = await verifyAuth();
  if (!authCheck.ok) {
    console.warn(`  Warning: ${authCheck.error}`);
    console.warn("  Run: claude auth login");
    console.warn("  Server will start but requests will return 401 until authenticated.\n");
  } else {
    console.log("  Authentication: OK\n");
  }

  // Pool configuration
  const routerConfig: RouterConfig = {
    opusSize: parseEnvInt("POOL_OPUS_SIZE", 6),
    sonnetSize: parseEnvInt("POOL_SONNET_SIZE", 4),
    maxRequestsPerProcess: parseEnvInt("POOL_MAX_REQUESTS_PER_PROCESS", 50),
    maxTotalProcesses: parseEnvInt("MAX_TOTAL_PROCESSES", 30),
    sweepIdleThresholdMs: parseEnvInt("SWEEP_IDLE_THRESHOLD_MS", 7_200_000),
    requestQueueDepth: parseEnvInt("POOL_REQUEST_QUEUE_DEPTH", 3),
    requestTimeoutMs: parseEnvInt("POOL_REQUEST_TIMEOUT_MS", 300_000),
  };

  console.log("Pool configuration:");
  console.log(`  POOL_OPUS_SIZE=${routerConfig.opusSize} POOL_SONNET_SIZE=${routerConfig.sonnetSize}`);
  console.log(`  MAX_TOTAL_PROCESSES=${routerConfig.maxTotalProcesses} POOL_MAX_REQUESTS_PER_PROCESS=${routerConfig.maxRequestsPerProcess}`);
  console.log(`  POOL_REQUEST_TIMEOUT_MS=${routerConfig.requestTimeoutMs} POOL_REQUEST_QUEUE_DEPTH=${routerConfig.requestQueueDepth}`);
  console.log(`  SWEEP_IDLE_THRESHOLD_MS=${routerConfig.sweepIdleThresholdMs}\n`);

  // Initialize session pool router
  const router = new SessionPoolRouter(routerConfig);
  setPoolRouter(router);
  await router.initialize();

  const initialStats = router.stats();
  console.log(`Pool initialized: warm.opus=${initialStats.warm.opus} warm.sonnet=${initialStats.warm.sonnet} total=${initialStats.total}\n`);

  // Schedule nightly sweep at 3 AM ET (DST-aware via node-cron timezone)
  const sweepHour = parseEnvInt("SWEEP_HOUR", 3);
  cron.schedule(`0 ${sweepHour} * * *`, () => {
    console.log(`[Cron] Running nightly sweep at ${sweepHour}:00 ET`);
    router.sweep();
  }, { timezone: "America/New_York" });
  console.log(`Nightly sweep scheduled at ${sweepHour}:00 AM ET (DST-aware)\n`);

  // Start HTTP server
  let httpServer: { close: (cb?: () => void) => void } | null = null;
  try {
    const result = await startServer({ port });
    httpServer = (result as any)?.server || null;
    console.log("\nServer ready. Test with:");
    console.log(`  curl -X POST http://localhost:${port}/v1/chat/completions \\`);
    console.log(`    -H "Content-Type: application/json" \\`);
    console.log(`    -H "x-openclaw-session-key: agent:test:discord:channel:123" \\`);
    console.log(`    -d '{"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}]}'`);
    console.log("\nPress Ctrl+C to stop.\n");
  } catch (err) {
    console.error("Failed to start server:", err);
    await router.shutdown();
    process.exit(1);
  }

  // Graceful shutdown — per spec Finding N18:
  // 1. Close listening socket immediately (no new connections)
  // 2. Wait 30s for in-flight requests
  // 3. Call router.shutdown()
  // 4. Exit
  const shutdown = async (signal: string) => {
    console.log(`\nReceived ${signal}. Shutting down...`);

    // Step 1: Close listening socket immediately
    if (httpServer) {
      httpServer.close(() => {
        console.log("  HTTP server closed (no new connections accepted)");
      });
    }

    // Step 2: Wait up to 30s for in-flight requests
    await new Promise<void>((resolve) => setTimeout(resolve, 30_000));

    // Step 3: Shut down pool — rejects all queued requests, kills processes
    await router.shutdown();

    // Step 4: Exit
    process.exit(0);
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}

main().catch((err) => {
  console.error("Unexpected error:", err);
  process.exit(1);
});
