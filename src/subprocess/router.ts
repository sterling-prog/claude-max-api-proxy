/**
 * SessionPoolRouter — Session-aware Claude CLI process pool
 *
 * Locks warm CLI processes to OpenClaw session keys, preventing cross-agent
 * context contamination and eliminating per-request spawn overhead.
 */

import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import type { ClaudeModel } from "../adapter/openai-to-cli.js";
import { ClaudeSubprocess } from "./manager.js";

// ─── Constants ────────────────────────────────────────────────────────────────

const OPENCLAW_TOOL_MAPPING_PROMPT = [
  "## Tool Name Mapping",
  "You are running inside Claude Code CLI, not OpenClaw. The system prompt may reference OpenClaw tool names — map them to your actual tools:",
  "",
  "### Direct tool replacements",
  "- `exec` or `process` → use `Bash` (run shell commands)",
  "- `read` → use `Read` (read file contents)",
  "- `write` → use `Write` (write files)",
  "- `edit` → use `Edit` (edit files)",
  "- `grep` → use `Grep` (search file contents)",
  "- `find` or `ls` → use `Glob` or `Bash(ls ...)`",
  "- `web_search` → use `WebSearch`",
  "- `web_fetch` → use `WebFetch`",
  "- `image` → use `Read` (Claude Code can read images)",
  "",
  "### OpenClaw CLI tools (use via Bash)",
  "These OpenClaw tools are available through the `openclaw` CLI. Use `Bash` to run them:",
  '- `memory_search` → `Bash(openclaw memory search "<query>")` — semantic search across memory files',
  "- `memory_get` → `Read` on the memory file directly, OR `Bash(openclaw memory search \"<query>\")` for discovery",
  '- `message` → `Bash(openclaw message send --to <target> "<text>")` — send messages to channels (Telegram, Discord, etc.)',
  "  - Also: `openclaw message read`, `openclaw message broadcast`, `openclaw message react`, `openclaw message poll`",
  "- `cron` → `Bash(openclaw cron list)`, `Bash(openclaw cron add ...)`, `Bash(openclaw cron status)` — manage scheduled jobs",
  "  - Also: `openclaw cron rm`, `openclaw cron enable`, `openclaw cron disable`, `openclaw cron runs`, `openclaw cron run`, `openclaw cron edit`",
  '- `sessions_list` → `Bash(openclaw agent --local --message "list sessions")` or check session files directly',
  '- `sessions_history` → `Bash(openclaw agent --local --message "show history for session <key>")` or check session files',
  "- `nodes` → `Bash(openclaw nodes status)`, `Bash(openclaw nodes describe <node>)`, `Bash(openclaw nodes invoke --node <id> --command <cmd>)`",
  '  - Also: `openclaw nodes run --node <id> "<shell command>"` for running commands on paired nodes',
  "",
  "### Not available via CLI",
  "- `browser` — requires OpenClaw's dedicated browser server (no CLI equivalent)",
  "- `canvas` — requires paired node with canvas capability; use `openclaw nodes invoke` if a node is available",
  "",
  "### Skills",
  "When a skill says to run a bash/python command, use the `Bash` tool directly.",
  "Skills are located in the `skills/` directory relative to your working directory.",
  "To use a skill: `Read` its SKILL.md file first, then follow the instructions using `Bash`.",
  "Run `openclaw skills list --eligible --json` to see all available skills.",
].join("\n");

/** Models that have dedicated warm pools */
const POOLED_MODELS = new Set<ClaudeModel>(["opus", "sonnet"]);

// ─── Types ────────────────────────────────────────────────────────────────────

export interface RouterConfig {
  opusSize: number;
  sonnetSize: number;
  maxRequestsPerProcess: number;
  maxTotalProcesses: number;
  sweepIdleThresholdMs: number;
  requestQueueDepth: number;
  requestTimeoutMs: number;
}

interface PendingRequest {
  prompt: string;
  model: ClaudeModel;
  emitter: EventEmitter;
  timeoutHandle: NodeJS.Timeout;
}

/** Sentinel placed synchronously in lockedSessions while a real process is being claimed/spawned */
interface PendingSentinel {
  isPending: true;
  requestQueue: PendingRequest[];
}

interface PooledProcess {
  pid: number;
  process: ChildProcess;
  model: ClaudeModel;
  lockedTo: string | null;
  agentChannel: string | null;
  lastRequestAt: number;
  spawnedAt: number;
  requestCount: number;
  state: "idle" | "busy" | "recycling";
  requestQueue: PendingRequest[];
  buffer: string;
  currentEmitter: EventEmitter | null;
  timeoutHandle: NodeJS.Timeout | null;
}

type LockEntry = PooledProcess | PendingSentinel;

function isPending(entry: LockEntry): entry is PendingSentinel {
  return (entry as PendingSentinel).isPending === true;
}

// ─── Stats ────────────────────────────────────────────────────────────────────

interface RouterStats {
  total: number;
  locked: { total: number; opus: number; sonnet: number };
  warm: { opus: number; sonnet: number };
  busy: number;
  queued: number;
  orphansReclaimed: number;
  totalRequests: number;
  processRecycles: number;
  routeHits: { locked: number; warm: number; cold: number; fallback: number };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Extract agentChannel from a session key.
 * "agent:scope:discord:channel:123" → "scope:discord:channel:123"
 * Falls back to the full key if format is unexpected.
 */
function extractAgentChannel(sessionKey: string): string {
  const match = sessionKey.match(/^agent:(.+)$/);
  return match ? match[1] : sessionKey;
}

function rejectPending(req: PendingRequest, status: number, retryAfter: number): void {
  clearTimeout(req.timeoutHandle);
  req.emitter.emit("pool_error", { status, retryAfter, message: `HTTP ${status}` });
  req.emitter.emit("error", Object.assign(new Error(`HTTP ${status}`), { poolStatus: status, retryAfter }));
}

// ─── Auth error detection ─────────────────────────────────────────────────────

const AUTH_ERROR_PATTERNS = /auth|unauthorized|token expired|invalid_token/i;

function isAuthError(text: string): boolean {
  return AUTH_ERROR_PATTERNS.test(text);
}

// ─── SessionPoolRouter ────────────────────────────────────────────────────────

export class SessionPoolRouter {
  private config: RouterConfig;
  private lockedSessions: Map<string, LockEntry> = new Map();
  private warmPool: { opus: PooledProcess[]; sonnet: PooledProcess[] } = { opus: [], sonnet: [] };
  private shuttingDown = false;

  // Stats counters
  private orphansReclaimed = 0;
  private processRecycles = 0;
  private routeHits = { locked: 0, warm: 0, cold: 0, fallback: 0 };

  constructor(config: RouterConfig) {
    this.config = config;
  }

  /** Initialize warm pool on startup */
  async initialize(): Promise<void> {
    console.log("[Router] Initializing session pool...");
    const spawns: Promise<void>[] = [];
    for (let i = 0; i < this.config.opusSize; i++) {
      spawns.push(this.spawnWarm("opus"));
    }
    for (let i = 0; i < this.config.sonnetSize; i++) {
      spawns.push(this.spawnWarm("sonnet"));
    }
    await Promise.all(spawns);
    const s = this.stats();
    console.log(`[Router] Ready. warm.opus=${s.warm.opus} warm.sonnet=${s.warm.sonnet}`);
  }

  /**
   * Execute a prompt for a given model and session key.
   * Returns an EventEmitter that emits the same events as ClaudeSubprocess.
   */
  execute(prompt: string, model: ClaudeModel, sessionKey: string | null): EventEmitter {
    // No session key or non-pooled model → fallback to ClaudeSubprocess
    if (!sessionKey || !POOLED_MODELS.has(model)) {
      if (!POOLED_MODELS.has(model)) {
        console.log(`[Router] Non-pooled model "${model}" — falling back to ClaudeSubprocess`);
      }
      this.routeHits.fallback++;
      return this.fallbackSubprocess(prompt, model);
    }

    // Pool saturated check
    if (this.totalProcessCount() >= this.config.maxTotalProcesses) {
      console.warn(`[Router] Pool saturated (total=${this.totalProcessCount()} >= max=${this.config.maxTotalProcesses}) — falling back for sessionKey=${sessionKey}`);
      this.routeHits.fallback++;
      return this.fallbackSubprocess(prompt, model);
    }

    const emitter = new EventEmitter();
    this.routeRequest(emitter, prompt, model, sessionKey);
    return emitter;
  }

  private routeRequest(
    emitter: EventEmitter,
    prompt: string,
    model: ClaudeModel,
    sessionKey: string,
  ): void {
    const existing = this.lockedSessions.get(sessionKey);

    if (existing !== undefined) {
      if (isPending(existing)) {
        // Sentinel in place — queue behind it
        this.enqueueOnSentinel(existing, prompt, model, emitter);
        return;
      }

      const proc = existing;

      // Orphan reclamation: check if a different session for the same agent+channel exists
      this.reclaimOrphan(sessionKey, model);

      // Route to locked process
      if (proc.state === "idle") {
        this.routeHits.locked++;
        this.assignToProcess(proc, prompt, emitter);
        return;
      }

      // Busy or recycling — enqueue on per-process queue
      if (proc.requestQueue.length >= this.config.requestQueueDepth) {
        // 429 backpressure
        setImmediate(() => {
          emitter.emit("pool_error", { status: 429, retryAfter: 5 });
          emitter.emit("error", Object.assign(new Error("HTTP 429 Too Many Requests"), { poolStatus: 429, retryAfter: 5 }));
        });
        return;
      }

      const pending: PendingRequest = {
        prompt,
        model,
        emitter,
        timeoutHandle: setTimeout(() => {
          const idx = proc.requestQueue.indexOf(pending);
          if (idx >= 0) proc.requestQueue.splice(idx, 1);
          emitter.emit("error", Object.assign(new Error("Request queue timeout"), { poolStatus: 503, retryAfter: 3 }));
        }, this.config.requestTimeoutMs),
      };
      proc.requestQueue.push(pending);
      this.routeHits.locked++;
      return;
    }

    // New session key — set sentinel synchronously before any async work
    const sentinel: PendingSentinel = { isPending: true, requestQueue: [] };
    this.lockedSessions.set(sessionKey, sentinel);

    this.claimProcess(sessionKey, model, sentinel).then((proc) => {
      if (!proc) return; // sentinel already cleaned up (spawn failed)
      this.routeHits[proc.requestCount === 0 ? "warm" : "cold"]++;
      // Assign the triggering request
      this.assignToProcess(proc, prompt, emitter);
      // Drain any requests that queued against the sentinel
      this.drainSentinelQueue(proc, sentinel);
    }).catch(() => {
      // claimProcess already cleaned up sentinel and rejected queue
    });
  }

  /** Claim a process from warm pool or spawn cold */
  private async claimProcess(
    sessionKey: string,
    model: ClaudeModel,
    sentinel: PendingSentinel,
  ): Promise<PooledProcess | null> {
    try {
      let proc: PooledProcess;
      const pool = this.warmPool[model as "opus" | "sonnet"];

      if (pool.length > 0) {
        proc = pool.pop()!;
        this.routeHits.warm++;
      } else {
        this.routeHits.cold++;
        proc = await this.spawnCold(model);
      }

      proc.lockedTo = sessionKey;
      proc.agentChannel = extractAgentChannel(sessionKey);
      this.lockedSessions.set(sessionKey, proc);
      return proc;
    } catch (err) {
      // Failed spawn — reject all queued requests then clean up sentinel
      for (const req of sentinel.requestQueue) {
        rejectPending(req, 503, 3);
      }
      this.lockedSessions.delete(sessionKey);
      console.error(`[Router] Cold spawn failed for sessionKey=${sessionKey} model=${model}:`, err);
      throw err;
    }
  }

  private enqueueOnSentinel(sentinel: PendingSentinel, prompt: string, model: ClaudeModel, emitter: EventEmitter): void {
    const pending: PendingRequest = {
      prompt,
      model,
      emitter,
      timeoutHandle: setTimeout(() => {
        const idx = sentinel.requestQueue.indexOf(pending);
        if (idx >= 0) sentinel.requestQueue.splice(idx, 1);
        emitter.emit("error", Object.assign(new Error("Request sentinel timeout"), { poolStatus: 503, retryAfter: 3 }));
      }, this.config.requestTimeoutMs),
    };
    sentinel.requestQueue.push(pending);
  }

  private drainSentinelQueue(proc: PooledProcess, sentinel: PendingSentinel): void {
    for (const req of sentinel.requestQueue) {
      clearTimeout(req.timeoutHandle);
      if (proc.state === "idle") {
        this.assignToProcess(proc, req.prompt, req.emitter);
      } else {
        if (proc.requestQueue.length < this.config.requestQueueDepth) {
          const re: PendingRequest = {
            prompt: req.prompt,
            model: req.model,
            emitter: req.emitter,
            timeoutHandle: setTimeout(() => {
              const idx = proc.requestQueue.indexOf(re);
              if (idx >= 0) proc.requestQueue.splice(idx, 1);
              req.emitter.emit("error", Object.assign(new Error("Queue timeout after sentinel drain"), { poolStatus: 503, retryAfter: 3 }));
            }, this.config.requestTimeoutMs),
          };
          proc.requestQueue.push(re);
        } else {
          rejectPending(req, 503, 3);
        }
      }
    }
  }

  /** Orphan reclamation: if a different session key has the same agentChannel, reclaim it */
  private reclaimOrphan(newSessionKey: string, _model: ClaudeModel): void {
    const newChannel = extractAgentChannel(newSessionKey);
    for (const [key, entry] of this.lockedSessions) {
      if (key === newSessionKey || isPending(entry)) continue;
      const proc = entry;
      if (proc.agentChannel === newChannel && proc.lockedTo !== newSessionKey) {
        console.log(`[Router] Orphan reclaimed: old key=${key} new key=${newSessionKey}`);
        this.orphansReclaimed++;

        // Reject all queued requests on the orphaned process
        for (const req of proc.requestQueue) {
          rejectPending(req, 503, 3);
        }
        proc.requestQueue = [];

        if (proc.state === "idle") {
          this.clearSessionLock(key, proc);
          this.returnToWarmPool(proc);
        } else {
          // Mark recycling — will be cleaned after current request completes
          (proc as any)._orphaned = true;
        }
        break;
      }
    }
  }

  /**
   * Canonical session lock clearing — ALL unlock paths must use this.
   * Caller is responsible for what happens to the process afterward.
   */
  private clearSessionLock(sessionKey: string, proc: PooledProcess): void {
    this.lockedSessions.delete(sessionKey);
    proc.lockedTo = null;
    proc.agentChannel = null;
    proc.requestCount = 0;
  }

  private assignToProcess(proc: PooledProcess, prompt: string, emitter: EventEmitter): void {
    proc.state = "busy";
    proc.requestCount++;
    proc.lastRequestAt = Date.now();
    proc.currentEmitter = emitter;

    // Per-request timeout
    proc.timeoutHandle = setTimeout(() => {
      console.error(`[Router:${proc.pid}] Request timeout after ${this.config.requestTimeoutMs}ms — treating as dead`);
      this.handleProcessDeath(proc, new Error(`Request timeout after ${this.config.requestTimeoutMs}ms`));
    }, this.config.requestTimeoutMs);

    const message = JSON.stringify({
      type: "user",
      message: { role: "user", content: prompt },
    });

    proc.process.stdin?.write(message + "\n");
  }

  private handleRequestComplete(proc: PooledProcess): void {
    if (proc.timeoutHandle) {
      clearTimeout(proc.timeoutHandle);
      proc.timeoutHandle = null;
    }

    // Check if orphaned mid-flight
    if ((proc as any)._orphaned) {
      (proc as any)._orphaned = false;
      const key = proc.lockedTo;
      if (key) this.clearSessionLock(key, proc);
      this.returnToWarmPool(proc);
      this.processRecycles++;
      return;
    }

    // Context accumulation threshold
    const overThreshold = proc.requestCount >= this.config.maxRequestsPerProcess;

    if (overThreshold) {
      if (proc.requestQueue.length === 0) {
        // Recycle immediately
        const key = proc.lockedTo;
        if (key) this.clearSessionLock(key, proc);
        this.returnToWarmPool(proc);
        this.processRecycles++;
        return;
      } else {
        // Set recycling state — drain queue, then recycle
        proc.state = "recycling";
        this.drainNextFromQueue(proc);
        return;
      }
    }

    // Normal case: drain queue or go idle
    if (proc.requestQueue.length > 0) {
      this.drainNextFromQueue(proc);
    } else {
      proc.state = "idle";
      proc.currentEmitter = null;
    }
  }

  private drainNextFromQueue(proc: PooledProcess): void {
    const next = proc.requestQueue.shift();
    if (!next) {
      // Queue empty — check if we were recycling
      if (proc.state === "recycling") {
        const key = proc.lockedTo;
        if (key) this.clearSessionLock(key, proc);
        this.returnToWarmPool(proc);
        this.processRecycles++;
      } else {
        proc.state = "idle";
        proc.currentEmitter = null;
      }
      return;
    }
    clearTimeout(next.timeoutHandle);
    proc.state = "busy";
    proc.currentEmitter = next.emitter;
    proc.lastRequestAt = Date.now();
    proc.requestCount++;

    proc.timeoutHandle = setTimeout(() => {
      console.error(`[Router:${proc.pid}] Queued request timeout — treating as dead`);
      this.handleProcessDeath(proc, new Error("Queued request timeout"));
    }, this.config.requestTimeoutMs);

    const message = JSON.stringify({
      type: "user",
      message: { role: "user", content: next.prompt },
    });
    proc.process.stdin?.write(message + "\n");
  }

  private handleProcessDeath(proc: PooledProcess, error?: Error): void {
    if (proc.timeoutHandle) {
      clearTimeout(proc.timeoutHandle);
      proc.timeoutHandle = null;
    }

    const err = error || new Error("Pool process died unexpectedly");

    // Notify active request
    if (proc.currentEmitter) {
      proc.currentEmitter.emit("error", err);
      proc.currentEmitter = null;
    }

    // Reject queued requests
    for (const req of proc.requestQueue) {
      rejectPending(req, 503, 3);
    }
    proc.requestQueue = [];

    // Remove from lockedSessions
    if (proc.lockedTo) {
      this.lockedSessions.delete(proc.lockedTo);
      proc.lockedTo = null;
      proc.agentChannel = null;
      proc.requestCount = 0;
    } else {
      // Might be in warm pool — remove it
      for (const model of ["opus", "sonnet"] as const) {
        const idx = this.warmPool[model].indexOf(proc);
        if (idx >= 0) this.warmPool[model].splice(idx, 1);
      }
    }

    // Spawn replacement into warm pool
    if (!this.shuttingDown) {
      const model = proc.model;
      this.spawnWarm(model).catch((e) => {
        console.error(`[Router] Failed to spawn replacement for dead ${model} process:`, e);
      });
    }
  }

  /** Spawn a warm process and add it to the warm pool */
  private async spawnWarm(model: ClaudeModel): Promise<void> {
    if (this.totalProcessCount() >= this.config.maxTotalProcesses) {
      console.warn(`[Router] Cannot spawn warm ${model}: total=${this.totalProcessCount()} >= max=${this.config.maxTotalProcesses}`);
      return;
    }
    const proc = await this.spawnCold(model);
    this.warmPool[model as "opus" | "sonnet"].push(proc);
  }

  /** Spawn a new CLI process */
  private async spawnCold(model: ClaudeModel): Promise<PooledProcess> {
    return new Promise((resolve, reject) => {
      const args = [
        "--print",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--model", model,
        "--append-system-prompt", OPENCLAW_TOOL_MAPPING_PROMPT,
      ];

      const child = spawn(process.env.CLAUDE_BIN || "claude", args, {
        cwd: process.env.HOME || "/tmp",
        env: Object.fromEntries(
          Object.entries(process.env).filter(([k]) => k !== "CLAUDECODE")
        ),
        stdio: ["pipe", "pipe", "pipe"],
      });

      const proc: PooledProcess = {
        pid: child.pid || 0,
        process: child,
        model,
        lockedTo: null,
        agentChannel: null,
        lastRequestAt: 0,
        spawnedAt: Date.now(),
        requestCount: 0,
        state: "idle",
        requestQueue: [],
        buffer: "",
        currentEmitter: null,
        timeoutHandle: null,
      };

      child.on("error", (err) => {
        reject(err);
      });

      child.on("spawn", () => {
        // Attach stdout handler after successful spawn
        child.stdout?.on("data", (chunk: Buffer) => {
          proc.buffer += chunk.toString();
          this.processBuffer(proc);
        });

        child.stderr?.on("data", (chunk: Buffer) => {
          const text = chunk.toString().trim();
          if (!text) return;
          if (process.env.DEBUG_SUBPROCESS) {
            console.error(`[Router:${proc.pid}] stderr: ${text.slice(0, 200)}`);
          }
          if (isAuthError(text)) {
            console.error(`[Router:${proc.pid}] Auth error detected — triggering death recovery`);
            this.handleProcessDeath(proc, new Error(`Auth error: ${text.slice(0, 100)}`));
          }
        });

        child.on("close", (code) => {
          if (proc.state !== "idle" || proc.currentEmitter) {
            // Died mid-request
            this.handleProcessDeath(proc, new Error(`Process exited unexpectedly with code ${code}`));
          } else {
            // Clean close — remove from wherever it is
            const model = proc.model;
            const idx = this.warmPool[model as "opus" | "sonnet"].indexOf(proc);
            if (idx >= 0) this.warmPool[model as "opus" | "sonnet"].splice(idx, 1);
            if (proc.lockedTo) {
              this.lockedSessions.delete(proc.lockedTo);
            }
            if (!this.shuttingDown) {
              this.spawnWarm(model).catch((e) => console.error("[Router] Respawn failed:", e));
            }
          }
        });

        resolve(proc);
      });

      // If no spawn event (older Node), resolve after a tick
      setTimeout(() => {
        if (child.pid) {
          child.stdout?.on("data", (chunk: Buffer) => {
            proc.buffer += chunk.toString();
            this.processBuffer(proc);
          });
          child.stderr?.on("data", (chunk: Buffer) => {
            const text = chunk.toString().trim();
            if (!text) return;
            if (isAuthError(text)) {
              this.handleProcessDeath(proc, new Error(`Auth error: ${text.slice(0, 100)}`));
            }
          });
          child.on("close", (code) => {
            if (proc.state !== "idle" || proc.currentEmitter) {
              this.handleProcessDeath(proc, new Error(`Process exited with code ${code}`));
            }
          });
          resolve(proc);
        }
      }, 100);
    });
  }

  private returnToWarmPool(proc: PooledProcess): void {
    proc.state = "idle";
    proc.currentEmitter = null;
    proc.requestCount = 0;
    proc.lockedTo = null;
    proc.agentChannel = null;

    const model = proc.model as "opus" | "sonnet";
    const target = model === "opus" ? this.config.opusSize : this.config.sonnetSize;

    if (this.warmPool[model].length < target && this.totalProcessCount() < this.config.maxTotalProcesses) {
      this.warmPool[model].push(proc);
    } else {
      // Over target — kill the extra process
      proc.process.kill("SIGTERM");
    }
  }

  /** Process stdout buffer and emit events to the current request's emitter */
  private processBuffer(proc: PooledProcess): void {
    const lines = proc.buffer.split("\n");
    proc.buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const message = JSON.parse(trimmed);
        const emitter = proc.currentEmitter;
        if (!emitter) continue;

        emitter.emit("message", message);

        if (message.type === "system" && message.subtype === "init") continue;

        if (message.type === "assistant") {
          emitter.emit("assistant", message);
        }

        if (message.type === "content_block_start") {
          const block = message.content_block;
          if (block?.type === "text") {
            emitter.emit("text_block_start", { event: message });
          } else if (block?.type === "tool_use") {
            emitter.emit("tool_use_start", { event: message });
          }
        }

        if (message.type === "content_block_delta") {
          const delta = message.delta;
          if (delta?.type === "text_delta") {
            emitter.emit("content_delta", { event: message });
          } else if (delta?.type === "input_json_delta") {
            emitter.emit("input_json_delta", { event: message });
          }
        }

        if (message.type === "content_block_stop") {
          emitter.emit("content_block_stop", { event: message });
        }

        if (message.type === "result") {
          emitter.emit("result", message);
          this.handleRequestComplete(proc);
        }
      } catch {
        if (process.env.DEBUG_SUBPROCESS) {
          console.error(`[Router:${proc.pid}] Non-JSON: ${trimmed.slice(0, 100)}`);
        }
      }
    }
  }

  /** Use ClaudeSubprocess for non-pooled or fallback requests */
  private fallbackSubprocess(prompt: string, model: ClaudeModel): EventEmitter {
    const sub = new ClaudeSubprocess();
    sub.start(prompt, { model }).catch((err) => {
      sub.emit("error", err);
    });
    return sub;
  }

  private totalProcessCount(): number {
    return this.lockedSessions.size + this.warmPool.opus.length + this.warmPool.sonnet.length;
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  /**
   * Nightly sweep — recycle idle/overused processes, refill warm pool.
   * Called externally by the scheduler (3 AM ET).
   */
  sweep(): void {
    console.log("[Router] Sweep started");
    let recycled = 0;

    for (const [key, entry] of this.lockedSessions) {
      if (isPending(entry)) continue;
      const proc = entry;

      if (proc.state === "busy" || proc.state === "recycling") continue;

      const idleMs = Date.now() - proc.lastRequestAt;
      const overThreshold = proc.requestCount >= this.config.maxRequestsPerProcess;

      if (idleMs > this.config.sweepIdleThresholdMs || overThreshold) {
        this.clearSessionLock(key, proc);
        proc.process.kill("SIGTERM");
        recycled++;
        this.processRecycles++;
      }
    }

    // Refill warm pool — check cap before EACH spawn
    for (const model of ["opus", "sonnet"] as const) {
      const target = model === "opus" ? this.config.opusSize : this.config.sonnetSize;
      while (this.warmPool[model].length < target) {
        if (this.totalProcessCount() >= this.config.maxTotalProcesses) {
          console.warn(`[Router] Sweep refill stopped: total=${this.totalProcessCount()} >= max=${this.config.maxTotalProcesses}`);
          break;
        }
        this.spawnWarm(model).catch((e) => console.error(`[Router] Sweep spawn failed:`, e));
      }
    }

    console.log(`[Router] Sweep complete. recycled=${recycled} warm.opus=${this.warmPool.opus.length} warm.sonnet=${this.warmPool.sonnet.length}`);
  }

  /** Graceful shutdown — close all processes, reject all queued requests */
  async shutdown(): Promise<void> {
    this.shuttingDown = true;
    console.log("[Router] Shutdown started");

    // Reject all queued requests in locked sessions
    for (const [key, entry] of this.lockedSessions) {
      if (isPending(entry)) {
        for (const req of entry.requestQueue) {
          rejectPending(req, 503, 3);
        }
        this.lockedSessions.delete(key);
      } else {
        const proc = entry;
        for (const req of proc.requestQueue) {
          rejectPending(req, 503, 3);
        }
        proc.requestQueue = [];
        if (proc.currentEmitter) {
          proc.currentEmitter.emit("error", new Error("Server shutting down"));
          proc.currentEmitter = null;
        }
        proc.process.kill("SIGTERM");
      }
    }
    this.lockedSessions.clear();

    // Kill warm pool processes
    for (const model of ["opus", "sonnet"] as const) {
      for (const proc of this.warmPool[model]) {
        proc.process.kill("SIGTERM");
      }
      this.warmPool[model] = [];
    }

    console.log("[Router] Shutdown complete");
  }

  /** Stats for health endpoint */
  stats(): RouterStats {
    let lockedOpus = 0;
    let lockedSonnet = 0;
    let busy = 0;
    let queued = 0;
    let totalRequests = 0;

    for (const [, entry] of this.lockedSessions) {
      if (isPending(entry)) {
        queued += entry.requestQueue.length;
        continue;
      }
      const proc = entry;
      if (proc.model === "opus") lockedOpus++;
      else if (proc.model === "sonnet") lockedSonnet++;
      if (proc.state === "busy" || proc.state === "recycling") busy++;
      queued += proc.requestQueue.length;
      totalRequests += proc.requestCount;
    }

    return {
      total: this.totalProcessCount(),
      locked: { total: lockedOpus + lockedSonnet, opus: lockedOpus, sonnet: lockedSonnet },
      warm: { opus: this.warmPool.opus.length, sonnet: this.warmPool.sonnet.length },
      busy,
      queued,
      orphansReclaimed: this.orphansReclaimed,
      totalRequests,
      processRecycles: this.processRecycles,
      routeHits: { ...this.routeHits },
    };
  }
}
