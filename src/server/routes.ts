/**
 * API Route Handlers
 *
 * Implements OpenAI-compatible endpoints for Clawdbot integration
 */

import type { Request, Response } from "express";
import { EventEmitter } from "events";
import { v4 as uuidv4 } from "uuid";
import { ClaudeSubprocess } from "../subprocess/manager.js";
import { SessionPoolRouter } from "../subprocess/router.js";
import { openaiToCli, extractModel } from "../adapter/openai-to-cli.js";
import {
  cliResultToOpenai,
  createDoneChunk,
} from "../adapter/cli-to-openai.js";
import type { OpenAIChatRequest, OpenAIToolCall } from "../types/openai.js";
import type { ClaudeCliAssistant, ClaudeCliResult, ClaudeCliStreamEvent } from "../types/claude-cli.js";

/** Shared pool router — initialized by standalone.ts on startup */
let poolRouter: SessionPoolRouter | null = null;

export function setPoolRouter(router: SessionPoolRouter): void {
  poolRouter = router;
}

/**
 * Handle POST /v1/chat/completions
 *
 * Main endpoint for chat requests, supports both streaming and non-streaming
 */
export async function handleChatCompletions(
  req: Request,
  res: Response
): Promise<void> {
  const requestId = uuidv4().replace(/-/g, "").slice(0, 24);
  const body = req.body as OpenAIChatRequest;
  const stream = body.stream === true;

  try {
    // Validate request
    if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
      res.status(400).json({
        error: {
          message: "messages is required and must be a non-empty array",
          type: "invalid_request_error",
          code: "invalid_messages",
        },
      });
      return;
    }

    // Convert to CLI input format
    const cliInput = openaiToCli(body);
    const sessionKey = (req.headers["x-openclaw-session-key"] as string | undefined) || null;
    const startMs = Date.now();

    let emitter: EventEmitter;
    let cacheHit: "locked" | "warm" | "cold" | "fallback" | "subprocess" = "subprocess";

    if (poolRouter && sessionKey) {
      // Pool routing
      const modelAlias = extractModel(body.model || "opus");
      emitter = poolRouter.execute(cliInput.prompt, modelAlias, sessionKey);
      cacheHit = "locked"; // actual hit type tracked inside router stats
      console.log(`[Route] pool sessionKey=${sessionKey} model=${modelAlias}`);
    } else {
      // Fallback: subprocess-per-request
      const subprocess = new ClaudeSubprocess();
      subprocess.start(cliInput.prompt, {
        model: cliInput.model,
        sessionId: cliInput.sessionId,
      }).catch((err) => subprocess.emit("error", err));
      emitter = subprocess;
      console.log(`[Route] subprocess (no sessionKey or no pool) model=${cliInput.model}`);
    }

    const pid = (emitter as any).pid ?? "n/a";

    if (stream) {
      await handleStreamingResponse(req, res, emitter, cliInput, requestId, sessionKey, pid, startMs);
    } else {
      await handleNonStreamingResponse(res, emitter, cliInput, requestId);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[handleChatCompletions] Error:", message);

    if (!res.headersSent) {
      res.status(500).json({
        error: {
          message,
          type: "server_error",
          code: null,
        },
      });
    }
  }
}

/**
 * Convert Claude tool_use ID to OpenAI-compatible call ID.
 * Claude uses "toolu_abc123", OpenAI uses "call_abc123".
 */
function toOpenAICallId(claudeId: string): string {
  return `call_${claudeId.replace("toolu_", "")}`;
}

/**
 * Handle streaming response (SSE)
 *
 * IMPORTANT: The Express req.on("close") event fires when the request body
 * is fully received, NOT when the client disconnects. For SSE connections,
 * we use res.on("close") to detect actual client disconnection.
 */
async function handleStreamingResponse(
  req: Request,
  res: Response,
  emitter: EventEmitter,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  sessionKey: string | null = null,
  processPid: string | number = "n/a",
  startMs: number = Date.now()
): Promise<void> {
  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Request-Id", requestId);

  // CRITICAL: Flush headers immediately to establish SSE connection
  // Without this, headers are buffered and client times out waiting
  res.flushHeaders();

  // Send initial comment to confirm connection is alive
  res.write(":ok\n\n");

  return new Promise<void>((resolve, reject) => {
    let isFirst = true;
    let lastModel = "claude-sonnet-4";
    let isComplete = false;
    let hasEmittedText = false;
    let toolCallIndex = 0;
    let inToolBlock = false;
    let clientDisconnected = false;

    // Handle actual client disconnect (response stream closed)
    res.on("close", () => {
      clientDisconnected = true;
      if (!isComplete) {
        if (sessionKey && poolRouter) {
          // Pooled process: detach emitter — let process finish, return to idle
          emitter.removeAllListeners();
        } else if ((emitter as any).kill) {
          // Subprocess fallback: kill it
          (emitter as any).kill();
        }
      }
      resolve();
    });

    // When a new text content block starts after we've already emitted text,
    // insert a separator so text from different blocks doesn't run together
    emitter.on("text_block_start", () => {
      if (hasEmittedText && !res.writableEnded) {
        const sepChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{
            index: 0,
            delta: {
              content: "\n\n",
            },
            finish_reason: null,
          }],
        };
        res.write(`data: ${JSON.stringify(sepChunk)}\n\n`);
      }
    });

    // Handle streaming content deltas
    emitter.on("content_delta", (event: ClaudeCliStreamEvent) => {
      const delta = event.event.delta;
      const text = (delta?.type === "text_delta" && delta.text) || "";
      if (text && !res.writableEnded) {
        const chunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{
            index: 0,
            delta: {
              role: isFirst ? "assistant" : undefined,
              content: text,
            },
            finish_reason: null,
          }],
        };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        isFirst = false;
        hasEmittedText = true;
      }
    });

    // DISABLED: Tool call forwarding causes an agentic loop — OpenClaw interprets
    // Claude Code's internal tool_use (Read, Bash, etc.) as calls it needs to
    // handle, triggering repeated requests. Claude Code handles tools internally
    // via --print mode; only the final text result should be forwarded.
    // TODO: Re-enable with a non-tool_calls display mechanism (e.g. inline text).
    //
    // subprocess.on("tool_use_start", (event: ClaudeCliStreamEvent) => {
    //   if (res.writableEnded) return;
    //   const block = event.event.content_block;
    //   if (block?.type !== "tool_use") return;
    //
    //   inToolBlock = true;
    //   const chunk = {
    //     id: `chatcmpl-${requestId}`,
    //     object: "chat.completion.chunk",
    //     created: Math.floor(Date.now() / 1000),
    //     model: lastModel,
    //     choices: [{
    //       index: 0,
    //       delta: {
    //         role: isFirst ? "assistant" : undefined,
    //         tool_calls: [{
    //           index: toolCallIndex,
    //           id: toOpenAICallId(block.id),
    //           type: "function" as const,
    //           function: {
    //             name: block.name,
    //             arguments: "",
    //           },
    //         }],
    //       },
    //       finish_reason: null,
    //     }],
    //   };
    //   res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    //   isFirst = false;
    // });
    //
    // subprocess.on("input_json_delta", (event: ClaudeCliStreamEvent) => {
    //   if (res.writableEnded) return;
    //   const delta = event.event.delta;
    //   if (delta?.type !== "input_json_delta") return;
    //
    //   const chunk = {
    //     id: `chatcmpl-${requestId}`,
    //     object: "chat.completion.chunk",
    //     created: Math.floor(Date.now() / 1000),
    //     model: lastModel,
    //     choices: [{
    //       index: 0,
    //       delta: {
    //         tool_calls: [{
    //           index: toolCallIndex,
    //           function: {
    //             arguments: delta.partial_json,
    //           },
    //         }],
    //       },
    //       finish_reason: null,
    //     }],
    //   };
    //   res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    // });
    //
    // subprocess.on("content_block_stop", () => {
    //   if (inToolBlock) {
    //     toolCallIndex++;
    //     inToolBlock = false;
    //   }
    // });

    // Handle final assistant message (for model name)
    emitter.on("assistant", (message: ClaudeCliAssistant) => {
      lastModel = message.message?.model || lastModel;
    });

    emitter.on("result", (result: ClaudeCliResult) => {
      isComplete = true;
      const latencyMs = Date.now() - startMs;
      console.log(`[Route] result sessionKey=${sessionKey ?? "none"} model=${cliInput.model} pid=${processPid} latencyMs=${latencyMs}`);
      if (!clientDisconnected && !res.writableEnded) {
        const doneChunk = createDoneChunk(requestId, lastModel);
        if (result.usage) {
          doneChunk.usage = {
            prompt_tokens: result.usage.input_tokens || 0,
            completion_tokens: result.usage.output_tokens || 0,
            total_tokens:
              (result.usage.input_tokens || 0) + (result.usage.output_tokens || 0),
          };
        }
        res.write(`data: ${JSON.stringify(doneChunk)}\n\n`);
        res.write("data: [DONE]\n\n");
        res.end();
      }
      resolve();
    });

    emitter.on("error", (error: Error) => {
      console.error("[Streaming] Error:", error.message);
      if (!clientDisconnected && !res.writableEnded) {
        res.write(
          `data: ${JSON.stringify({
            error: { message: error.message, type: "server_error", code: null },
          })}\n\n`
        );
        res.end();
      }
      resolve();
    });

    emitter.on("close", (code: number | null) => {
      if (!clientDisconnected && !res.writableEnded) {
        if (code !== 0 && !isComplete) {
          res.write(`data: ${JSON.stringify({
            error: { message: `Process exited with code ${code}`, type: "server_error", code: null },
          })}\n\n`);
        }
        res.write("data: [DONE]\n\n");
        res.end();
      }
      resolve();
    });
  });
}

/**
 * Handle non-streaming response
 */
async function handleNonStreamingResponse(
  res: Response,
  emitter: EventEmitter,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string
): Promise<void> {
  return new Promise((resolve) => {
    let finalResult: ClaudeCliResult | null = null;

    emitter.on("result", (result: ClaudeCliResult) => {
      finalResult = result;
    });

    emitter.on("error", (error: Error) => {
      console.error("[NonStreaming] Error:", error.message);
      if (!res.headersSent) {
        res.status(500).json({
          error: {
            message: error.message,
            type: "server_error",
            code: null,
          },
        });
      }
      resolve();
    });

    emitter.on("close", (code: number | null) => {
      if (finalResult) {
        res.json(cliResultToOpenai(finalResult, requestId));
      } else if (!res.headersSent) {
        res.status(500).json({
          error: {
            message: `Claude CLI exited with code ${code} without response`,
            type: "server_error",
            code: null,
          },
        });
      }
      resolve();
    });
  });
}

/**
 * Handle GET /v1/models
 *
 * Returns available models
 */
export function handleModels(_req: Request, res: Response): void {
  const now = Math.floor(Date.now() / 1000);
  const modelIds = [
    "claude-opus-4",
    "claude-opus-4-6",
    "claude-sonnet-4",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-haiku-4",
    "claude-haiku-4-5",
  ];
  res.json({
    object: "list",
    data: modelIds.map((id) => ({
      id,
      object: "model",
      owned_by: "anthropic",
      created: now,
    })),
  });
}

/**
 * Handle GET /health
 *
 * Health check endpoint
 */
export function handleHealth(_req: Request, res: Response): void {
  const response: Record<string, unknown> = {
    status: "ok",
    provider: "claude-code-cli",
    timestamp: new Date().toISOString(),
  };
  if (poolRouter) {
    response.pool = poolRouter.stats();
  }
  res.json(response);
}
