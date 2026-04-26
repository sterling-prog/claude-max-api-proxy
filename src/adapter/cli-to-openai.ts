/**
 * Converts Claude CLI output to OpenAI-compatible response format
 */

import type { ClaudeCliAssistant, ClaudeCliResult } from "../types/claude-cli.js";
import type { OpenAIChatResponse, OpenAIChatChunk, OpenAIToolCall } from "../types/openai.js";

/**
 * Extract text content from Claude CLI assistant message
 */
export function extractTextContent(message: ClaudeCliAssistant): string {
  return message.message.content
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("\n\n");
}

/**
 * Convert Claude CLI assistant message to OpenAI streaming chunk
 */
export function cliToOpenaiChunk(
  message: ClaudeCliAssistant,
  requestId: string,
  isFirst: boolean = false
): OpenAIChatChunk {
  const text = extractTextContent(message);

  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(message.message.model),
    choices: [
      {
        index: 0,
        delta: {
          role: isFirst ? "assistant" : undefined,
          content: text,
        },
        finish_reason: message.message.stop_reason ? "stop" : null,
      },
    ],
  };
}

/**
 * Create a final "done" chunk for streaming
 */
export function createDoneChunk(requestId: string, model: string): OpenAIChatChunk {
  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(model),
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: "stop",
      },
    ],
  };
}

/**
 * Convert Claude CLI result to OpenAI non-streaming response
 *
 * `requestedModel` (when provided) is echoed back as `response.model`,
 * matching OpenAI's contract that the response model reflects the request.
 * When omitted, falls back to picking the highest-output-tokens entry from
 * `modelUsage` — `Object.keys(...)[0]` is unreliable because the CLI's
 * `modelUsage` map can include internal tool-routing models (typically
 * Haiku) alongside the primary assistant model.
 */
export function cliResultToOpenai(
  result: ClaudeCliResult,
  requestId: string,
  toolCalls?: OpenAIToolCall[],
  requestedModel?: string
): OpenAIChatResponse {
  const modelName = requestedModel ?? pickPrimaryModel(result);

  const message: OpenAIChatResponse["choices"][0]["message"] = {
    role: "assistant",
    content: result.result,
  };

  if (toolCalls && toolCalls.length > 0) {
    message.tool_calls = toolCalls;
  }

  return {
    id: `chatcmpl-${requestId}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: normalizeModelName(modelName),
    choices: [
      {
        index: 0,
        message,
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: result.usage?.input_tokens || 0,
      completion_tokens: result.usage?.output_tokens || 0,
      total_tokens:
        (result.usage?.input_tokens || 0) + (result.usage?.output_tokens || 0),
    },
  };
}

/**
 * Pick the model that produced the user-visible response from `modelUsage`.
 * The CLI may invoke internal helpers (typically Haiku) alongside the primary
 * assistant model; selecting by max output_tokens reliably picks the primary.
 */
function pickPrimaryModel(result: ClaudeCliResult): string {
  if (!result.modelUsage) return "claude-sonnet-4";
  let bestKey = "";
  let bestTokens = -1;
  for (const [key, usage] of Object.entries(result.modelUsage)) {
    const tokens = usage?.outputTokens ?? 0;
    if (tokens > bestTokens) {
      bestTokens = tokens;
      bestKey = key;
    }
  }
  return bestKey || "claude-sonnet-4";
}

/**
 * Normalize Claude model names to a consistent format
 * e.g., "claude-sonnet-4-5-20250929" -> "claude-sonnet-4"
 */
function normalizeModelName(model: string | undefined): string {
  if (!model) return "claude-sonnet-4";
  if (model.includes("opus")) return "claude-opus-4";
  if (model.includes("sonnet")) return "claude-sonnet-4";
  if (model.includes("haiku")) return "claude-haiku-4";
  return model;
}
