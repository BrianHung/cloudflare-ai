import { createOpenAI } from "@ai-sdk/openai";
import { createUnified } from "../src/providers/unified";
import { generateObject, generateText, jsonSchema, stepCountIs, tool } from "ai";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { createAiGateway } from "../src";

const TEST_ACCOUNT_ID = "test-account-id";
const TEST_API_KEY = "test-api-key";
const TEST_GATEWAY = "my-gateway";
const GATEWAY_URL = `https://gateway.ai.cloudflare.com/v1/${TEST_ACCOUNT_ID}/${TEST_GATEWAY}`;

function responsesEnvelope(output: unknown[]) {
	return {
		id: "resp-test",
		created_at: Math.floor(Date.now() / 1000),
		model: "gpt-4o-mini",
		output,
		incomplete_details: null,
		object: "response",
		usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
	};
}

let captured: { body: unknown } = { body: null };
const server = setupServer();

function gateway() {
	const aigateway = createAiGateway({
		accountId: TEST_ACCOUNT_ID,
		apiKey: TEST_API_KEY,
		gateway: TEST_GATEWAY,
	});
	const openai = createOpenAI({ apiKey: TEST_API_KEY });
	return { aigateway, openai };
}

describe("Tool calling through the gateway", () => {
	beforeAll(() => server.listen());
	afterEach(() => {
		server.resetHandlers();
		captured = { body: null };
	});
	afterAll(() => server.close());

	it("forwards the tool definitions and parses a returned tool call", async () => {
		server.use(
			http.post(GATEWAY_URL, async ({ request }) => {
				captured.body = await request.json();
				return HttpResponse.json(
					responsesEnvelope([
						{
							type: "function_call",
							call_id: "call_1",
							name: "get_weather",
							arguments: JSON.stringify({ location: "San Francisco" }),
							id: "fc_1",
						},
					]),
				);
			}),
		);

		const { aigateway, openai } = gateway();
		const result = await generateText({
			model: aigateway([openai("gpt-4o-mini")]),
			prompt: "What's the weather in San Francisco?",
			tools: {
				get_weather: tool({
					description: "Get the weather for a location",
					inputSchema: jsonSchema<{ location: string }>({
						type: "object",
						properties: { location: { type: "string" } },
						required: ["location"],
						additionalProperties: false,
					}),
				}),
			},
			maxRetries: 0,
		});

		// agp forwarded the tool catalog through to the gateway entry.
		const entry = (captured.body as Array<{ query: { tools?: unknown[] } }>)[0];
		expect(Array.isArray(entry?.query.tools)).toBe(true);
		expect(JSON.stringify(entry?.query.tools)).toContain("get_weather");

		// …and the tool call round-tripped back through the selected model.
		expect(result.toolCalls).toHaveLength(1);
		expect(result.toolCalls[0]?.toolName).toBe("get_weather");
		expect(result.toolCalls[0]?.input).toMatchObject({ location: "San Francisco" });
	});

	it("preserves Gemini thought signatures across unified tool-call turns", async () => {
		const signature = "SIG-TEST-ONLY";
		const capturedRequests: unknown[] = [];

		server.use(
			http.post(GATEWAY_URL, async ({ request }) => {
				capturedRequests.push(await request.json());

				if (capturedRequests.length === 1) {
					return HttpResponse.json({
						id: "chatcmpl-1",
						object: "chat.completion",
						created: 1,
						model: "gemini-3-pro",
						choices: [
							{
								index: 0,
								message: {
									role: "assistant",
									content: null,
									tool_calls: [
										{
											id: "call-1",
											type: "function",
											function: {
												name: "get_weather",
												arguments: JSON.stringify({ location: "Tokyo" }),
											},
											extra_content: {
												google: { thought_signature: signature },
											},
										},
									],
								},
								finish_reason: "tool_calls",
							},
						],
						usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
					});
				}

				return HttpResponse.json({
					id: "chatcmpl-2",
					object: "chat.completion",
					created: 1,
					model: "gemini-3-pro",
					choices: [
						{
							index: 0,
							message: { role: "assistant", content: "done" },
							finish_reason: "stop",
						},
					],
					usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
				});
			}),
		);

		const { aigateway } = gateway();
		const result = await generateText({
			model: aigateway([createUnified()("gemini-3-pro")]),
			prompt: "Get the weather in Tokyo.",
			tools: {
				get_weather: tool({
					description: "Get the weather for a location",
					inputSchema: jsonSchema<{ location: string }>({
						type: "object",
						properties: { location: { type: "string" } },
						required: ["location"],
						additionalProperties: false,
					}),
					execute: async ({ location }) => ({ location, temperature: 21 }),
				}),
			},
			stopWhen: stepCountIs(2),
			maxRetries: 0,
		});

		expect(result.text).toBe("done");
		expect(capturedRequests).toHaveLength(2);

		const secondRequest = (
			capturedRequests[1] as Array<{
				provider: string;
				endpoint: string;
				query: {
					messages?: Array<{
						role?: string;
						tool_calls?: Array<{
							extra_content?: {
								google?: { thought_signature?: string };
							};
						}>;
					}>;
				};
			}>
		)[0];
		expect(secondRequest?.provider).toBe("compat");
		expect(secondRequest?.endpoint).toBe("chat/completions");
		const assistantToolCall = secondRequest?.query.messages?.find(
			(message) => message.role === "assistant" && message.tool_calls,
		);

		expect(assistantToolCall?.tool_calls?.[0]?.extra_content?.google?.thought_signature).toBe(
			signature,
		);
	});
});

describe("Structured output through the gateway", () => {
	beforeAll(() => server.listen());
	afterEach(() => {
		server.resetHandlers();
		captured = { body: null };
	});
	afterAll(() => server.close());

	it("forwards the json schema and parses the object", async () => {
		server.use(
			http.post(GATEWAY_URL, async ({ request }) => {
				captured.body = await request.json();
				return HttpResponse.json(
					responsesEnvelope([
						{
							type: "message",
							role: "assistant",
							id: "msg-1",
							content: [
								{
									type: "output_text",
									text: JSON.stringify({
										capital: "Paris",
										population_millions: 2,
									}),
									annotations: [],
								},
							],
						},
					]),
				);
			}),
		);

		const { aigateway, openai } = gateway();
		const result = await generateObject({
			model: aigateway([openai("gpt-4o-mini")]),
			prompt: "Tell me about France's capital.",
			schema: jsonSchema<{ capital: string; population_millions: number }>({
				type: "object",
				properties: {
					capital: { type: "string" },
					population_millions: { type: "number" },
				},
				required: ["capital", "population_millions"],
				additionalProperties: false,
			}),
			maxRetries: 0,
		});

		expect(result.object).toEqual({ capital: "Paris", population_millions: 2 });
		// The schema reached the gateway entry (Responses API `text.format`).
		expect(JSON.stringify((captured.body as unknown[])[0])).toContain("json_schema");
	});
});
