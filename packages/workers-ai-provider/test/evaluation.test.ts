import { describe, expect, it } from "vitest";
import { createWorkersAI } from "../src/index";

// Jev's output as `env.AI.run("typesafe/jev", …)` returns it, captured live.
const JEV_RUN = {
	state: "Completed",
	result: {
		model: "jev-1.13.0",
		answers: {
			is_urgent: { type: "noul", noul: 0.95 },
			department: {
				type: "choice",
				choice: "billing",
				probabilities: { sales: 0, technical: 0.02, billing: 0.98 },
				confidence: 0.97,
			},
			frustration: {
				type: "score",
				score: 1.04,
				legend: { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
				probabilities: { "0": 0, "1": 0.96, "2": 0.04 },
				confidence: 0.94,
			},
		},
		usage: { input_tokens: 385, output_tokens: 73 },
	},
	gatewayMetadata: { keySource: "Unified" },
};

const QUESTIONS = {
	is_urgent: { type: "boolean", instructions: "Does this convey urgency?" },
	department: {
		type: "choice",
		instructions: "Which team should handle this?",
		criteria: { billing: "Payments", technical: "Bugs", sales: "Pricing" },
	},
	frustration: {
		type: "score",
		instructions: "How frustrated is the customer?",
		criteria: ["Calm", "Frustrated", "Very angry"],
	},
} as const;

describe("Evaluation - Binding", () => {
	it("should evaluate boolean, choice and score questions with Jev", async () => {
		let captured: { model: string; inputs: any; options: any } | undefined;
		const workersai = createWorkersAI({
			binding: {
				run: async (model: string, inputs: any, options: any) => {
					captured = { model, inputs, options };
					return JEV_RUN;
				},
			} as any,
			gateway: { id: "my-gateway" },
		});

		const result = await workersai.evaluationModel("typesafe/jev").doEvaluate({
			state: "Help! My payouts have been failing for 3 days.",
			questions: QUESTIONS,
		});

		expect(result.answers.is_urgent).toEqual({ type: "boolean", probability: 0.95 });
		expect(result.answers.department).toEqual({
			type: "choice",
			choice: "billing",
			probabilities: { sales: 0, technical: 0.02, billing: 0.98 },
		});
		expect(result.answers.frustration).toMatchObject({ type: "score", score: 1.04 });
		expect(result.rounding).toEqual({ probabilityDecimals: 2, scoreDecimals: 2 });
		expect(result.usage?.inputTokens).toBe(385);
		expect(result.providerMetadata).toEqual({
			workersai: { confidence: { department: 0.97, frustration: 0.94 } },
		});

		// Jev calls a boolean question `noul`.
		expect(captured?.model).toBe("typesafe/jev");
		expect(captured?.inputs.questions.is_urgent.type).toBe("noul");
		expect(captured?.options.gateway).toEqual({ id: "my-gateway" });
	});

	it("should reject a run that did not complete", async () => {
		const workersai = createWorkersAI({
			binding: { run: async () => ({ state: "Queued", result: {} }) } as any,
		});
		await expect(
			workersai.evaluationModel("typesafe/jev").doEvaluate({
				state: "s",
				questions: { q: { type: "boolean", instructions: "?" } },
			}),
		).rejects.toThrow("did not complete (state: Queued)");
	});
});

describe("Evaluation - REST API", () => {
	it("should post third-party models to /ai/run with the model in the body", async () => {
		let captured: { url: string; body: any } | undefined;
		const workersai = createWorkersAI({
			accountId: "test-account",
			apiKey: "test-key",
			fetch: (async (url: string, init: RequestInit) => {
				captured = { url, body: JSON.parse(init.body as string) };
				return Response.json({ result: JEV_RUN, success: true, errors: [] });
			}) as typeof fetch,
		});

		const result = await workersai.evaluationModel("typesafe/jev").doEvaluate({
			state: "Help! My payouts have been failing for 3 days.",
			questions: QUESTIONS,
		});

		expect(result.answers.is_urgent).toEqual({ type: "boolean", probability: 0.95 });
		expect(captured?.url).toBe(
			"https://api.cloudflare.com/client/v4/accounts/test-account/ai/run",
		);
		expect(captured?.body.model).toBe("typesafe/jev");
		expect(captured?.body.input.questions.is_urgent.type).toBe("noul");
	});
});
