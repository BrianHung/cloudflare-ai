import {
	InvalidResponseDataError,
	type Experimental_EvaluationModelV4 as EvaluationModelV4,
	type Experimental_EvaluationModelV4Answer as EvaluationModelV4Answer,
} from "@ai-sdk/provider";
import { normalizeBindingError } from "./workersai-error";
import type { EvaluationModels } from "./workersai-models";

export type WorkersAIEvaluationConfig = {
	provider: string;
	binding: Ai;
	gateway?: GatewayOptions;
};

type JevAnswer =
	| { type: "noul"; noul: number }
	| {
			type: "choice";
			choice: string;
			probabilities?: Record<string, number>;
			confidence?: number;
	  }
	| {
			type: "score";
			score: number;
			probabilities?: Record<string, number>;
			confidence?: number;
	  };

type JevOutput = {
	model?: string;
	answers: Record<string, JevAnswer>;
	usage?: { input_tokens?: number; output_tokens?: number };
};

/**
 * Workers AI evaluation model implementing the AI SDK's experimental
 * `EvaluationModelV4` interface, for use with `experimental_evaluate`.
 *
 * Supports TypeSafe's Jev (`typesafe/jev`), which answers boolean, choice and
 * score questions about one shared state with calibrated probabilities.
 *
 * Workers AI evaluation API:
 * - Input: `{ state, questions }`, where a boolean question has type `noul`
 * - Output: `{ state: "Completed", result: { model, answers, usage } }`
 */
export class WorkersAIEvaluationModel implements EvaluationModelV4 {
	readonly specificationVersion = "v4";
	readonly supportedQuestionTypes = ["choice", "score", "boolean"] as const;

	get provider(): string {
		return this.config.provider;
	}

	constructor(
		readonly modelId: EvaluationModels,
		readonly config: WorkersAIEvaluationConfig,
	) {}

	async doEvaluate(
		options: Parameters<EvaluationModelV4["doEvaluate"]>[0],
	): Promise<Awaited<ReturnType<EvaluationModelV4["doEvaluate"]>>> {
		const { state, questions, abortSignal } = options;

		// Jev calls a boolean question `noul`.
		const inputs = {
			state,
			questions: Object.fromEntries(
				Object.entries(questions).map(([id, question]) => [
					id,
					question.type === "boolean" ? { ...question, type: "noul" } : question,
				]),
			),
		};

		let result: unknown;
		try {
			result = await this.config.binding.run(
				this.modelId as Parameters<Ai["run"]>[0],
				inputs as Parameters<Ai["run"]>[1],
				{ gateway: this.config.gateway, signal: abortSignal } as AiOptions,
			);
		} catch (error) {
			throw normalizeBindingError(error, {
				model: this.modelId,
				requestBodyValues: inputs,
			});
		}

		const output = unwrapRun(result);
		const confidence = Object.fromEntries(
			Object.entries(output.answers).flatMap(([id, answer]) =>
				answer.type !== "noul" && answer.confidence != null
					? [[id, answer.confidence]]
					: [],
			),
		);

		return {
			answers: Object.fromEntries(
				Object.entries(output.answers).map(([id, answer]) => [id, toAnswer(answer)]),
			),
			usage: {
				inputTokens: output.usage?.input_tokens,
				outputTokens: output.usage?.output_tokens,
			},
			// Jev rounds probabilities and scores to two decimal places.
			rounding: { probabilityDecimals: 2, scoreDecimals: 2 },
			warnings: [],
			providerMetadata: { workersai: { confidence } },
			response: {
				timestamp: new Date(),
				modelId: output.model ?? this.modelId,
				body: result,
			},
		};
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * The binding and the direct REST API return the run,
 * `{ state: "Completed", result: output }`. Through AI Gateway over REST,
 * `createRun` unwraps it once more, leaving the output itself.
 */
function unwrapRun(result: unknown): JevOutput {
	const run = result as { state?: unknown; result?: JevOutput; answers?: unknown } | null;
	if (typeof run?.answers === "object" && run.answers !== null) {
		return run as unknown as JevOutput;
	}
	if (run?.state !== "Completed" || typeof run.result?.answers !== "object") {
		throw new InvalidResponseDataError({
			data: result,
			message: `Workers AI evaluation did not complete (state: ${String(run?.state)}).`,
		});
	}
	return run.result;
}

function toAnswer(answer: JevAnswer): EvaluationModelV4Answer {
	switch (answer.type) {
		case "noul":
			return { type: "boolean", probability: answer.noul };
		case "choice":
			return {
				type: "choice",
				choice: answer.choice,
				...(answer.probabilities && { probabilities: answer.probabilities }),
			};
		case "score":
			return {
				type: "score",
				score: answer.score,
				...(answer.probabilities && { probabilities: answer.probabilities }),
			};
	}
}
