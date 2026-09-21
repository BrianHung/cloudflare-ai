---
"workers-ai-provider": minor
---

Add evaluation models for the AI SDK's `experimental_evaluate`: `workersai.evaluationModel("typesafe/jev")` answers boolean, choice and score questions with TypeSafe's Jev.

The REST API now also reaches third-party models such as `typesafe/jev` without a gateway, by posting to `/ai/run` with the model in the body. The direct API has no `/ai/run/<vendor>/<model>` route.
