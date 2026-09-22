---
"workers-ai-provider": minor
---

Add evaluation models for the AI SDK's `experimental_evaluate`: `workersai.evaluationModel("typesafe/jev")` answers boolean, choice and score questions with TypeSafe's Jev.

Over the REST API without a gateway, third-party models such as `typesafe/jev` are now posted to `/ai/run` with the model id in the body. The direct API has no `/ai/run/<vendor>/<model>` route for them.
