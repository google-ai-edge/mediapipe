# MediaPipe Tasks GenAI Experimental Package

This package contains experimental GenAI tasks for MediaPipe.

## RAG Pipeline Inference

You can use the RAG Pipeline to augment an LLM Inference Task with existing
knowledge.

```
const genaiFileset = await FilesetResolver.forGenAiTasks();
const genaiExperimentalFileset =
  await FilesetResolver.forGenAiExperimentalTasks();
const llmInference = await LlmInference.createFromModelPath(genaiFileset, ...);
const ragPipeline = await RagPipeline.createWithEmbeddingModel(
  genaiExperimentalFileset,
  llmInference,
  EMBEDDING_MODEL_URL,
);
await ragPipeline.recordBatchedMemory([
  'Paris is the capital of France.',
  'Berlin is the capital of Germany.',
]);
const result = await ragPipeline.generateResponse(
  'What is the capital of France?',
);
console.log(result);
```
