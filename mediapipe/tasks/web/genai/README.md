# MediaPipe Tasks GenAI Package

This package contains the GenAI tasks for MediaPipe.

## LLM Inference

The MediaPipe LLM Inference task generates text response from input text.

```
const genai = await FilesetResolver.forGenAiTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm"
);
const llmInference = await LlmInference.createFromModelPath(genai, MODEL_URL);
const response = llmInference.generateResponse(inputText);
```

<!-- TODO: Complete README for MediaPipe GenAI Task. -->
