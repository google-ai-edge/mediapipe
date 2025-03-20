# MediaPipe Tasks GenAI Package

This package contains the GenAI tasks for MediaPipe.

## LLM Inference

The MediaPipe LLM Inference task generates text response from input text.

In order to begin, you must have a model available. You can download [Gemma
2B](https://www.kaggle.com/models/google/gemma/frameworks/tfLite/variations/gemma-2b-it-gpu-int4)
(TensorFlow Lite 2b-it-gpu-int4 or 2b-it-gpu-int8) and [Gemma
7B](https://www.kaggle.com/models/google/gemma/tfLite/gemma-1.1-7b-it-gpu-int8)
(TensorFlow Lite 7b-it-gpu-int8) or convert an external LLM
(Phi-2, Falcon, or StableLM) following the
[guide](https://developers.google.com/mediapipe/solutions/genai/llm_inference/web_js#convert-model).
Note that only models encoded for the GPU backend are currently supported.
```
const genai = await FilesetResolver.forGenAiTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm"
);
const llmInference = await LlmInference.createFromModelPath(genai, MODEL_URL);
const response = await llmInference.generateResponse(inputText);
```
