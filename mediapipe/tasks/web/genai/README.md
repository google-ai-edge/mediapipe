# MediaPipe Tasks GenAI Package

This package contains the GenAI tasks for MediaPipe.

## LLM Inference

The MediaPipe LLM Inference task generates text responses from input text. For
Gemma 3n models, it can process input images and audio as well.

In order to begin, you must have a model available. You can download [Gemma 3n
E4B](https://huggingface.co/google/gemma-3n-E4B-it-litert-lm/blob/main/gemma-3n-E4B-it-int4-Web.litertlm) or [Gemma 3n E2B](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/blob/main/gemma-3n-E2B-it-int4-Web.litertlm), or
browse for more pre-converted models on our [LiteRT HuggingFace community](https://huggingface.co/litert-community/models), where files named "-web.task" are
specially converted to run optimally in the browser. All text-only variants of
Gemma 3 are available there, as well as [MedGemma-27B-Text](https://huggingface.co/litert-community/MedGemma-27B-IT/blob/main/medgemma-27b-it-int8-web.task). See
our web inference [guide](https://developers.google.com/mediapipe/solutions/genai/llm_inference/web_js) for more information.
Note that only models encoded for the GPU backend are currently supported.
```
const genai = await FilesetResolver.forGenAiTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm"
);
const llmInference = await LlmInference.createFromModelPath(genai, MODEL_URL);
const response = await llmInference.generateResponse(inputText);
```
