# MediaPipe Tasks Text Package

This package contains the text tasks for MediaPipe.

## Language Detection

The MediaPipe Language Detector task predicts the language of an input text.

```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@latest/wasm"
);
const languageDetector = await LanguageDetector.createFromModelPath(text,
    "model.tflite"
);
const result = languageDetector.detect(textData);
```

## Text Classification

The MediaPipe Text Classifier task lets you classify text into a set of defined
categories, such as positive or negative sentiment.

```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@latest/wasm"
);
const textClassifier = await TextClassifier.createFromModelPath(text,
    "https://storage.googleapis.com/mediapipe-tasks/text_classifier/bert_text_classifier.tflite"
);
const classifications = textClassifier.classify(textData);
```

For more information, refer to the [Text Classification](https://developers.google.com/mediapipe/solutions/text/text_classifier/web_js) documentation.

## Text Embedding

The MediaPipe Text Embedding task extracts embeddings from text data.

```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@latest/wasm"
);
const textEmbedder = await TextEmbedder.createFromModelPath(text,
    "https://storage.googleapis.com/mediapipe-tasks/text_embedder/mobilebert_embedding_with_metadata.tflite"
);
const embeddings = textEmbedder.embed(textData);
```
