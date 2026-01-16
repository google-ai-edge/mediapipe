# MediaPipe Tasks Text Package

This package contains the text tasks for MediaPipe.

## Language Detector

The MediaPipe Language Detector task predicts the language of an input text.

```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text/wasm"
);
const languageDetector = await LanguageDetector.createFromModelPath(text,
    "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite
);
const result = languageDetector.detect(textData);
```

For more information, refer to the [Language Detector](https://developers.google.com/mediapipe/solutions/text/language_detector/web_js) documentation.

## Text Classifier

The MediaPipe Text Classifier task lets you classify text into a set of defined
categories, such as positive or negative sentiment.

```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text/wasm"
);
const textClassifier = await TextClassifier.createFromModelPath(text,
    "https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/1/bert_classifier.tflite"
);
const classifications = textClassifier.classify(textData);
```

For more information, refer to the [Text Classification](https://developers.google.com/mediapipe/solutions/text/text_classifier/web_js) documentation.

## Text Embedder

The MediaPipe Text Embedder task extracts embeddings from text data.

```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text/wasm"
);
const textEmbedder = await TextEmbedder.createFromModelPath(text,
    "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite"
);
const embeddings = textEmbedder.embed(textData);
```

For more information, refer to the [Text Embedder](https://developers.google.com/mediapipe/solutions/text/text_embedder/web_js) documentation.
