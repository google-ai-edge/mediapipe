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

### Privacy Notice

Last modified: June 5, 2026

When you use MediaPipe Tasks, processing of the input data (e.g. images, video,
text) takes place on device, and MediaPipe does not send that input data to
Google servers. As a result, you can use our MediaPipe Tasks APIs for
processing data that should not leave the device.

MediaPipe Tasks APIs send metrics about the performance and utilization of the
APIs in your app to Google. Google uses this metrics data to measure
performance, usage, debug, maintain and improve the MediaPipe Tasks, as further
described in our [Privacy Policy](https://policies.google.com/privacy).

**You are responsible for obtaining informed consent from your app users about
Google's processing of MediaPipe metrics data as required by applicable law.**
