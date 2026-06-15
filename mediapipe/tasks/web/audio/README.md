# MediaPipe Tasks Vision Package

This package contains the audio tasks for MediaPipe.

## Audio Classifier

The MediaPipe Audio Classifier task performs classification on audio data.

```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm"
);
const audioClassifier = await AudioClassifier.createFromModelPath(audio,
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
);
const classifications = audioClassifier.classify(audioData);
```

For more information, refer to the [Audio Classifier](https://developers.google.com/mediapipe/solutions/audio/audio_classifier/web_js) documentation.

## Audio Embedding

The MediaPipe Audio Embedding task extracts embeddings from audio data.

```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm"
);
const audioEmbedder = await AudioEmbedder.createFromModelPath(audio,
    "https://storage.googleapis.com/mediapipe-assets/yamnet_embedding_metadata.tflite?generation=1668295071595506"
);
const embeddings = audioEmbedder.embed(audioData);
```

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
