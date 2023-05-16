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
