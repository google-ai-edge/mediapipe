# MediaPipe Tasks Vision Package

This package contains the audio tasks for MediaPipe.

## Audio Classification

The MediaPipe Audio Classification task performs classification on audio data.

```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@latest/wasm"
);
const audioClassifier = await AudioClassifier.createFromModelPath(audio,
    "https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_audio_classifier_with_metadata.tflite"
);
const classifications = audioClassifier.classify(audioData);
```

## Audio Embedding

The MediaPipe Audio Embedding task extracts embeddings from audio data.

```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@latest/wasm"
);
const audioEmbedder = await AudioEmbedder.createFromModelPath(audio,
    "model.tflite"
);
const embeddings = audioEmbedder.embed(audioData);
```
