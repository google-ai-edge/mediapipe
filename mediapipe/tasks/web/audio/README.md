# MediaPipe Tasks Audio Package

This package contains the audio tasks for MediaPipe.

## Audio Classifier

The MediaPipe Audio Classifier task performs classification on audio data.

```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm"
);
const audioClassifier = await AudioClassifier.createFromModelPath(audio,
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
);
const classifications = audioClassifier.classify(audioData);
```

For more information, refer to the [Audio Classifier](https://developers.google.com/mediapipe/solutions/audio/audio_classifier/web_js) documentation.

## Audio Transcriber

The MediaPipe Audio Transcriber task transcribes speech in audio data into
text, either as a complete audio clip or as a continuous stream.

```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm"
);
const audioTranscriber = await AudioTranscriber.createFromModelPath(audio,
    "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm"
);
const result = await audioTranscriber.transcribe(audioSamples);
console.log(result.text);
```

For more information, refer to the [Audio Transcriber](https://developers.google.com/mediapipe/solutions/audio/audio_transcriber) documentation.

## Text-to-Speech

The MediaPipe Text-to-Speech task synthesizes natural-sounding speech audio
from text.

```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm"
);
const textToSpeech = await TextToSpeech.createFromModelPath(audio,
    "https://huggingface.co/litert-community/Kokoro-82M/resolve/main/kokoro_int8_gpu_fp16.litertlm"
);
const result = await textToSpeech.synthesize("Hello from MediaPipe!");
// result.samples contains 24 kHz mono audio as a Float32Array.
```

For more information, refer to the [Text-to-Speech](https://developers.google.com/mediapipe/solutions/audio/text_to_speech) documentation.

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
