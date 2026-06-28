# YOLO Object Detection Integration Design

**Date:** 2026-04-24  
**Status:** Approved  
**Scope:** Port YOLO detection (v8/9/10/11/12/26) from mediapipe-yolo into the base mediapipe project  
**Branch:** `dev/extend-capabilities`

---

## 1. Goals & Non-Goals

**In scope:**
- Single-image YOLO object detection, C++ only
- TFLite inference backend (ONNX/PyTorch/TensorRT in a future PR)
- Support YOLOv8, v9, v10 (standard + e2e export), v11, v12, v26-e2e
- Enhance `TfLiteInferenceCalculator` to emit `MODEL_METADATA` side packet
- New `AutoNmsCalculator` for zero-cost NMS bypass on end-to-end models
- Python Task API in a future PR
- Batch inference (batch > 1) in a future PR (foundation laid via `MODEL_METADATA`)
- SAHI / tiling in a future PR

---

## 2. YOLO Output Tensor Families

Two fundamentally different output formats require separate decode paths.

### Family A — Ultralytics Detection Head
Models: YOLOv8, YOLOv9, YOLO11, YOLOv12, YOLOv10 (standard export)

| Property | Value |
|---|---|
| Output shape | `[B, 4+C, A]` (FEATURES_FIRST) or `[B, A, 4+C]` (BOXES_FIRST) |
| Example (COCO 80) | `[1, 84, 8400]` or `[1, 8400, 84]` |
| Box format | `cx, cy, w, h` (center + size, pixel coords) |
| Class | argmax over C score logits |
| NMS | Required externally |

### Family B — End-to-End NMS
Models: YOLOv10-e2e, YOLOv26-e2e

| Property | Value |
|---|---|
| Output shape | `[B, N, 6]` where N is fixed (typically 300) |
| Box format | `x1, y1, x2, y2` (corner coords, pixel) |
| Feature layout | `[x1, y1, x2, y2, score, class_id(int)]` |
| NMS | Built into model — skip externally |

**Auto-detection rule** (from output shape, after squeezing leading singleton dims):
- One dimension is exactly `6` → **Family B (END_TO_END)**
- Both dims > 4 and neither is 6, larger dim >> smaller → **Family A (ULTRALYTICS)**
- Ambiguous (e.g. `[6, 6]`) → error, user must set `decode_mode` explicitly

---

## 3. Architecture

New components slot into the existing 4-tier MediaPipe architecture without modifying any existing calculator or graph.

```
Tier 3: YoloObjectDetectorGraph          (new)
          mediapipe/tasks/cc/vision/yolo_object_detector/

Tier 1: YoloTensorsToDetectionsCalculator (new)
        AutoNmsCalculator                 (new)
          mediapipe/calculators/yolo/

Tier 1: TfLiteInferenceCalculator        (enhanced — MODEL_METADATA side packet)
        TfLiteConverterCalculator        (reused, unchanged)
        ImageTransformationCalculator    (reused, unchanged)
        NonMaxSuppressionCalculator      (reused, unchanged)
```

---

## 4. Data Flow

```
                         YoloObjectDetectorOptions
                         (input_width/height pre-read from model
                          flatbuffer at graph build time by
                          yolo_object_detector_graph_utils.cc)
                                    │
IMAGE input stream                  │ [side packet: options]
  │                                 ▼
  ▼                     ┌─────────────────────────┐
ImageTransformationCalculator       │  H×W from options
  resize to model H×W ◄─────────────┘
  │
  ▼
TfLiteConverterCalculator
  image → NHWC float32 tensor
  │
  ▼
TfLiteInferenceCalculator ──────────────────────────► MODEL_METADATA side packet
  model inference (TFLite)                              (emitted during Open(),
  │ TENSORS                                              flows to downstream calcs)
  ▼                                         ┌───────────────────────┘
YoloTensorsToDetectionsCalculator ◄──────────┤ MODEL_METADATA
  Family A: argmax + cx/cy/w/h → corners    │
  Family B: direct x1/y1/x2/y2, threshold  │
  │ DETECTIONS                              │
  ▼                                         │
AutoNmsCalculator ◄────────────────────────── MODEL_METADATA (same side packet)
  Options: decode_mode, postprocess_mode    (also receives options from graph)
  Open(): resolves skip_nms_ (see table below)
  Process(): if skip_nms_ → forward unchanged
             else         → run NMS
  │
  ▼
DETECTIONS output stream
```

**Note on H×W:** `ImageTransformationCalculator` is upstream of `TfLiteInferenceCalculator` and cannot receive `MODEL_METADATA` (a side packet emitted downstream). Instead, `YoloObjectDetectorGraph` uses `yolo_object_detector_graph_utils.cc` to read the model's input shape from the TFLite flatbuffer **at graph build time** (cheap — reads schema only, no tensor allocation). The resolved H×W is injected as a `ConstantSidePacketCalculator` side packet to `ImageTransformationCalculator`. `MODEL_METADATA` from `TfLiteInferenceCalculator` serves only the downstream calculators (`YoloTensorsToDetectionsCalculator`, `AutoNmsCalculator`).

**NMS decision — static per graph instance, resolved in `Open()`:**

| `decode_mode` option | `postprocess_mode` option | AutoNmsCalculator behavior |
|---|---|---|
| `END_TO_END` | any | skip NMS |
| `ULTRALYTICS` | any | run NMS |
| `AUTO` | `SKIP_NMS` | skip NMS |
| `AUTO` | `APPLY_NMS` | run NMS |
| `AUTO` | `AUTO` | read `MODEL_METADATA.outputs[0].shape`: dim==6 → skip; else run |

`skip_nms_` is a bool member set once in `Open()`. `Process()` is a branch-free forward or NMS call — no per-frame shape inspection.

---

## 5. New Files

### 5a. Proto — `TfLiteModelMetadata`

```
mediapipe/calculators/tflite/proto/tflite_model_metadata.proto
```

```proto
message TfLiteTensorSpec {
  optional string name = 1;
  repeated int32 shape = 2;              // e.g. [1, 640, 640, 3]
  optional TfLiteTensorType type = 3;    // FLOAT32 / UINT8 / INT8
  optional float quantization_scale = 4;
  optional int32 quantization_zero_point = 5;
}

message TfLiteModelMetadata {
  repeated TfLiteTensorSpec inputs = 1;
  repeated TfLiteTensorSpec outputs = 2;
}
```

### 5b. `TfLiteInferenceCalculator` enhancement

File: `mediapipe/calculators/tflite/tflite_inference_calculator.cc`

- Add optional output side packet tag `MODEL_METADATA`
- During `Open()`, after model load and `AllocateTensors()`: iterate `interpreter_->input_tensors()` and `output_tensors()`, populate `TfLiteModelMetadata`, emit side packet
- Backward compatible: graphs that do not connect `MODEL_METADATA` are unaffected

### 5c. `YoloTensorsToDetectionsCalculator`

```
mediapipe/calculators/yolo/
├── BUILD
└── yolo_tensors_to_detections_calculator.cc
```

Port from `mediapipe-yolo/mediapipe/calculators/yolo/yolo_tensors_to_detections_calculator.cc`.

Key adaptations:
- Consumes optional `MODEL_METADATA` side packet to cross-check resolved decode_mode
- Handles `float32`, `uint8`, `int8` output tensors
- `decode_mode=AUTO`: resolves in `Open()` via `SqueezeLeadingSingletonDims` + dim==6 check
- Family A decode: `DecodeUltralytics()` — argmax class scores, cx/cy/w/h → corners
- Family B decode: `DecodeEndToEnd()` — direct read of `[x1, y1, x2, y2, score, class_id]`

Proto: `mediapipe/tasks/cc/components/processors/proto/yolo_tensors_to_detections_calculator.proto`  
(port from mediapipe-yolo, namespace unchanged)

### 5d. `AutoNmsCalculator`

```
mediapipe/calculators/yolo/
└── auto_nms_calculator.cc
```

New calculator. Inputs: `DETECTIONS` stream + `MODEL_METADATA` side packet. Options: `AutoNmsCalculatorOptions` (contains `decode_mode`, `postprocess_mode`, NMS params).

```
Open():
  // Priority: explicit options > MODEL_METADATA shape inference
  if options.postprocess_mode == SKIP_NMS
      || options.decode_mode == END_TO_END:
    skip_nms_ = true
  else if options.postprocess_mode == APPLY_NMS
      || options.decode_mode == ULTRALYTICS:
    skip_nms_ = false
  else:  // both AUTO
    if MODEL_METADATA connected:
      skip_nms_ = (outputs[0].shape contains dim == 6)
    else:
      skip_nms_ = false  // conservative default: run NMS, log warning

Process():
  if skip_nms_: forward packet unchanged  // zero NMS cost
  else: apply NonMaxSuppression logic
```

`MODEL_METADATA` not connected + AUTO mode → runs NMS (safe fallback, backward compatible).

### 5e. `YoloObjectDetectorGraph`

```
mediapipe/tasks/cc/vision/yolo_object_detector/
├── BUILD
├── proto/
│   └── yolo_object_detector_options.proto  (port from mediapipe-yolo)
├── yolo_object_detector_graph_utils.cc/h   (port from mediapipe-yolo)
└── yolo_object_detector_graph.cc           (port + adapt from mediapipe-yolo)
```

Graph registers as `YoloObjectDetectorGraph`. Wires all calculators described in Section 4. Reads `YoloObjectDetectorOptions.decode_mode` and `postprocess_mode` at build time to configure `AutoNmsCalculator`.

---

## 6. Error Handling

### `YoloTensorsToDetectionsCalculator`

| Condition | Response |
|---|---|
| Output tensor rank not 2 or 3 after squeezing | `kInvalidArgument` with actual shape in message |
| AUTO mode, shape ambiguous (e.g. `[6, 6]`) | `kInvalidArgument`: "unable to infer decode_mode, set explicitly" |
| `quantization_scale == 0` with quantized tensor | `kInvalidArgument`: "quantization_scale must be non-zero" |
| `class_id` out of label map range | Skip detection silently (model noise) |

### `AutoNmsCalculator`

| Condition | Response |
|---|---|
| `MODEL_METADATA` not connected | Default to running NMS (safe fallback, logged once at `Open()`) |
| Multiple output tensors | Use `outputs[0]` only, ignore rest |
| Empty DETECTIONS packet | Forward empty packet, skip NMS |

### `TfLiteInferenceCalculator` (`MODEL_METADATA` path)

| Condition | Response |
|---|---|
| Model load fails | Existing error path unchanged; `MODEL_METADATA` side packet not emitted |
| Downstream requires `MODEL_METADATA` but not received | Downstream returns `kNotFound` in `Open()` |

---

## 7. Testing

### Unit Tests

| File | Coverage |
|---|---|
| `yolo_tensors_to_detections_calculator_test.cc` | Family A: argmax + center→corner decode; Family B: direct x1/y1/x2/y2 read; AUTO shape inference; INT8 quantized input; ambiguous shape → `kInvalidArgument` |
| `auto_nms_calculator_test.cc` | dim==6 metadata → pass-through; non-6 metadata → NMS runs; no metadata → NMS runs |
| `tflite_inference_calculator_metadata_test.cc` | `MODEL_METADATA` shape/dtype/quantization correctness after `Open()` |

### Integration Test

`yolo_object_detector_graph_test.cc` — synthetic tensors for both families:
- Family A: output detections have passed NMS (overlapping boxes removed)
- Family B: output detections are score-threshold-only filtered, box count ≤ model's N (300)

### Ported Tests

`yolo_object_detector_graph_utils_test.cc` — direct port from mediapipe-yolo.

---

## 8. Future PRs (Out of Scope Here)

| Capability | Dependency on this PR |
|---|---|
| Multi-backend inference (ONNX/PyTorch/TensorRT) | `InferenceRunnerFactory` replaces `TfLiteInferenceCalculator`; `MODEL_METADATA` pattern reused |
| Batch inference (B > 1) | `MODEL_METADATA.inputs[0].shape[0]` already carries B; `YoloTensorsToDetectionsCalculator` tensor-slicing logic ready to extend |
| SAHI / tiling | Builds on single-image graph; batch support prerequisite |
| Python Task API | Wraps `YoloObjectDetectorGraph` |
| ByteTrack / BoTSORT tracking | Consumes `DETECTIONS` output of this graph |
