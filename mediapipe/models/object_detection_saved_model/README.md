## TensorFlow/TFLite Object Detection Model

### TensorFlow model

The model is trained on [MSCOCO 2014](http://cocodataset.org) dataset using [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). It is a MobileNetV2-based SSD model with 0.5 depth multiplier. Detailed training configuration is in the provided `pipeline.config`. The model is a relatively compact model which has `0.171 mAP` to achieve real-time performance on mobile devices. You can compare it with other models from the [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).


### TFLite model

The TFLite model is converted from the TensorFlow above. The steps needed to convert the model are similar to [this tutorial](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193) with minor modifications. Assuming now we have a trained TensorFlow model which includes the checkpoint files and the training configuration file, for example the files provided in this repo:

   * `model.ckpt.index`
   * `model.ckpt.meta`
   * `model.ckpt.data-00000-of-00001`
   * `pipeline.config`

Make sure you have installed these [python libraries](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md). Then to get the frozen graph, run the `export_tflite_ssd_graph.py` script from the `models/research` directory with this command:

```bash
$ PATH_TO_MODEL=path/to/the/model
$ bazel run object_detection:export_tflite_ssd_graph -- \
    --pipeline_config_path ${PATH_TO_MODEL}/pipeline.config \
    --trained_checkpoint_prefix ${PATH_TO_MODEL}/model.ckpt \
    --output_directory ${PATH_TO_MODEL} \
    --add_postprocessing_op=False
```

The exported model contains two files:

   * `tflite_graph.pb`
   * `tflite_graph.pbtxt`

The difference between this step and the one in [the tutorial](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193) is that we set `add_postprocessing_op` to False. In MediaPipe, we have provided all the calculators needed for post-processing such that we can exclude the custom TFLite ops for post-processing in the original graph, e.g., non-maximum suppression. This enables the flexibility to integrate with different post-processing algorithms and implementations.

Optional: You can install and use the [graph tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms) to inspect the input/output of the exported model:

```bash
$ bazel run graph_transforms:summarize_graph -- \
    --in_graph=${PATH_TO_MODEL}/tflite_graph.pb
```

You should be able to see the input image size of the model is 320x320 and the outputs of the model are:

   * `raw_outputs/box_encodings`
   * `raw_outputs/class_predictions`

The last step is to convert the model to TFLite. You can look at [this guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/cmdline_examples.md) for more detail. For this example, you just need to run:

```bash
$ tflite_convert --  \
  --graph_def_file=${PATH_TO_MODEL}/tflite_graph.pb \
  --output_file=${PATH_TO_MODEL}/model.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shapes=1,320,320,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=raw_outputs/box_encodings,raw_outputs/class_predictions

```

Now you have the TFLite model `model.tflite` ready to use with MediaPipe Object Detection graphs. Please see the examples for more detail.
