---
layout: default
title: Dataset Preparation with MediaSequence
parent: Solutions
nav_order: 15
---

# Dataset Preparation with MediaSequence
{: .no_toc }

<details close markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>
---

## Overview

MediaPipe is a useful and general framework for media processing that can
assist with research, development, and deployment of ML models. This example
focuses on development by demonstrating how to prepare video data for training
a TensorFlow model.

The MediaSequence library provides an extensive set of tools for storing data in
TensorFlow.SequenceExamples. SequenceExamples provide matched semantics to most
video tasks and are efficient to use with TensorFlow. The sequence semantics
allow for a variable number of annotations per frame, which is necessary for
tasks like video object detection, but very difficult to encode in
TensorFlow.Examples. The goal of MediaSequence is to simplify working with
SequenceExamples and to automate common preparation tasks. Much more information
is available about the MediaSequence pipeline, including how to use it to
process new data sets, in the documentation of
[MediaSequence](https://github.com/google/mediapipe/tree/master/mediapipe/util/sequence).

## Preparing an example data set

1.  Checkout the mediapipe repository

    ```bash
    git clone https://github.com/google/mediapipe.git
    cd mediapipe
    ```

1.  Compile the MediaSequence demo C++ binary

    ```bash
    bazel build -c opt mediapipe/examples/desktop/media_sequence:media_sequence_demo --define MEDIAPIPE_DISABLE_GPU=1
    ```

    MediaSequence uses C++ binaries to improve multimedia processing speed and
    encourage a strong separation between annotations and the image data or
    other features. The binary code is very general in that it reads from files
    into input side packets and writes output side packets to files when
    completed, but it also links in all of the calculators for necessary for the
    MediaPipe graphs preparing the Charades data set.

1.  Download and prepare the data set through Python

    To run this step, you must have Python 2.7 or 3.5+ installed with the
    TensorFlow 1.14+ package installed.

    ```bash
    python -m mediapipe.examples.desktop.media_sequence.demo_dataset \
      --path_to_demo_data=/tmp/demo_data/ \
      --path_to_mediapipe_binary=bazel-bin/mediapipe/examples/desktop/media_sequence/media_sequence_demo \
      --path_to_graph_directory=mediapipe/graphs/media_sequence/
    ```

    The arguments define where data is stored. `--path_to_demo_data` defines
    where the data will be downloaded to and where prepared data will be
    generated. `--path_to_mediapipe_binary` is the path to the binary built in
    the previous step. `--path_to_graph_directory` defines where to look for
    MediaPipe graphs during processing.

    Running this module

    1.  Downloads videos from the internet.
    1.  For each annotation in a CSV, creates a structured metadata file.
    1.  Runs MediaPipe to extract images as defined by the metadata.
    1.  Stores the results in numbered set of TFRecords files.

    MediaSequence uses SequenceExamples as the format of both inputs and
    outputs. Annotations are encoded as inputs in a SequenceExample of metadata
    that defines the labels and the path to the cooresponding video file. This
    metadata is passed as input to the C++ `media_sequence_demo` binary, and the
    output is a SequenceExample filled with images and annotations ready for
    model training.

1.  Reading the data in TensorFlow

    To read the data in tensorflow, first add the repo to your PYTHONPATH

    ```bash
    PYTHONPATH="${PYTHONPATH};"+`pwd`
    ```

    and then you can import the data set in Python using
    [read_demo_dataset.py](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/media_sequence/read_demo_dataset.py)

## Preparing a practical data set

As an example of processing a practical data set, a similar set of commands will
prepare the [Charades data set](https://allenai.org/plato/charades/). The
Charades data set is a data set of human action recognition collected with and
maintained by the Allen Institute for Artificial Intelligence. To follow this
code lab, you must abide by the
[license](https://allenai.org/plato/charades/license.txt) for the Charades data
set provided by the Allen Institute.

The Charades data set is large (~150 GB), and will take considerable time to
download and process (4-8 hours).

```bash
bazel build -c opt mediapipe/examples/desktop/media_sequence:media_sequence_demo --define MEDIAPIPE_DISABLE_GPU=1

python -m mediapipe.examples.desktop.media_sequence.charades_dataset \
  --alsologtostderr \
  --path_to_charades_data=/tmp/demo_data/ \
  --path_to_mediapipe_binary=bazel-bin/mediapipe/examples/desktop/media_sequence/media_sequence_demo \
  --path_to_graph_directory=mediapipe/graphs/media_sequence/
```

## Preparing your own data set

The process for preparing your own data set is described in the
[MediaSequence documentation](https://github.com/google/mediapipe/tree/master/mediapipe/util/sequence).
The Python code for Charades can easily be modified to process most annotations,
but the MediaPipe processing warrants further discussion. MediaSequence uses
MediaPipe graphs to extract features related to the metadata or previously
extracted data. Each graph can focus on extracting a single type of feature, and
graphs can be chained together to extract derived features in a composable way.
For example, one graph may extract images from a video at 10 fps and another
graph extract images at 24 fps. A subsequent graph can extract ResNet-50
features from the output of either preceding graph. MediaPipe enables a
composable interface of data process for machine learning at multiple levels.

The MediaPipe graph with brief annotations for adding images to a data set is as
follows. Common changes would be to change the frame_rate or encoding quality of
frames.

```
# Convert the string input into a decoded SequenceExample.
node {
  calculator: "StringToSequenceExampleCalculator"
  input_side_packet: "STRING:input_sequence_example"
  output_side_packet: "SEQUENCE_EXAMPLE:parsed_sequence_example"
}

# Unpack the data path and clip timing from the SequenceExample.
node {
  calculator: "UnpackMediaSequenceCalculator"
  input_side_packet: "SEQUENCE_EXAMPLE:parsed_sequence_example"
  output_side_packet: "DATA_PATH:input_video_path"
  output_side_packet: "RESAMPLER_OPTIONS:packet_resampler_options"
  options {
    [type.googleapis.com/mediapipe.UnpackMediaSequenceCalculatorOptions]: {
      base_packet_resampler_options {
        frame_rate: 24.0
        base_timestamp: 0
      }
    }
  }
}

# Decode the entire video.
node {
  calculator: "OpenCvVideoDecoderCalculator"
  input_side_packet: "INPUT_FILE_PATH:input_video_path"
  output_stream: "VIDEO:decoded_frames"
}

# Extract the subset of frames we want to keep.
node {
  calculator: "PacketResamplerCalculator"
  input_stream: "decoded_frames"
  output_stream: "sampled_frames"
  input_side_packet: "OPTIONS:packet_resampler_options"
}

# Encode the images to store in the SequenceExample.
node {
  calculator: "OpenCvImageEncoderCalculator"
  input_stream: "sampled_frames"
  output_stream: "encoded_frames"
  node_options {
    [type.googleapis.com/mediapipe.OpenCvImageEncoderCalculatorOptions]: {
      quality: 80
    }
  }
}

# Store the images in the SequenceExample.
node {
  calculator: "PackMediaSequenceCalculator"
  input_side_packet: "SEQUENCE_EXAMPLE:parsed_sequence_example"
  output_side_packet: "SEQUENCE_EXAMPLE:sequence_example_to_serialize"
  input_stream: "IMAGE:encoded_frames"
}

# Serialize the SequenceExample to a string for storage.
node {
  calculator: "StringToSequenceExampleCalculator"
  input_side_packet: "SEQUENCE_EXAMPLE:sequence_example_to_serialize"
  output_side_packet: "STRING:output_sequence_example"
}
```
