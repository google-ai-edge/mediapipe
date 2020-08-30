---
layout: default
title: YouTube-8M Feature Extraction and Model Inference
parent: Solutions
nav_order: 14
---

# YouTube-8M Feature Extraction and Model Inference
{: .no_toc }

1. TOC
{:toc}
---

MediaPipe is a useful and general framework for media processing that can assist
with research, development, and deployment of ML models. This example focuses on
model development by demonstrating how to prepare training data and do model
inference for the YouTube-8M Challenge.

## Extracting Video Features for YouTube-8M Challenge

[Youtube-8M Challenge](https://www.kaggle.com/c/youtube8m-2019) is an annual
video classification challenge hosted by Google. Over the last two years, the
first two challenges have collectively drawn 1000+ teams from 60+ countries to
further advance large-scale video understanding research. In addition to the
feature extraction Python code released in the
[google/youtube-8m](https://github.com/google/youtube-8m/tree/master/feature_extractor)
repo, we release a MediaPipe based feature extraction pipeline that can extract
both video and audio features from a local video. The MediaPipe based pipeline
utilizes two machine learning models,
[Inception v3](https://github.com/tensorflow/models/tree/master/research/inception)
and
[VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish),
to extract features from video and audio respectively.

To visualize the
[graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/youtube8m/feature_extraction.pbtxt),
copy the text specification of the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). The feature extraction
pipeline is highly customizable. You are welcome to add new calculators or use
your own machine learning models to extract more advanced features from the
videos.

### Steps to run the YouTube-8M feature extraction graph

1.  Checkout the repository and follow
    [the installation instructions](https://github.com/google/mediapipe/blob/master/mediapipe/docs/install.md)
    to set up MediaPipe.

    ```bash
    git clone https://github.com/google/mediapipe.git
    cd mediapipe
    ```

2.  Download the PCA and model data.

    ```bash
    mkdir /tmp/mediapipe
    cd /tmp/mediapipe
    curl -O http://data.yt8m.org/pca_matrix_data/inception3_mean_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/inception3_projection_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/vggish_mean_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/vggish_projection_matrix_data.pb
    curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    tar -xvf /tmp/mediapipe/inception-2015-12-05.tgz
    ```

3.  Get the VGGish frozen graph.

    Note: To run step 3 and step 4, you must have Python 2.7 or 3.5+ installed
    with the TensorFlow 1.14+ package installed.

    ```bash
    # cd to the root directory of the MediaPipe repo
    cd -

    pip3 install tf_slim
    python -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph
    ```

4.  Generate a MediaSequence metadata from the input video.

    Note: the output file is /tmp/mediapipe/metadata.pb

    ```bash
    # change clip_end_time_sec to match the length of your video.
    python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
      --path_to_input_video=/absolute/path/to/the/local/video/file \
      --clip_end_time_sec=120
    ```

5.  Run the MediaPipe binary to extract the features.

    ```bash
    bazel build -c opt --linkopt=-s \
      --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true \
      mediapipe/examples/desktop/youtube8m:extract_yt8m_features

    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
      --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
      --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.pb  \
      --output_side_packets=output_sequence_example=/tmp/mediapipe/features.pb
    ```

6.  [Optional] Read the features.pb in Python.

    ```
    import tensorflow as tf

    sequence_example = open('/tmp/mediapipe/features.pb', 'rb').read()
    print(tf.train.SequenceExample.FromString(sequence_example))
    ```

## Model Inference for YouTube-8M Challenge

MediaPipe can help you do model inference for YouTube-8M Challenge with both
local videos and the YouTube-8M dataset. To visualize
[the graph for local videos](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/youtube8m/local_video_model_inference.pbtxt)
and
[the graph for the YouTube-8M dataset](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/youtube8m/yt8m_dataset_model_inference.pbtxt),
copy the text specification of the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). We use the baseline model
[(model card)](https://drive.google.com/file/d/1xTCi9-Nm9dt2KIk8WR0dDFrIssWawyXy/view)
in our example. But, the model inference pipeline is highly customizable. You
are welcome to add new calculators or use your own machine learning models to do
the inference for both local videos and the dataset

### Steps to run the YouTube-8M model inference graph with Web Interface

1.  Copy the baseline model
    [(model card)](https://drive.google.com/file/d/1xTCi9-Nm9dt2KIk8WR0dDFrIssWawyXy/view)
    to local.

    ```bash
    curl -o /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz http://data.yt8m.org/models/baseline/saved_model.tar.gz

    tar -xvf /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz -C /tmp/mediapipe
    ```

2.  Build the inference binary.

    ```bash
    bazel build -c opt --define='MEDIAPIPE_DISABLE_GPU=1' --linkopt=-s \
      mediapipe/examples/desktop/youtube8m:model_inference
    ```

3.  Run the python web server.

    Note: pip3 install absl-py

    ```bash
    python mediapipe/examples/desktop/youtube8m/viewer/server.py --root `pwd`
    ```

    Navigate to localhost:8008 in a web browser.
    [Here](https://drive.google.com/file/d/19GSvdAAuAlACpBhHOaqMWZ_9p8bLUYKh/view?usp=sharing)
    is a demo video showing the steps to use this web application. Also please
    read
    [youtube8m/README.md](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/youtube8m/README.md)
    if you prefer to run the underlying model_inference binary in command line.

### Steps to run the YouTube-8M model inference graph with a local video

1.  Make sure you have the features.pb from the feature extraction pipeline.

2.  Copy the baseline model
    [(model card)](https://drive.google.com/file/d/1xTCi9-Nm9dt2KIk8WR0dDFrIssWawyXy/view)
    to local.

    ```bash
    curl -o /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz http://data.yt8m.org/models/baseline/saved_model.tar.gz

    tar -xvf /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz -C /tmp/mediapipe
    ```

3.  Build and run the inference binary.

    ```bash
    bazel build -c opt --define='MEDIAPIPE_DISABLE_GPU=1' --linkopt=-s \
      mediapipe/examples/desktop/youtube8m:model_inference

    # segment_size is the number of seconds window of frames.
    # overlap is the number of seconds adjacent segments share.
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/model_inference \
      --calculator_graph_config_file=mediapipe/graphs/youtube8m/local_video_model_inference.pbtxt \
      --input_side_packets=input_sequence_example_path=/tmp/mediapipe/features.pb,input_video_path=/absolute/path/to/the/local/video/file,output_video_path=/tmp/mediapipe/annotated_video.mp4,segment_size=5,overlap=4
    ```

4.  View the annotated video.
