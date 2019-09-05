## Extracting Video Features for YouTube-8M Challenge

MediaPipe is a useful and general framework for media processing that can assist
with research, development, and deployment of ML models. This example focuses on
model development by demonstrating how to prepare training data for the
YouTube-8M Challenge.

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

1.  Checkout the mediapipe repository

    ```bash
    git clone https://github.com/google/mediapipe.git
    cd mediapipe
    ```

2.  Download the PCA and model data

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

3.  Get the VGGish frozen graph

    Note: To run step 3 and step 4, you must have Python 2.7 or 3.5+ installed
    with the TensorFlow 1.14+ package installed.

    ```bash
    # cd to the root directory of the MediaPipe repo
    cd -
    python -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph
    ```

4.  Generate a MediaSequence metadata from the input video

    Note: the output file is /tmp/mediapipe/metadata.tfrecord

    ```bash
    python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
      --path_to_input_video=/absolute/path/to/the/local/video/file
    ```

5.  Run the MediaPipe binary to extract the features

    ```bash
    bazel build -c opt \
      --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true \
      mediapipe/examples/desktop/youtube8m:extract_yt8m_features

    ./bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features
      --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
      --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.tfrecord  \
      --output_side_packets=output_sequence_example=/tmp/mediapipe/output.tfrecord
    ```
