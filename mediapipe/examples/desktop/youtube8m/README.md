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

### Steps to run the YouTube-8M inference graph with the YT8M dataset

1.  Download the YT8M dataset

    For example, download one shard of the training data:

    ```bash
    curl http://us.data.yt8m.org/2/frame/train/trainpj.tfrecord --output /tmp/mediapipe/trainpj.tfrecord
    ```

2.  Copy the baseline model [(model card)](https://drive.google.com/file/d/1xTCi9-Nm9dt2KIk8WR0dDFrIssWawyXy/view) to local.

    ```bash
    curl -o /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz http://data.yt8m.org/models/baseline/saved_model.tar.gz

    tar -xvf /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz -C /tmp/mediapipe
    ```

3.  Build and run the inference binary.

    ```bash
    bazel build -c opt --define='MEDIAPIPE_DISABLE_GPU=1' --linkopt=-s \
    mediapipe/examples/desktop/youtube8m:model_inference

    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/model_inference \
      --calculator_graph_config_file=mediapipe/graphs/youtube8m/yt8m_dataset_model_inference.pbtxt \
      --input_side_packets=tfrecord_path=/tmp/mediapipe/trainpj.tfrecord,record_index=0,desired_segment_size=5 \
      --output_stream=annotation_summary \
      --output_stream_file=/tmp/summary \
      --output_side_packets=yt8m_id \
      --output_side_packets_file=/tmp/yt8m_id
    ```

### Steps to run the YouTube-8M model inference graph with Web Interface

1.  Copy the baseline model [(model card)](https://drive.google.com/file/d/1xTCi9-Nm9dt2KIk8WR0dDFrIssWawyXy/view) to local.


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

### Steps to run the YouTube-8M model inference graph with a local video

1.  Make sure you have the features.pb from the feature extraction pipeline.

2.  Copy the baseline model [(model card)](https://drive.google.com/file/d/1xTCi9-Nm9dt2KIk8WR0dDFrIssWawyXy/view) to local.

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
