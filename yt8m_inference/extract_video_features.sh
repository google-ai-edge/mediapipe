#!/bin/bash


video_name=$1
echo $video_name

time_float=$(ffprobe -i $video_name -show_entries format=duration -v quiet -of csv="p=0")
time_int=${time_float/.*} 
echo $time_int

python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example --path_to_input_video=$video_name --clip_end_time_sec=$time_int
  
  
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.pb  --output_side_packets=output_sequence_example=/tmp/mediapipe/features.pb