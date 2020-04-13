bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
   --define 3D=true  mediapipe/examples/desktop/hand_landmark:hand_landmark_cpu__tflite

