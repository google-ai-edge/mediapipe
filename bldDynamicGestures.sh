bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
   --define 3D=true  myMediapipe/projects/dynamicGestures:dynamic_gestures_cpu_tflite

