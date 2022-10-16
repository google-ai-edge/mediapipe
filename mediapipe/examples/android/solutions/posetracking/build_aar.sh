POSE_TRACKING_OUTPUT_DIR=bazel-bin/mediapipe/java/com/google/mediapipe/solutions/posetracking
GRADLE_LIBS_DIR=mediapipe/examples/android/solutions/posetracking/libs
#
bazel build     -c opt --strip=ALWAYS\
                --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
                --fat_apk_cpu=arm64-v8a,armeabi-v7a \
                --legacy_whole_archive=0 \
                --features=-legacy_whole_archive \
                --copt=-fvisibility=hidden \
                --copt=-ffunction-sections \
                --copt=-fdata-sections \
                --copt=-fstack-protector \
                --copt=-Oz \
                --copt=-fomit-frame-pointer \
                --copt=-DABSL_MIN_LOG_LEVEL=2 \
                --linkopt=-Wl,--gc-sections,--strip-all \
                //mediapipe/java/com/google/mediapipe/solutions/posetracking:copperlabs-pose-api.aar \
                //mediapipe/java/com/google/mediapipe/solutions/posetracking:copperlabs-pose-landmark.aar \
                //mediapipe/java/com/google/mediapipe/solutions/posetracking:copperlabs-pose-detection.aar \
                //mediapipe/java/com/google/mediapipe/solutions/posetracking:copperlabs-pose-graph.aar \
                //mediapipe/java/com/google/mediapipe/solutioncore:copperlabs-mediapipe


mkdir $GRADLE_LIBS_DIR
rm -f $GRADLE_LIBS_DIR/copperlabs-*.aar

\cp $POSE_TRACKING_OUTPUT_DIR/copperlabs-pose-api.aar $GRADLE_LIBS_DIR
\cp $POSE_TRACKING_OUTPUT_DIR/copperlabs-pose-detection.aar $GRADLE_LIBS_DIR
\cp $POSE_TRACKING_OUTPUT_DIR/copperlabs-pose-graph.aar $GRADLE_LIBS_DIR
\cp $POSE_TRACKING_OUTPUT_DIR/copperlabs-pose-landmark.aar $GRADLE_LIBS_DIR
\cp bazel-bin/mediapipe/java/com/google/mediapipe/solutioncore/copperlabs-mediapipe.aar $GRADLE_LIBS_DIR

