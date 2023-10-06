VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_com_google_mediapipe_framework_AndroidAssetUtil*;
    Java_com_google_mediapipe_framework_AndroidPacketCreator*;
    Java_com_google_mediapipe_framework_Graph_nativeAddMultiStreamCallback;
    Java_com_google_mediapipe_framework_Graph_nativeAddPacketToInputStream;
    Java_com_google_mediapipe_framework_Graph_nativeCloseAllPacketSources;
    Java_com_google_mediapipe_framework_Graph_nativeCreateGraph;
    Java_com_google_mediapipe_framework_Graph_nativeGetCalculatorGraphConfig;
    Java_com_google_mediapipe_framework_Graph_nativeLoadBinaryGraph*;
    Java_com_google_mediapipe_framework_Graph_nativeMovePacketToInputStream;
    Java_com_google_mediapipe_framework_Graph_nativeReleaseGraph;
    Java_com_google_mediapipe_framework_Graph_nativeStartRunningGraph;
    Java_com_google_mediapipe_framework_Graph_nativeWaitUntilGraphDone;
    Java_com_google_mediapipe_framework_Graph_nativeWaitUntilGraphIdle;
    Java_com_google_mediapipe_framework_PacketCreator*;
    Java_com_google_mediapipe_framework_PacketGetter*;
    Java_com_google_mediapipe_framework_Packet*;
    Java_com_google_mediapipe_tasks_core_ModelResourcesCache*;

  # Hide everything else.
  local:
    *;
};
