VERS_1.0 {
  # Export JNI symbols.
  global:
    Java_com_google_mediapipe_tasks_decision_DecisionMaker_*;

  # Hide everything else.
  local:
    *;
};
