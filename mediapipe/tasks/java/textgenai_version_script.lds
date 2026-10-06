VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_com_google_mediapipe_tasks_text_textproofreader_TextProofreader_*;
    Java_com_google_mediapipe_tasks_text_textsummarizer_TextSummarizer_*;

  # Hide everything else.
  local:
    *;
};
