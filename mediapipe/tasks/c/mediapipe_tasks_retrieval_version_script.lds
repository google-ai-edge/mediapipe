# The retrieval library is self-contained: the semantic retriever and the
# universal embedder consume images and text/image embedders directly, so the
# corresponding vision/text C APIs are compiled into (and exported from) this
# library as well. Handles must not be shared with the vision/text libraries.
VERS_1.0 {
  global:
    MpErrorFree;
    MpFlatbuffer*;
    MpImage*;
    MpRemoveLoggingCallback;
    MpSemanticRetriever*;
    MpSetLoggingCallback;
    MpStringList*;
    MpTextEmbedder*;
    MpUniversalEmbedder*;
  local:
    *;
};
