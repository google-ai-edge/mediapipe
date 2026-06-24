"""External file definitions for MediaPipe.

This file is auto-generated.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

# buildifier: disable=unnamed-macro
def external_files():
    """External file definitions for MediaPipe."""

    http_file(
        name = "com_google_mediapipe_model_maker_models_gesture_recognizer_canned_gesture_classifier_tflite",
        sha256 = "ee121d85979de1b86126faabb0a0f4d2e4039c3e33e2cd687db50571001b24d0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/gesture_recognizer/canned_gesture_classifier.tflite?generation=1782183226378603"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_gesture_recognizer_gesture_embedder_tflite",
        sha256 = "927e4f6cbe6451da6b4fd1485e2576a6f8dbd95062666661cbd9dea893c41d01",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/gesture_recognizer/gesture_embedder.tflite?generation=1782183262639516"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_gesture_recognizer_gesture_embedder_keras_metadata_pb",
        sha256 = "c76b856101e2284293a5e5963b7c445e407a0b3e56ec63eb78f64d883e51e3aa",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/gesture_recognizer/gesture_embedder/keras_metadata.pb?generation=1782183233656693"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_gesture_recognizer_gesture_embedder_saved_model_pb",
        sha256 = "0082d37c5b85487fbf553e00a63f640945faf3da2d561a5f5a24c3194fecda6a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/gesture_recognizer/gesture_embedder/saved_model.pb?generation=1782183240839461"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_gesture_recognizer_gesture_embedder_variables_variables_data-00000-of-00001",
        sha256 = "c156c9654c9ffb1091bb9f06c71080bd1e428586276d3f39c33fbab27fe0522d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/gesture_recognizer/gesture_embedder/variables/variables.data-00000-of-00001?generation=1782229649693902"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_gesture_recognizer_gesture_embedder_variables_variables_index",
        sha256 = "76ea482b8da6bdb3d65d3b2ea989c1699c9fa0d6df0cb6d80863d1dc6fe7c4bd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/gesture_recognizer/gesture_embedder/variables/variables.index?generation=1782183255213096"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_gesture_recognizer_hand_landmark_full_tflite",
        sha256 = "11c272b891e1a99ab034208e23937a8008388cf11ed2a9d776ed3d01d0ba00e3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/gesture_recognizer/hand_landmark_full.tflite?generation=1782183270304229"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_gesture_recognizer_palm_detection_full_tflite",
        sha256 = "1b14e9422c6ad006cde6581a46c8b90dd573c07ab7f3934b5589e7cea3f89a54",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/gesture_recognizer/palm_detection_full.tflite?generation=1782183277660934"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_text_classifier_mobilebert_tiny_keras_metadata_pb",
        sha256 = "cef8131a414c602b9d4742ac57f4f90bc5d8a42baec36b65deece884e2d0cf0f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/text_classifier/mobilebert_tiny/keras_metadata.pb?generation=1782183291744100"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_text_classifier_mobilebert_tiny_saved_model_pb",
        sha256 = "323c997cd3e17df1b2e3bdebe3cfe2b17c5ffd9488a26a4afb59ee819196837a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/text_classifier/mobilebert_tiny/saved_model.pb?generation=1782183299359293"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_text_classifier_mobilebert_tiny_variables_variables_data-00000-of-00001",
        sha256 = "c3857370046cd3a2f345657cf1bb259a4e7e09185d7f0808e57803e9d41ebba4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/text_classifier/mobilebert_tiny/variables/variables.data-00000-of-00001?generation=1782229654892505"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_models_text_classifier_mobilebert_tiny_variables_variables_index",
        sha256 = "4df4d7c0fefe99903ab6ebf44b7478196ce613082d2ca692a5a37a7f24e562ed",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/models/text_classifier/mobilebert_tiny/variables/variables.index?generation=1782183316041709"],
    )

    http_file(
        name = "com_google_mediapipe_model_maker_python_core_data_testdata_test_jpg",
        sha256 = "798a12a466933842528d8438f553320eebe5137f02650f12dd68706a2f94fb4f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_maker/python/core/data/testdata/test.jpg?generation=1782183323252159"],
    )

    http_file(
        name = "com_google_mediapipe_models_embedding_gemma_embedding_gemma_task",
        sha256 = "913b7a1edc7c7c3d1da3979ec1d0648ed9e0a370f181bb59ab177ca4b97707ad",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/embedding_gemma/embedding_gemma.task?generation=1782183336674827"],
    )

    http_file(
        name = "com_google_mediapipe_models_face_landmark_tflite",
        sha256 = "2efcb4f4de43c7614b80a3cc3e8a37354b3b3b40f75cce20f6f38f0f25d65493",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/face_landmark.tflite?generation=1782183345452003"],
    )

    http_file(
        name = "com_google_mediapipe_models_hair_segmentation_tflite",
        sha256 = "d2c940c4fd80edeaf38f5d7387d1b4235ee320ed120080df67c663e749e77633",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/hair_segmentation.tflite?generation=1782183352625668"],
    )

    http_file(
        name = "com_google_mediapipe_models_hand_landmark_tflite",
        sha256 = "bad88ac1fd144f034e00f075afcade4f3a21d0d09c41bee8dd50504dacd70efd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/hand_landmark.tflite?generation=1782183360083038"],
    )

    http_file(
        name = "com_google_mediapipe_models_iris_landmark_tflite",
        sha256 = "d1744d2a09c25f501d39eba4faff47e53ecca8852c5ce19bce8eeac39357521f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/iris_landmark.tflite?generation=1782183388820309"],
    )

    http_file(
        name = "com_google_mediapipe_models_knift_float_tflite",
        sha256 = "40567854c2c1022c98cd2c55a7eef1c60999580ce67db118c1274000d0e22ace",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/knift_float.tflite?generation=1782183396096986"],
    )

    http_file(
        name = "com_google_mediapipe_models_knift_float_1k_tflite",
        sha256 = "5dbfa98c7a3caae97840576a278a1d1fe37c86bad4007d1acdffec094242837c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/knift_float_1k.tflite?generation=1782183403508179"],
    )

    http_file(
        name = "com_google_mediapipe_models_knift_float_400_tflite",
        sha256 = "3ee576050f3d5d45ea19a19dbd67267cb345b0348efde00952eddb8b7aabe2e5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/knift_float_400.tflite?generation=1782183410710646"],
    )

    http_file(
        name = "com_google_mediapipe_models_knift_index_pb",
        sha256 = "2c2b57a846e0adbf1e3f25bd20c7878ac9399460a1ad5d8147e3231ace8eb3dc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/knift_index.pb?generation=1782183417879837"],
    )

    http_file(
        name = "com_google_mediapipe_models_object_detection_saved_model_model_ckpt_data-00000-of-00001",
        sha256 = "ad2f733f271dd5000a8c7f926bfea1083e6408b34d4f3b60679e5a6f96251c97",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/object_detection_saved_model/model.ckpt.data-00000-of-00001?generation=1782183432718390"],
    )

    http_file(
        name = "com_google_mediapipe_models_object_detection_saved_model_model_ckpt_index",
        sha256 = "283816fcab228e6246d1c03b596f50dd40e4fe3e04c52a522a5b9d6f2cc43273",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/object_detection_saved_model/model.ckpt.index?generation=1782183439673255"],
    )

    http_file(
        name = "com_google_mediapipe_models_object_detection_saved_model_model_ckpt_meta",
        sha256 = "9d80696ab76a492a23f6ce1d0d33b2d13c26e118b86d3ef61b691ad67d0f1f5a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/object_detection_saved_model/model.ckpt.meta?generation=1782183447266505"],
    )

    http_file(
        name = "com_google_mediapipe_models_object_detection_saved_model_pipeline_config",
        sha256 = "995aff0b28af5f66eb98d0734494395710ae84c843aee207755e7bc5025c9abb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/object_detection_saved_model/pipeline.config?generation=1782183454650795"],
    )

    http_file(
        name = "com_google_mediapipe_models_object_detection_saved_model_saved_model_pb",
        sha256 = "f29606cf218397d5580c496e50fd28cddf66e2f59b819ab9c761b72270a5adf3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/object_detection_saved_model/saved_model.pb?generation=1782183462413855"],
    )

    http_file(
        name = "com_google_mediapipe_models_ssdlite_object_detection_tflite",
        sha256 = "8e10a2e2f5db85d8f90628f00752a89ff241c5b2ca82f3b92fc496c7bda122ef",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/ssdlite_object_detection.tflite?generation=1782183469940468"],
    )

    http_file(
        name = "com_google_mediapipe_models_summarization_200m_summarization_quant_200m_2modes_litertlm",
        sha256 = "8b2d4ef09236adb9ead3127325526ba1aa5a59feb7c5de2d3f5958f27479de59",
        urls = ["https://storage.googleapis.com/mediapipe-assets/models/summarization_200m/summarization_quant_200m_2modes.litertlm?generation=1782183486368100"],
    )

    http_file(
        name = "com_google_mediapipe_modules_face_detection_face_detection_full_range_tflite",
        sha256 = "99bf9494d84f50acc6617d89873f71bf6635a841ea699c17cb3377f9507cfec3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/face_detection/face_detection_full_range.tflite?generation=1782183493595806"],
    )

    http_file(
        name = "com_google_mediapipe_modules_face_detection_face_detection_full_range_sparse_tflite",
        sha256 = "671dd2f9ed11a78436fc21cc42357a803dfc6f73e9fb86541be942d5716c2dce",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/face_detection/face_detection_full_range_sparse.tflite?generation=1782183500697062"],
    )

    http_file(
        name = "com_google_mediapipe_modules_face_detection_face_detection_short_range_tflite",
        sha256 = "3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/face_detection/face_detection_short_range.tflite?generation=1782183508198291"],
    )

    http_file(
        name = "com_google_mediapipe_modules_face_landmark_face_landmark_tflite",
        sha256 = "c603fa6149219a3e9487dc9abd7a0c24474c77263273d24868378cdf40aa26d1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/face_landmark/face_landmark.tflite?generation=1782183515518124"],
    )

    http_file(
        name = "com_google_mediapipe_modules_face_landmark_face_landmark_with_attention_tflite",
        sha256 = "883b7411747bac657c30c462d305d312e9dec6adbf8b85e2f5d8d722fca9455d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/face_landmark/face_landmark_with_attention.tflite?generation=1782183522857456"],
    )

    http_file(
        name = "com_google_mediapipe_modules_hand_landmark_hand_landmark_full_tflite",
        sha256 = "8c026882c9ec059ce0f8e75266bee5a9a23c341a40e0000df755374d3d1b9b68",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/hand_landmark/hand_landmark_full.tflite?generation=1782183530499255"],
    )

    http_file(
        name = "com_google_mediapipe_modules_hand_landmark_hand_landmark_lite_tflite",
        sha256 = "d7fde8ac11f8ce03f8663775bfc323f4fc9f2a38062b4f4efa142874ef5b2a48",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/hand_landmark/hand_landmark_lite.tflite?generation=1782183538030829"],
    )

    http_file(
        name = "com_google_mediapipe_modules_holistic_landmark_hand_recrop_tflite",
        sha256 = "67d996ce96f9d36fe17d2693022c6da93168026ab2f028f9e2365398d8ac7d5d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/holistic_landmark/hand_recrop.tflite?generation=1782183545269780"],
    )

    http_file(
        name = "com_google_mediapipe_modules_image_classification_mobile_object_labeler_v1_tflite",
        sha256 = "9400671e04685f5277edd3052a311cc51533de9da94255c52ebde1e18484c77c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/image_classification/mobile_object_labeler_v1.tflite?generation=1782183552534480"],
    )

    http_file(
        name = "com_google_mediapipe_modules_iris_and_gaze_iris_and_gaze_tflite",
        sha256 = "b6dcb860a92a3c7264a8e50786f46cecb529672cdafc17d39c78931257da661d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/iris_and_gaze/iris_and_gaze.tflite?generation=1782183559903211"],
    )

    http_file(
        name = "com_google_mediapipe_modules_iris_landmark_iris_landmark_tflite",
        sha256 = "d1744d2a09c25f501d39eba4faff47e53ecca8852c5ce19bce8eeac39357521f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/iris_landmark/iris_landmark.tflite?generation=1782183567532581"],
    )

    http_file(
        name = "com_google_mediapipe_modules_object_detection_ssd_mobilenet_v1_tflite",
        sha256 = "cbdecd08b44c5dea3821f77c5468e2936ecfbf43cde0795a2729fdb43401e58b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/object_detection/ssd_mobilenet_v1.tflite?generation=1782183575309331"],
    )

    http_file(
        name = "com_google_mediapipe_modules_objectron_object_detection_3d_camera_tflite",
        sha256 = "f66e92e81ed3f4698f74d565a7668e016e2288ea92fb42938e33b778bd1e110d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/objectron/object_detection_3d_camera.tflite?generation=1782183582937846"],
    )

    http_file(
        name = "com_google_mediapipe_modules_objectron_object_detection_3d_chair_tflite",
        sha256 = "190e4ea49ba891ed242ddc73703e03d70164c27f3da07492d7010379e24f2a6b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/objectron/object_detection_3d_chair.tflite?generation=1782183590409653"],
    )

    http_file(
        name = "com_google_mediapipe_modules_objectron_object_detection_3d_chair_1stage_tflite",
        sha256 = "694af9bdcea270f2bad488beb4e5ef89aad819489d5d9aa4a774d2fad2a91ae9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/objectron/object_detection_3d_chair_1stage.tflite?generation=1782183597847489"],
    )

    http_file(
        name = "com_google_mediapipe_modules_objectron_object_detection_3d_cup_tflite",
        sha256 = "c4f4ea8def16bd191d11279f754e6f3f2a9d94839a956b975e5697e943157ac7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/objectron/object_detection_3d_cup.tflite?generation=1782183605371037"],
    )

    http_file(
        name = "com_google_mediapipe_modules_objectron_object_detection_3d_sneakers_tflite",
        sha256 = "4eb1633d646a43ae979ba497487e95dbf89f97406ed02200ae39ae46b0a0543d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/objectron/object_detection_3d_sneakers.tflite?generation=1782183612947778"],
    )

    http_file(
        name = "com_google_mediapipe_modules_objectron_object_detection_3d_sneakers_1stage_tflite",
        sha256 = "ef052353e882d93429ee90a8e8e5e781f04acdf44c0cef4d961d8cbfa89aad8c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/objectron/object_detection_3d_sneakers_1stage.tflite?generation=1782183620496729"],
    )

    http_file(
        name = "com_google_mediapipe_modules_objectron_object_detection_ssd_mobilenetv2_oidv4_fp16_tflite",
        sha256 = "d0a5255bf8c4f5a0bc4240741a76c41d5e939f7655078f945f50ab53a9375da6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/objectron/object_detection_ssd_mobilenetv2_oidv4_fp16.tflite?generation=1782183628179438"],
    )

    http_file(
        name = "com_google_mediapipe_modules_palm_detection_palm_detection_full_tflite",
        sha256 = "2f25e740121983f68ffc05f99991d524dc0ea812134f6316a26125816941ee85",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/palm_detection/palm_detection_full.tflite?generation=1782183635662937"],
    )

    http_file(
        name = "com_google_mediapipe_modules_palm_detection_palm_detection_lite_tflite",
        sha256 = "e9a4aaddf90dda56a87235303cf00e4c2d3fb28725f68fd88772997dac905c18",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/palm_detection/palm_detection_lite.tflite?generation=1782183643309056"],
    )

    http_file(
        name = "com_google_mediapipe_modules_pose_detection_pose_detection_tflite",
        sha256 = "a63c614bef30d35947f13be361820b1e4e3bec9cfeebf4d11216a18373108e85",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/pose_detection/pose_detection.tflite?generation=1782183650790966"],
    )

    http_file(
        name = "com_google_mediapipe_modules_pose_landmark_pose_landmark_full_tflite",
        sha256 = "e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/pose_landmark/pose_landmark_full.tflite?generation=1782183658450386"],
    )

    http_file(
        name = "com_google_mediapipe_modules_pose_landmark_pose_landmark_heavy_tflite",
        sha256 = "59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/pose_landmark/pose_landmark_heavy.tflite?generation=1782183666398574"],
    )

    http_file(
        name = "com_google_mediapipe_modules_pose_landmark_pose_landmark_lite_tflite",
        sha256 = "f17bfbecadb61c3be1baa8b8d851cc6619c870a87167b32848ad20db306b9d61",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/pose_landmark/pose_landmark_lite.tflite?generation=1782183674047673"],
    )

    http_file(
        name = "com_google_mediapipe_modules_selfie_segmentation_selfie_segmentation_tflite",
        sha256 = "8d13b7fae74af625c641226813616a2117bd6bca19eb3b75574621fc08557f27",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/selfie_segmentation/selfie_segmentation.tflite?generation=1782183681393267"],
    )

    http_file(
        name = "com_google_mediapipe_modules_selfie_segmentation_selfie_segmentation_landscape_tflite",
        sha256 = "4aafe6223bb8dac6fac8ca8ed56852870a33051ef3f6238822d282a109962894",
        urls = ["https://storage.googleapis.com/mediapipe-assets/modules/selfie_segmentation/selfie_segmentation_landscape.tflite?generation=1782183688568628"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_audio_model_without_metadata_tflite",
        sha256 = "05c5aea7ae00aeed0053a85f2b2e896b4ea272c5219052d32c06b655fbf5cc9b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/audio/model_without_metadata.tflite?generation=1782183703590697"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_audio_speech_16000_hz_mono_wav",
        sha256 = "71caf50b8757d6ab9cad5eae4d36669d3c20c225a51660afd7fe0dc44cdb74f6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/audio/speech_16000_hz_mono.wav?generation=1782183710848218"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_audio_speech_48000_hz_mono_wav",
        sha256 = "04d4590b61d0519170d7aa0686ab2ff5da2b8487d192e40413dd36d9c1a24304",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/audio/speech_48000_hz_mono.wav?generation=1782183718082660"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_audio_two_heads_tflite",
        sha256 = "bfa6ee4ccaf9180b69b39fa579b26b74bbf7758ae398e1d2265a58d323ca3d84",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/audio/two_heads.tflite?generation=1782183725982913"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_audio_two_heads_16000_hz_mono_wav",
        sha256 = "a291a9c22c39bba30138a26915e154a96286ba6ca3b413053123c504a58cce3b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/audio/two_heads_16000_hz_mono.wav?generation=1782183733219069"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_audio_two_heads_44100_hz_mono_wav",
        sha256 = "1bf525ad7b7bac2da65addb5593b49adaba52ec3a9ed891f70afe0b392db02cd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/audio/two_heads_44100_hz_mono.wav?generation=1782183740447201"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_audio_yamnet_audio_classifier_with_metadata_tflite",
        sha256 = "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/audio/yamnet_audio_classifier_with_metadata.tflite?generation=1782183748090824"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_audio_yamnet_embedding_metadata_tflite",
        sha256 = "7baa72708e3919bae5a5dc78d932847bc28008af14febd083eff62d28af9c72a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/audio/yamnet_embedding_metadata.tflite?generation=1782183755757464"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_core_corrupted_mobilenet_v1_0_25_224_1_default_1_tflite",
        sha256 = "f0cbeb8061f4c693e20de779ce255af923508492e8a24f6db320845a52facb51",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/core/corrupted_mobilenet_v1_0.25_224_1_default_1.tflite?generation=1782183762993657"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_core_dummy_gesture_recognizer_task",
        sha256 = "76de8c58d206d098557959d574953c2db3a4363fa52922ca198450d5d696814d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/core/dummy_gesture_recognizer.task?generation=1782183770196846"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_core_mobilenet_v1_0_25_224_quant_tflite",
        sha256 = "e480eb15572f86d3d5f1be6e83e35b3c7d509ab2bcec353707d1f614e14edca2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/core/mobilenet_v1_0.25_224_quant.tflite?generation=1782183784609446"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_core_test_model_add_op_tflite",
        sha256 = "298300ca8a9193b80ada1dca39d36f20bffeebde09e85385049b3bfe7be2272f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/core/test_model_add_op.tflite?generation=1782183792280660"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_core_test_model_with_custom_op_tflite",
        sha256 = "bafff7c8508ac24846e089ab70dcf48943a483a3e20290ff60e7740d073d7653",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/core/test_model_with_custom_op.tflite?generation=1782183799890422"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_core_test_model_without_custom_op_tflite",
        sha256 = "e17f0a1a22bc9242d9f825fe1edce07d2f90eb2a57e8b29a996244f194ee08a0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/core/test_model_without_custom_op.tflite?generation=1782183806998197"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_30k-clean_model",
        sha256 = "fefb02b667a6c5c2fe27602d28e5fb3428f66ab89c7d6f388e7c8d44a02d0336",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/30k-clean.model?generation=1782183814286955"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_associated_file_meta_json",
        sha256 = "5b2cba11ae893e1226af6570813955889e9f171d6d2c67b3e96ecb6b96d8c681",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/associated_file_meta.json?generation=1782183821478132"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_bert_text_classifier_no_metadata_tflite",
        sha256 = "9b4554f6e28a72a3f40511964eed1ccf4e74cc074f81543cacca4faf169a173e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/bert_text_classifier_no_metadata.tflite?generation=1782183829182083"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_bert_text_classifier_with_bert_tokenizer_json",
        sha256 = "49f148a13a4e3b486b1d3c2400e46e5ebd0d375674c0154278b835760e873a95",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/bert_text_classifier_with_bert_tokenizer.json?generation=1782183836617749"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_bert_text_classifier_with_sentence_piece_json",
        sha256 = "113091f3892691de57e379387256b2ce0cc18a1b5185af866220a46da8221f26",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/bert_text_classifier_with_sentence_piece.json?generation=1782183844260286"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_bert_tokenizer_meta_json",
        sha256 = "116d70c7c3ef413a8bff54ab758f9ed3d6e51fdc5621d8c920ad2f0035831804",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/bert_tokenizer_meta.json?generation=1782183851654585"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_bounding_box_tensor_meta_json",
        sha256 = "cc019cee86529955a24a3d43ca3d778fa366bcb90d67c8eaf55696789833841a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/bounding_box_tensor_meta.json?generation=1782183858905257"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_category_tensor_float_meta_json",
        sha256 = "d0cbe95a99ffc57046d7e66cf194600d12899216a4d3bf1a3851811648005e38",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/category_tensor_float_meta.json?generation=1782183866044980"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_classification_tensor_float_meta_json",
        sha256 = "1d10b1c9c87eabac330651136804074ddc134779e94a73cf783207c3aa2a5619",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/classification_tensor_float_meta.json?generation=1782183873554118"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_classification_tensor_uint8_meta_json",
        sha256 = "74f4d64ee0017d11e0fdc975a88d974d73b72b889fd4d67992356052edde0f1e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/classification_tensor_uint8_meta.json?generation=1782183880900751"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_classification_tensor_unsupported_meta_json",
        sha256 = "4810ad8a00f0078c6a693114d00f692aa70ff2d61030a6e516db1e654707e208",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/classification_tensor_unsupported_meta.json?generation=1782183888196416"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_coco_ssd_mobilenet_v1_1_0_quant_2018_06_29_no_metadata_tflite",
        sha256 = "e4b118e5e4531945de2e659742c7c590f7536f8d0ed26d135abcfe83b4779d13",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_no_metadata.tflite?generation=1782229659216632"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_coco_ssd_mobilenet_v1_score_calibration_json",
        sha256 = "a850674f9043bfc775527fee7f1b639f7fe0fb56e8d3ed2b710247967c888b09",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/coco_ssd_mobilenet_v1_score_calibration.json?generation=1782183902916379"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_deeplabv3_json",
        sha256 = "f299835bd9ea1cceb25fdf40a761a22716cbd20025cd67c365a860527f178b7f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/deeplabv3.json?generation=1782183910189668"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_deeplabv3_with_activation_json",
        sha256 = "a7633476d02f970db3cc30f5f027bcb608149e02207b2ccae36a4b69d730c82c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/deeplabv3_with_activation.json?generation=1782183917348797"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_deeplabv3_without_labels_json",
        sha256 = "7d045a583a4046f17a52d2078b0175607a45ed0cc187558325f9c66534c08401",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/deeplabv3_without_labels.json?generation=1782183924269863"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_deeplabv3_without_metadata_tflite",
        sha256 = "68a539782c2c6a72f8aac3724600124a85ed977162b44e84cbae5db717c933c6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/deeplabv3_without_metadata.tflite?generation=1782183931600645"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_efficientdet_lite0_fp16_no_nms_json",
        sha256 = "dc3b333e41c43fb49ace048c25c18d0e34df78fb5ee77edbe169264368f78b92",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/efficientdet_lite0_fp16_no_nms.json?generation=1782183938795314"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_efficientdet_lite0_fp16_no_nms_tflite",
        sha256 = "bcda125c96d3767bca894c8cbe7bc458379c9974c9fd8bdc6204e7124a74082a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/efficientdet_lite0_fp16_no_nms.tflite?generation=1782183946422124"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_efficientdet_lite0_fp16_no_nms_anchors_csv",
        sha256 = "284475a0f16e34afcc6c0fe68b05bd871aca5b20c83db0870c6a36dd63827176",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/efficientdet_lite0_fp16_no_nms_anchors.csv?generation=1782183953547895"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_efficientdet_lite0_v1_json",
        sha256 = "ef9706696a3ea5d87f4324ac56e877a92033d33e522c4b7d5a416fbcab24d8fc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/efficientdet_lite0_v1.json?generation=1782183960962195"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_efficientdet_lite0_v1_tflite",
        sha256 = "f97efd21d6009a7b4b94b3e57baaeb77ec3225b42d32477f5003736a8084c081",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/efficientdet_lite0_v1.tflite?generation=1782183968410290"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_face_stylizer_json",
        sha256 = "ad89860d5daba6a1c4163a576428713fc3ddab76d6bbaf06d675164423ae159f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/face_stylizer.json?generation=1782183983095348"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_feature_tensor_meta_json",
        sha256 = "b2c30ddfd495956ce81085f8a143422f4310b002cfbf1c594ff2ee0576e29d6f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/feature_tensor_meta.json?generation=1782183990314321"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_general_meta_json",
        sha256 = "b95363e4bae89b9c2af484498312aaad4efc7ff57c7eadcc4e5e7adca641445f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/general_meta.json?generation=1782183997231619"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_golden_json_json",
        sha256 = "55c0c88748d099aa379930504df62c6c8f1d8874ea52d2f8a925f352c4c7f09c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/golden_json.json?generation=1782184004484897"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_image_tensor_meta_json",
        sha256 = "aad86fde3defb379c82ff7ee48e50493a58529cdc0623cf0d7bf135c3577060e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/image_tensor_meta.json?generation=1782184011676665"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_input_image_tensor_float_meta_json",
        sha256 = "426ecf5c3ace61db3936b950c3709daece15827ea21905ddbcdc81b1c6e70232",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/input_image_tensor_float_meta.json?generation=1782184018875345"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_input_image_tensor_uint8_meta_json",
        sha256 = "dc7ff86b606641e480c7d154b5f467e1f8c895f85733c73ba47a259a66ed187b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/input_image_tensor_uint8_meta.json?generation=1782184026203211"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_input_image_tensor_unsupported_meta_json",
        sha256 = "443d436c2068df8201b9822c35e724acfd8004a788d388e7d74c38a2425c55df",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/input_image_tensor_unsupported_meta.json?generation=1782184033189907"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_input_text_tensor_default_meta_json",
        sha256 = "9723e59960b0e6ca60d120494c32e798b054ea6e5a441b359c84f759bd2b3a36",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/input_text_tensor_default_meta.json?generation=1782184040516436"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_input_text_tensor_meta_json",
        sha256 = "c6782f676220e2cc89b70bacccb649fc848c18e33bedc449bf49f5d839b3cc6c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/input_text_tensor_meta.json?generation=1782184047612527"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobile_ica_8bit-with-custom-metadata_tflite",
        sha256 = "31f34f0dd0dc39e69e9c3deb1e3f3278febeb82ecf57c235834348a75df8fb51",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobile_ica_8bit-with-custom-metadata.tflite?generation=1782184069351812"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobile_ica_8bit-with-large-min-parser-version_tflite",
        sha256 = "53d0ea047682539964820fcfc5dc81f4928957470f453f2065f4c2ab87406803",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobile_ica_8bit-with-large-min-parser-version.tflite?generation=1782184076737078"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobile_ica_8bit-with-metadata_tflite",
        sha256 = "4afa3970d3efd6726d147d505e28c7ff1e4fe1c24be7bcda6b5429eb099777a5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobile_ica_8bit-with-metadata.tflite?generation=1782184084142567"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobile_ica_8bit-with-unsupported-metadata-version_tflite",
        sha256 = "5ea0341c481367df51741d7aa2fab4e3ba59f67ab366b18f6dcd50cb859ed548",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobile_ica_8bit-with-unsupported-metadata-version.tflite?generation=1782184091523497"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobile_ica_8bit-without-model-metadata_tflite",
        sha256 = "407d7b11da4b9e3f56f0cff7075e86a3d70813c74a15cf11975176912c65cbde",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobile_ica_8bit-without-model-metadata.tflite?generation=1782184098793829"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobile_object_classifier_v0_2_3-metadata-no-name_tflite",
        sha256 = "27fdb2dce68b8bd9a0f16583eefc4df13605808c1417cec268d1e838920c1a81",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobile_object_classifier_v0_2_3-metadata-no-name.tflite?generation=1782184106077245"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobilenet_v1_0_25_224_1_default_1_tflite",
        sha256 = "446ec673881cd46371a8726075b714194ada39d144762260cb76d15318597df7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobilenet_v1_0.25_224_1_default_1.tflite?generation=1782184120624370"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobilenet_v2_1_0_224_json",
        sha256 = "94613ea9539a20a3352604004be6d4d64d4d76250bc9042fcd8685c9a8498517",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobilenet_v2_1.0_224.json?generation=1782184127840999"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobilenet_v2_1_0_224_quant_json",
        sha256 = "3703eadcf838b65bbc2b2aa11dbb1f1bc654c7a09a7aba5ca75a26096484a8ac",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobilenet_v2_1.0_224_quant.json?generation=1782184134939972"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobilenet_v2_1_0_224_quant_tflite",
        sha256 = "f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobilenet_v2_1.0_224_quant.tflite?generation=1782184142444279"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobilenet_v2_1_0_224_quant_without_metadata_tflite",
        sha256 = "f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobilenet_v2_1.0_224_quant_without_metadata.tflite?generation=1782184150124935"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_mobilenet_v2_1_0_224_without_metadata_tflite",
        sha256 = "9f3bc29e38e90842a852bfed957dbf5e36f2d97a91dd17736b1e5c0aca8d3303",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/mobilenet_v2_1.0_224_without_metadata.tflite?generation=1782184157622927"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_movie_review_json",
        sha256 = "c09b88af05844cad5133b49744fed3a0bd514d4a1c75b9d2f23e9a40bd7bc04e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/movie_review.json?generation=1782184164531426"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_movie_review_tflite",
        sha256 = "3935ee73b13d435327d05af4d6f37dc3c146e117e1c3d572ae4d2ae0f5f412fe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/movie_review.tflite?generation=1782184171448334"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_score_calibration_csv",
        sha256 = "3ff4962162387ab8851940d2f063ce2b3a4734a8893c007a3c92d11170b020c3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/score_calibration.csv?generation=1782184193290242"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_score_calibration_file_meta_json",
        sha256 = "6a3c305620371f662419a496f75be5a10caebca7803b1e99d8d5d22ba51cda94",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/score_calibration_file_meta.json?generation=1782184207610266"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_score_calibration_tensor_meta_json",
        sha256 = "24cbde7f76dd6a09a55d07f30493c2f254d61154eb2e8d18ed947ff56781186d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/score_calibration_tensor_meta.json?generation=1782184214801139"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_score_thresholding_meta_json",
        sha256 = "7bb74f21c2d7f0237675ed7c09d7b7afd3507c8373f51dc75fa0507852f6ee19",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/score_thresholding_meta.json?generation=1782184221981312"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_segmentation_mask_meta_json",
        sha256 = "4294d53b309c1fbe38a5184de4057576c3dec14e07d16491f1dd459ac9116ab3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/segmentation_mask_meta.json?generation=1782184229223357"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_sentence_piece_tokenizer_meta_json",
        sha256 = "416bfe231710502e4a93e1b1950c0c6e5db49cffb256d241ef3d3f2d0d57718b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/sentence_piece_tokenizer_meta.json?generation=1782184243408688"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_ssd_mobilenet_v1_no_metadata_json",
        sha256 = "ae5a5971a1c3df705307448ef97c854d846b7e6f2183fb51015bd5af5d7deb0f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/ssd_mobilenet_v1_no_metadata.json?generation=1782184250445533"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_ssd_mobilenet_v1_no_metadata_tflite",
        sha256 = "e4b118e5e4531945de2e659742c7c590f7536f8d0ed26d135abcfe83b4779d13",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/ssd_mobilenet_v1_no_metadata.tflite?generation=1782184257880479"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_metadata_tensor_group_meta_json",
        sha256 = "eea454ae15b0c4f7e1f84aad9700bc936627fe22a085d335a40269740bc33c69",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/metadata/tensor_group_meta.json?generation=1782184264972521"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_30k-clean_model",
        sha256 = "fefb02b667a6c5c2fe27602d28e5fb3428f66ab89c7d6f388e7c8d44a02d0336",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/30k-clean.model?generation=1782184272381218"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_albert_with_metadata_tflite",
        sha256 = "6012e264092d40a2e14f634579b95c6fa9938d7995de810e26fcec65cbcd6442",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/albert_with_metadata.tflite?generation=1782184280862328"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_average_word_classifier_tflite",
        sha256 = "13bf6f7f35964f1e85d6cc762ee7b1952009b532b233baa5bdb4bf7441097f34",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/average_word_classifier.tflite?generation=1782184288208679"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_bert_text_classifier_tflite",
        sha256 = "1e5a550c09bff0a13e61858bcfac7654d7fcc6d42106b4f15e11117695069600",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/bert_text_classifier.tflite?generation=1782184296163140"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_dynamic_input_classifier_tflite",
        sha256 = "c5499daf5773cef89ce984df329c6324194a83bea7c7cf83159bf660a58de85c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/dynamic_input_classifier.tflite?generation=1782184303674174"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_falcon_rw_1b_test_weight_pt",
        sha256 = "62972530d362e881747f0f309573f32421a13b787603ab89874a23f4a5d44f44",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/falcon_rw_1b_test_weight.pt?generation=1782184318092936"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_gecko_task",
        sha256 = "165aded023aa067d8c29dae73eece2bbdec49bcb48e83c69083b83ff3928bc9c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/gecko.task?generation=1782184327811387"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_language_detector_tflite",
        sha256 = "5f64d821110dd2a3280546e8cd59dff09547e25d5f5c9711ec3f03416414dbb2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/language_detector.tflite?generation=1782184334735649"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_mobilebert_embedding_with_metadata_tflite",
        sha256 = "fa47142dcc6f446168bc672f2df9605b6da5d0c0d6264e9be62870282365b95c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/mobilebert_embedding_with_metadata.tflite?generation=1782184342485189"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_mobilebert_with_metadata_tflite",
        sha256 = "5984e86eb5d4cb95f004ff78e6f44d5f59b17120575c6313955d95afbb843ca3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/mobilebert_with_metadata.tflite?generation=1782184358927285"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_regex_one_embedding_with_metadata_tflite",
        sha256 = "b8f5d6d090c2c73984b2b92cd2fda27e5630562741a93d127b9a744d60505bc0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/regex_one_embedding_with_metadata.tflite?generation=1782184366258926"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_stablelm_3b_4e1t_test_weight_safetensors",
        sha256 = "c732deb063697cb46ad55013ed87372d57fd22b9e1cdf913a5e563601f50b7ec",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/stablelm_3b_4e1t_test_weight.safetensors?generation=1782184373558003"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_test_model_text_classifier_bool_output_tflite",
        sha256 = "09877ac6d718d78da6380e21fe8179854909d116632d6d770c12f8a51792e310",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/test_model_text_classifier_bool_output.tflite?generation=1782184380916725"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_test_model_text_classifier_with_regex_tokenizer_tflite",
        sha256 = "cb12618d084b813cb7b90ceb39c9fe4b18dae4de9880b912cdcd4b577cd65b4f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/test_model_text_classifier_with_regex_tokenizer.tflite?generation=1782184388044917"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_text_universal_sentence_encoder_qa_with_metadata_tflite",
        sha256 = "82c2d0450aa458adbec2f78eff33cfbf2a41b606b44246726ab67373926e32bc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/text/universal_sentence_encoder_qa_with_metadata.tflite?generation=1782184395846019"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_burger_jpg",
        sha256 = "97c15bbbf3cf3615063b1031c85d669de55839f59262bbe145d15ca75b36ecbf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/burger.jpg?generation=1782184424868164"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_burger_crop_jpg",
        sha256 = "8f58de573f0bf59a49c3d86cfabb9ad4061481f574aa049177e8da3963dddc50",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/burger_crop.jpg?generation=1782184432033088"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_burger_rotated_jpg",
        sha256 = "b7bb5e59ef778f3ce6b3e616c511908a53d513b83a56aae58b7453e14b0a4b2a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/burger_rotated.jpg?generation=1782184439219970"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cat_jpg",
        sha256 = "2533197401eebe9410ea4d063f86c43fbd2666f3e8165a38aca155c0d09c21be",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cat.jpg?generation=1782184446508641"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cat_large_jpg",
        sha256 = "f5e8996df94e2257cd92838954f57ac5e07bef1238228e518c893f0878511f96",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cat_large.jpg?generation=1782184453859664"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cat_large_mask_png",
        sha256 = "16b6398efc3835403e2d20c101014ba47f47595ec51f2ea0da8bf402d1019c37",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cat_large_mask.png?generation=1782184461159557"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cat_mask_jpg",
        sha256 = "bae065a685f2d32f1856151b5181671aa4d09925b55766935a30bbc8dafadcd0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cat_mask.jpg?generation=1782184468570584"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cat_rotated_jpg",
        sha256 = "b78cee5ad14c9f36b1c25d103db371d81ca74d99030063c46a38e80bb8f38649",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cat_rotated.jpg?generation=1782184476724651"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cat_rotated_mask_jpg",
        sha256 = "f336973e7621d602f2ebc9a6ab1c62d8502272d391713f369d3b99541afda861",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cat_rotated_mask.jpg?generation=1782184483853970"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cats_and_dogs_jpg",
        sha256 = "a2eaa7ad3a1aae4e623dd362a5f737e8a88d122597ecd1a02b3e1444db56df9c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cats_and_dogs.jpg?generation=1782184491144775"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cats_and_dogs_mask_dog1_png",
        sha256 = "2ab37d56ba1e46e70b3ddbfe35dac51b18b597b76904c68d7d34c7c74c677d4c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cats_and_dogs_mask_dog1.png?generation=1782184498359575"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cats_and_dogs_mask_dog2_png",
        sha256 = "2010850e2dd7f520fe53b9086d70913b6fb53b178cae15a373e5ee7ffb46824a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cats_and_dogs_mask_dog2.png?generation=1782184505567649"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cats_and_dogs_no_resizing_jpg",
        sha256 = "9d55933ed66bcdc63cd6509ee2518d7eed75d12db609238387ee4cc50b173e58",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cats_and_dogs_no_resizing.jpg?generation=1782184512849979"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_cats_and_dogs_rotated_jpg",
        sha256 = "5384926d16ddd8802555ae3108deedefb42a2ea78d99e5ad0933c5e11f43244a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/cats_and_dogs_rotated.jpg?generation=1782184519850110"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_coco_efficientdet_lite0_v1_1_0_quant_2021_09_06_tflite",
        sha256 = "dee1b4af055a644804d5594442300ecc9e4f7080c25b7c044c98f527eeabb6cf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/coco_efficientdet_lite0_v1_1.0_quant_2021_09_06.tflite?generation=1782184527668231"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_coco_ssd_mobilenet_v1_1_0_quant_2018_06_29_tflite",
        sha256 = "61d598093ed03ed41aa47c3a39a28ac01e960d6a810a5419b9a5016a1e9c469b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite?generation=1782184535239215"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_coco_ssd_mobilenet_v1_1_0_quant_2018_06_29_with_dummy_score_calibration_tflite",
        sha256 = "81b2681e3631c3813769396ff914a8f333b191fefcd8c61297fd165bc81e7e19",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_with_dummy_score_calibration.tflite?generation=1782229663481375"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_conv2d_input_channel_1_tflite",
        sha256 = "ccb667092f3aed3a35a57fb3478fecc0c8f6360dbf477a9db9c24e5b3ec4273e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/conv2d_input_channel_1.tflite?generation=1782184550465416"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_deeplabv3_tflite",
        sha256 = "5faed2c653905d3e22a8f6f29ee198da84e9b0e7936a207bf431f17f6b4d87ff",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/deeplabv3.tflite?generation=1782184557689389"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_dense_tflite",
        sha256 = "6795e7c3a263f44e97be048a5e1166e0921b453bfbaf037f4f69ac5c059ee945",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/dense.tflite?generation=1782184564772363"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_efficientdet_lite0_fp16_no_nms_tflite",
        sha256 = "237a58389081333e5cf4154e42b593ce7dd357445536fcaf4ca5bc51c2c50f1c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/efficientdet_lite0_fp16_no_nms.tflite?generation=1782184572419696"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_detection_full_range_tflite",
        sha256 = "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_detection_full_range.tflite?generation=1782184630620789"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_detection_full_range_sparse_tflite",
        sha256 = "2c3728e6da56f21e21a320433396fb06d40d9088f2247c05e5635a688d45dfe1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_detection_full_range_sparse.tflite?generation=1782184637874534"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_detection_full_range_sparse_with_metadata_tflite",
        sha256 = "0a058d9248f61fa8c902e12307c999c3b0312ce87e070600444c6f2f6ae727b9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_detection_full_range_sparse_with_metadata.tflite?generation=1782229498622025"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_detection_full_range_with_metadata_tflite",
        sha256 = "f5fd43a368d0eab9873f021eb741223a3e015759a46240a8dddf27af2def3f3e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_detection_full_range_with_metadata.tflite?generation=1782229502308028"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_detection_short_range_tflite",
        sha256 = "bbff11cebd1eb27a1e004cae0b0e63ec8c551cbf34a4451148b4908b8db3eca8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_detection_short_range.tflite?generation=1782184645070605"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_detection_short_range_with_metadata_tflite",
        sha256 = "42012b078cc201e994674ac863862577279644f3ba405729ccd964f03e9f12d5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_detection_short_range_with_metadata.tflite?generation=1782229506068834"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_landmark_tflite",
        sha256 = "1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_landmark.tflite?generation=1782184659286057"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_landmark_with_attention_tflite",
        sha256 = "e06a804e0144f9929eda782122916b35d60c697c3c9344013ca2bbe76a6ce2b4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_landmark_with_attention.tflite?generation=1782184666739724"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_landmarker_task",
        sha256 = "7cf2bbf1842c429e9defee38e7f1c4238978d8a6faf2da145bb19846f86bd2f4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_landmarker.task?generation=1782184673866996"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_landmarker_v2_task",
        sha256 = "af23fc7c1ff21d034deaa2b7fc1d56bb670ce69a4cbdc9579b6f1afd680835f4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_landmarker_v2.task?generation=1782184681283568"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_landmarker_v2_with_blendshapes_task",
        sha256 = "b261925d4aad812b47a0e8d58c1baa1223270a5d1f663d78338bc881c003879d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_landmarker_v2_with_blendshapes.task?generation=1782184688504388"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_face_landmarker_with_blendshapes_task",
        sha256 = "b44e4cae6f5822456d60f33e7c852640d78c7e342aee7eacc22589451a0b9dc2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/face_landmarker_with_blendshapes.task?generation=1782184695653228"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_facemesh2_lite_iris_faceflag_2023_02_14_tflite",
        sha256 = "bc5ee5de06d8c3a5465c3155227615b164480a52105a2b3df5748250ab4d914f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/facemesh2_lite_iris_faceflag_2023_02_14.tflite?generation=1782184703144548"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_fist_jpg",
        sha256 = "43fa1cabf3f90d574accc9a56986e2ee48638ce59fc65af1846487f73bb2ef24",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/fist.jpg?generation=1782184710240231"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_fist_png",
        sha256 = "4397b3d3f590c88a8de7d21c08d73a0df4a97fd93f92cbd086eef37fd246daaa",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/fist.png?generation=1782184717474049"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_gesture_recognizer_task",
        sha256 = "d48562f535fd4ecd3cfea739d9663dd818eeaf6a8afb1b5e6f8f4747661f73d9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/gesture_recognizer.task?generation=1782184732258353"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_hair_segmentation_tflite",
        sha256 = "7cbddcfe6f6e10c3e0a509eb2e14225fda5c0de6c35e2e8c6ca8e3971988fc17",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/hair_segmentation.tflite?generation=1782184739420340"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_hand_detector_tflite",
        sha256 = "1b14e9422c6ad006cde6581a46c8b90dd573c07ab7f3934b5589e7cea3f89a54",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/hand_detector.tflite?generation=1782184746800341"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_hand_landmark_full_tflite",
        sha256 = "11c272b891e1a99ab034208e23937a8008388cf11ed2a9d776ed3d01d0ba00e3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/hand_landmark_full.tflite?generation=1782184776133771"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_hand_landmark_lite_tflite",
        sha256 = "048edd3645c9bf7397d19a9f6e3a42957d6e414c9bea6598030a2e9b624156e6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/hand_landmark_lite.tflite?generation=1782184783517240"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_hand_landmarker_task",
        sha256 = "32d1eab97e80a9a20edb29231e15301ce65abfd0fa9d41cf1757e0ecc8078a4e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/hand_landmarker.task?generation=1782184791153758"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_hand_landmarks_detector_tflite",
        sha256 = "11c272b891e1a99ab034208e23937a8008388cf11ed2a9d776ed3d01d0ba00e3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/hand_landmarks_detector.tflite?generation=1782184798773030"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_handrecrop_2020_07_21_v0_f16_tflite",
        sha256 = "d40b15e15f93f6c909a3cfb881ce16c9ff9aa6d57417a0c906a6624f1f60b60c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/handrecrop_2020_07_21_v0.f16.tflite?generation=1782184813360376"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_holistic_landmarker_task",
        sha256 = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/holistic_landmarker.task?generation=1782184828108467"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_left_hands_jpg",
        sha256 = "240c082e80128ff1ca8a83ce645e2ba4d8bc30f0967b7991cf5fa375bab489e1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/left_hands.jpg?generation=1782184842333929"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_left_hands_rotated_jpg",
        sha256 = "b3bdf692f0d54b86c8b67e6d1286dd0078fbe6e9dfcd507b187e3bd8b398c0f9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/left_hands_rotated.jpg?generation=1782184849517264"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_leopard_jpg",
        sha256 = "d66fda0aa655f87c9fe87965a642e7b33ec990a3d9ed5812f1e5513da9d7d744",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/leopard.jpg?generation=1782184856570155"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_leopard_bg_removal_result_png",
        sha256 = "afd33f2058fd58d189cda86ec931647741a6139970c9bcbc637cdd151ec657c5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/leopard_bg_removal_result.png?generation=1782184863592377"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_leopard_bg_removal_result_512x512_png",
        sha256 = "30be22e89fdd1d7b985294498ec67509b0caa1ca941fe291fa25f43a3873e4dd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/leopard_bg_removal_result_512x512.png?generation=1782184870806813"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_living_room_jpg",
        sha256 = "8d74535dfe58e7d62dee99df5ab7741ad373a456797cf4d99048dfd17ccb0d6c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/living_room.jpg?generation=1782184877747422"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_magic_touch_tflite",
        sha256 = "e24338a717c1b7ad8d159666677ef400babb7f33b8ad60c4d96db4ecf694cd25",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/magic_touch.tflite?generation=1782184885441763"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_male_full_height_hands_jpg",
        sha256 = "8a7fe5be8b90d6078b09913ca28f7e5d342f8d3cde856ab4e3327d2970b887f8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/male_full_height_hands.jpg?generation=1782184892683350"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobile_bg_removal_tflite",
        sha256 = "f85797391cd1ef03988441710781342a77a980665965771fba603e5aee940ee8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobile_bg_removal.tflite?generation=1782184907970982"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobile_raid_det_nms_max_detections_40_max_labels_per_box_5_norm_coord_tflite",
        sha256 = "7df0e6fa124c6f30e5c5661f244e6c98bb8470ce2b487597541996b610c7cc87",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobile_raid_det_nms_max_detections_40_max_labels_per_box_5_norm_coord.tflite?generation=1782229667615941"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobile_raid_one_stage_v2_1_uint8_tflite",
        sha256 = "2e397b750d8f270e3f41731c1ec1f5b7811f93bc3a39fb81a4c47dd5e9055915",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobile_raid_one_stage_v2_1_uint8.tflite?generation=1782184923293415"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenet_v1_0_25_192_quantized_1_default_1_tflite",
        sha256 = "f80999b6324c6f101300c3ee38fbe7e11e74a743b5e0be7350602087fe7430a3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenet_v1_0.25_192_quantized_1_default_1.tflite?generation=1782184930388882"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenet_v1_0_25_224_1_default_1_tflite",
        sha256 = "446ec673881cd46371a8726075b714194ada39d144762260cb76d15318597df7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenet_v1_0.25_224_1_default_1.tflite?generation=1782184937833823"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenet_v1_0_25_224_1_metadata_1_tflite",
        sha256 = "348cc1221740b9fe1f609c964eff5bf09650bda76341c30aa27800b3da6171f4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenet_v1_0.25_224_1_metadata_1.tflite?generation=1782184945157232"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenet_v1_0_25_224_quant_tflite",
        sha256 = "e480eb15572f86d3d5f1be6e83e35b3c7d509ab2bcec353707d1f614e14edca2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenet_v1_0.25_224_quant.tflite?generation=1782184952510639"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenet_v1_0_25_224_quant_with_dummy_score_calibration_tflite",
        sha256 = "1fc6578a8f85f1f0454af6d908fba897fe17500c921e4d79434395abfb0e92f1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenet_v1_0.25_224_quant_with_dummy_score_calibration.tflite?generation=1782229671163207"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenet_v1_0_25_224_quant_without_subgraph_metadata_tflite",
        sha256 = "78f8b9bb5c873d3ad53ffc03b27651213016e45b6a2df42010c93191293bf694",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenet_v1_0.25_224_quant_without_subgraph_metadata.tflite?generation=1782229674549601"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenet_v2_1_0_224_tflite",
        sha256 = "ff5cb7f9e62c92ebdad971f8a98aa6b3106d82a64587a7787c6a385c9e791339",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenet_v2_1.0_224.tflite?generation=1782184975266616"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenet_v3_small_100_224_embedder_tflite",
        sha256 = "f7b9a563cb803bdcba76e8c7e82abde06f5c7a8e67b5e54e43e23095dfe79a78",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenet_v3_small_100_224_embedder.tflite?generation=1782184982945130"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mobilenetsweep_dptrigmqn384_unit_384_384_fp16quant_fp32input_opt_tflite",
        sha256 = "3c4c7e36b35fc903ecfb51b351b4849b23c57cc18d1416cf6cabaa1522d84760",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mobilenetsweep_dptrigmqn384_unit_384_384_fp16quant_fp32input_opt.tflite?generation=1782229678226165"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mozart_square_jpg",
        sha256 = "4feb4dadc5d6f853ade57b8c9d4c9a1f5ececd6469616c8e505f9a14823392b6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mozart_square.jpg?generation=1782184997394347"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mraid_v25_max_boxes_40_max_classes_per_box_5_tflite",
        sha256 = "1c4667b1a3caf90cc6971517df2525c0f8b6807d6598cf4554a9ca224faf42c5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mraid_v25_max_boxes_40_max_classes_per_box_5.tflite?generation=1782185004905212"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_mraid_v2_2_multiclass_nms_200_40_qat_tflite",
        sha256 = "1507a03e3c0e2567d2384b159dc17e21fb67f86c6390b10aab5d74275640ebda",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/mraid_v2_2_multiclass_nms_200_40_qat.tflite?generation=1782185012432660"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_multi_objects_jpg",
        sha256 = "ada6e36b40519cf0a4fbdf1b535de7fa7d0c472f2c0a08ada8ee5728e16c0c68",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/multi_objects.jpg?generation=1782185019846965"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_multi_objects_rotated_jpg",
        sha256 = "175f6c572ffbab6554e382fd5056d09720eef931ccc4ed79481bdc47a8443911",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/multi_objects_rotated.jpg?generation=1782185026792297"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_ocr_text_jpg",
        sha256 = "88052e93aa910330433741f5cef140f8f9ec463230a332aef7038b5457b06482",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/ocr_text.jpg?generation=1782185034263502"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_palm_detection_full_tflite",
        sha256 = "1b14e9422c6ad006cde6581a46c8b90dd573c07ab7f3934b5589e7cea3f89a54",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/palm_detection_full.tflite?generation=1782185042117966"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_penguins_large_jpg",
        sha256 = "3a7a74bf946b3e2b53a3953516a552df854b2854c91b3372d2d6343497ca2160",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/penguins_large.jpg?generation=1782185050235164"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_penguins_large_mask_png",
        sha256 = "8f78486266dabb1a3f28bf52750c0d005f96233fe505d5e8dcba02c6ee3a13cb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/penguins_large_mask.png?generation=1782185057357526"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_penguins_small_jpg",
        sha256 = "708ca356d8be4fbf5b76d4f2fcd094e97122cc24934cfcca22ac3ab0f13c4632",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/penguins_small.jpg?generation=1782185064460054"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_penguins_small_mask_png",
        sha256 = "65523dd7ed468ee4be3cd0cfed5badcfa41eaa5cd06444c9ab9b71b2d5951abe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/penguins_small_mask.png?generation=1782185071709378"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_pointing_up_jpg",
        sha256 = "ecf8ca2611d08fa25948a4fc10710af9120e88243a54da6356bacea17ff3e36e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/pointing_up.jpg?generation=1782185079090086"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_pointing_up_rotated_jpg",
        sha256 = "50ff66f50281207072a038e5bb6648c43f4aacbfb8204a4d2591868756aaeff1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/pointing_up_rotated.jpg?generation=1782185093709880"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_portrait_jpg",
        sha256 = "a6f11efaa834706db23f275b6115058fa87fc7f14362681e6abe14e82749de3e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/portrait.jpg?generation=1782185108020964"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_portrait_hair_expected_mask_jpg",
        sha256 = "d9ffc4f2ed0ee2d551d9239942e4dfceebf0c33a56858c84410f32ea4f0c1b2c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/portrait_hair_expected_mask.jpg?generation=1782185158407771"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_portrait_rotated_jpg",
        sha256 = "f91ca0e4f827b06e9ac037cf58d95f1f3ffbe34238119b7d47eda35456007f33",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/portrait_rotated.jpg?generation=1782185165784604"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_portrait_selfie_segmentation_expected_category_mask_jpg",
        sha256 = "1400c6fccf3805bfd1644d7ed9be98dfa4f900e1720838c566963f8d9f10f5d0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/portrait_selfie_segmentation_expected_category_mask.jpg?generation=1782185179990949"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_portrait_selfie_segmentation_expected_confidence_mask_jpg",
        sha256 = "25b723e90608edaf6ed92f382da703dc904a59c87525b6d271e60d9eed7a90e9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/portrait_selfie_segmentation_expected_confidence_mask.jpg?generation=1782185187076662"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_portrait_selfie_segmentation_landscape_expected_category_mask_jpg",
        sha256 = "a208aeeeb615fd40046d883e2c7982458e1b12edd6526e88c305c4053b0a9399",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/portrait_selfie_segmentation_landscape_expected_category_mask.jpg?generation=1782229681800895"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_portrait_small_jpg",
        sha256 = "873a1a5e4cc86c040101362c5dea6a71cf524563b0700640175e5c3763a4246a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/portrait_small.jpg?generation=1782185201319602"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_pose_jpg",
        sha256 = "c8a830ed683c0276d713dd5aeda28f415f10cd6291972084a40d0d8b934ed62b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/pose.jpg?generation=1782185208547605"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_pose_detection_tflite",
        sha256 = "9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/pose_detection.tflite?generation=1782185215901406"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_pose_landmark_lite_tflite",
        sha256 = "1150dc68a713b80660b90ef46ce4e85c1c781bb88b6e3512cc64e6a685ba5588",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/pose_landmark_lite.tflite?generation=1782185237821690"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_pose_landmarker_task",
        sha256 = "fb9cc326c88fc2a4d9a6d355c28520d5deacfbaa375b56243b0141b546080596",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/pose_landmarker.task?generation=1782185245556889"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_pose_segmentation_mask_golden_png",
        sha256 = "62ee418e18f317327572da5fcc988af703eb31e6f0b9e0bf3d55e6f4797d6953",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/pose_segmentation_mask_golden.png?generation=1782185259974348"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_ptm_512_hdt_ptm_woid_tflite",
        sha256 = "2baa1c9783d03dd26f91e3c49efbcab11dd1361ff80e40e7209e81f84f281b6a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/ptm_512_hdt_ptm_woid.tflite?generation=1782185267734223"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_right_hands_jpg",
        sha256 = "4b5134daa4cb60465535239535f9f74c2842aba3aa5fd30bf04ef5678f93d87f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/right_hands.jpg?generation=1782185275057115"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_right_hands_rotated_jpg",
        sha256 = "8609c6202bca43a99bbf23fa8e687e49fa525e89481152e4c0987f46d60d7931",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/right_hands_rotated.jpg?generation=1782185282074214"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_segmentation_golden_rotation0_png",
        sha256 = "9ee993919b753118928ba2d14f7c5c83a6cfc23355e6943dac4ad81eedd73069",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/segmentation_golden_rotation0.png?generation=1782185289166865"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_segmentation_input_rotation0_jpg",
        sha256 = "5bf58d8af1f1c33224f3f3bc0ce451c8daf0739cc15a86d59d8c3bf2879afb97",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/segmentation_input_rotation0.jpg?generation=1782185296406038"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_segmentation_input_rotation90_jpg",
        sha256 = "5289212cc399dd1d3e0589f36b46f65891b21606ab5a73f677a1baee3890fbe8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/segmentation_input_rotation90.jpg?generation=1782185303688934"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_selfie_segm_128_128_3_tflite",
        sha256 = "8322982866488b063af6531b1d16ac27c7bf404135b7905f20aaf5e6af7aa45b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/selfie_segm_128_128_3.tflite?generation=1782185311005260"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_selfie_segm_128_128_3_expected_mask_jpg",
        sha256 = "1a2a068287d8bcd4184492485b3dbb95a09b763f4653fd729d14a836147eb383",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/selfie_segm_128_128_3_expected_mask.jpg?generation=1782185318335658"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_selfie_segm_144_256_3_tflite",
        sha256 = "f16a9551a408edeadd53f70d1d2911fc20f9f9de7a394129a268ca9faa2d6a08",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/selfie_segm_144_256_3.tflite?generation=1782185325462430"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_selfie_segm_144_256_3_expected_mask_jpg",
        sha256 = "2de433b6e8adabec2aaf80135232db900903ead4f2811c0c9378a6792b2a68b5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/selfie_segm_144_256_3_expected_mask.jpg?generation=1782185332688182"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_selfie_segmentation_tflite",
        sha256 = "9ee168ec7c8f2a16c56fe8e1cfbc514974cbbb7e434051b455635f1bd1462f5c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/selfie_segmentation.tflite?generation=1782185339780682"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_selfie_segmentation_landscape_tflite",
        sha256 = "a77d03f4659b9f6b6c1f5106947bf40e99d7655094b6527f214ea7d451106edd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/selfie_segmentation_landscape.tflite?generation=1782185347043395"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_thumb_up_jpg",
        sha256 = "5d673c081ab13b8a1812269ff57047066f9c33c07db5f4178089e8cb3fdc0291",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/thumb_up.jpg?generation=1782185354353621"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_thumb_up_rgba_png",
        sha256 = "1f1fa9a627193b171cdb37000daf809f35a3adb0321e98275faa588d0b48e701",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/thumb_up_rgba.png?generation=1782185368672696"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_victory_jpg",
        sha256 = "84cb8853e3df614e0cb5c93a25e3e2f38ea5e4f92fd428ee7d867ed3479d5764",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/victory.jpg?generation=1782185383577587"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_vit_multiclass_256x256-2022_10_14-xenoformer_f32_tflite",
        sha256 = "768b7dd613c5b9f263b289cdbe1b9bc716f65e92c51e7ae57fae01f97f9658cd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/vit_multiclass_256x256-2022_10_14-xenoformer.f32.tflite?generation=1782185398213328"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_testdata_vision_vit_multiclass_512x512-2022_12_02_f32_tflite",
        sha256 = "7ac7f0a037cd451b9be8eb25da86339aaba54fa821a0bd44e18768866ed0205a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tasks/testdata/vision/vit_multiclass_512x512-2022_12_02.f32.tflite?generation=1782185405871755"],
    )

    http_file(
        name = "com_google_mediapipe_third_party_libc___shared_so",
        sha256 = "816d497229b6678db485b5dc16ae7d2ac63dc015691b1828bc35c4aa2ed6eed4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/third_party/libc++_shared.so?generation=1782183695934448"],
    )
