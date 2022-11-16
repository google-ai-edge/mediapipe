"""
External file definitions for MediaPipe.

This file is auto-generated.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

# buildifier: disable=unnamed-macro
def external_files():
    """External file definitions for MediaPipe."""

    http_file(
        name = "com_google_mediapipe_30k-clean_model",
        sha256 = "fefb02b667a6c5c2fe27602d28e5fb3428f66ab89c7d6f388e7c8d44a02d0336",
        urls = ["https://storage.googleapis.com/mediapipe-assets/30k-clean.model?generation=1663006350848402"],
    )

    http_file(
        name = "com_google_mediapipe_albert_with_metadata_tflite",
        sha256 = "6012e264092d40a2e14f634579b95c6fa9938d7995de810e26fcec65cbcd6442",
        urls = ["https://storage.googleapis.com/mediapipe-assets/albert_with_metadata.tflite?generation=1661875651648830"],
    )

    http_file(
        name = "com_google_mediapipe_associated_file_meta_json",
        sha256 = "5b2cba11ae893e1226af6570813955889e9f171d6d2c67b3e96ecb6b96d8c681",
        urls = ["https://storage.googleapis.com/mediapipe-assets/associated_file_meta.json?generation=1665422792304395"],
    )

    http_file(
        name = "com_google_mediapipe_bert_text_classifier_no_metadata_tflite",
        sha256 = "9b4554f6e28a72a3f40511964eed1ccf4e74cc074f81543cacca4faf169a173e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/bert_text_classifier_no_metadata.tflite?generation=1667948360250899"],
    )

    http_file(
        name = "com_google_mediapipe_bert_text_classifier_tflite",
        sha256 = "1e5a550c09bff0a13e61858bcfac7654d7fcc6d42106b4f15e11117695069600",
        urls = ["https://storage.googleapis.com/mediapipe-assets/bert_text_classifier.tflite?generation=1666144699858747"],
    )

    http_file(
        name = "com_google_mediapipe_bert_text_classifier_with_bert_tokenizer_json",
        sha256 = "49f148a13a4e3b486b1d3c2400e46e5ebd0d375674c0154278b835760e873a95",
        urls = ["https://storage.googleapis.com/mediapipe-assets/bert_text_classifier_with_bert_tokenizer.json?generation=1667948363241334"],
    )

    http_file(
        name = "com_google_mediapipe_bert_text_classifier_with_sentence_piece_json",
        sha256 = "113091f3892691de57e379387256b2ce0cc18a1b5185af866220a46da8221f26",
        urls = ["https://storage.googleapis.com/mediapipe-assets/bert_text_classifier_with_sentence_piece.json?generation=1667948366009530"],
    )

    http_file(
        name = "com_google_mediapipe_bert_tokenizer_meta_json",
        sha256 = "116d70c7c3ef413a8bff54ab758f9ed3d6e51fdc5621d8c920ad2f0035831804",
        urls = ["https://storage.googleapis.com/mediapipe-assets/bert_tokenizer_meta.json?generation=1667948368809108"],
    )

    http_file(
        name = "com_google_mediapipe_bounding_box_tensor_meta_json",
        sha256 = "cc019cee86529955a24a3d43ca3d778fa366bcb90d67c8eaf55696789833841a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/bounding_box_tensor_meta.json?generation=1665422797529909"],
    )

    http_file(
        name = "com_google_mediapipe_BUILD",
        sha256 = "d2b2a8346202691d7f831887c84e9642e974f64ed67851d9a58cf15c94b1f6b3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/BUILD?generation=1661875663693976"],
    )

    http_file(
        name = "com_google_mediapipe_burger_crop_jpg",
        sha256 = "8f58de573f0bf59a49c3d86cfabb9ad4061481f574aa049177e8da3963dddc50",
        urls = ["https://storage.googleapis.com/mediapipe-assets/burger_crop.jpg?generation=1664184735043119"],
    )

    http_file(
        name = "com_google_mediapipe_burger_jpg",
        sha256 = "97c15bbbf3cf3615063b1031c85d669de55839f59262bbe145d15ca75b36ecbf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/burger.jpg?generation=1661875667922678"],
    )

    http_file(
        name = "com_google_mediapipe_burger_rotated_jpg",
        sha256 = "b7bb5e59ef778f3ce6b3e616c511908a53d513b83a56aae58b7453e14b0a4b2a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/burger_rotated.jpg?generation=1665065843774448"],
    )

    http_file(
        name = "com_google_mediapipe_canned_gesture_classifier_tflite",
        sha256 = "ee121d85979de1b86126faabb0a0f4d2e4039c3e33e2cd687db50571001b24d0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/canned_gesture_classifier.tflite?generation=1668550473107417"],
    )

    http_file(
        name = "com_google_mediapipe_cat_jpg",
        sha256 = "2533197401eebe9410ea4d063f86c43fbd2666f3e8165a38aca155c0d09c21be",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cat.jpg?generation=1661875672441459"],
    )

    http_file(
        name = "com_google_mediapipe_cat_mask_jpg",
        sha256 = "bae065a685f2d32f1856151b5181671aa4d09925b55766935a30bbc8dafadcd0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cat_mask.jpg?generation=1661875677203533"],
    )

    http_file(
        name = "com_google_mediapipe_cat_rotated_jpg",
        sha256 = "b78cee5ad14c9f36b1c25d103db371d81ca74d99030063c46a38e80bb8f38649",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cat_rotated.jpg?generation=1666304165042123"],
    )

    http_file(
        name = "com_google_mediapipe_cat_rotated_mask_jpg",
        sha256 = "f336973e7621d602f2ebc9a6ab1c62d8502272d391713f369d3b99541afda861",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cat_rotated_mask.jpg?generation=1666304167148173"],
    )

    http_file(
        name = "com_google_mediapipe_cats_and_dogs_jpg",
        sha256 = "a2eaa7ad3a1aae4e623dd362a5f737e8a88d122597ecd1a02b3e1444db56df9c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cats_and_dogs.jpg?generation=1661875684064150"],
    )

    http_file(
        name = "com_google_mediapipe_cats_and_dogs_no_resizing_jpg",
        sha256 = "9d55933ed66bcdc63cd6509ee2518d7eed75d12db609238387ee4cc50b173e58",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cats_and_dogs_no_resizing.jpg?generation=1661875687251296"],
    )

    http_file(
        name = "com_google_mediapipe_cats_and_dogs_rotated_jpg",
        sha256 = "5384926d16ddd8802555ae3108deedefb42a2ea78d99e5ad0933c5e11f43244a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cats_and_dogs_rotated.jpg?generation=1665609933260747"],
    )

    http_file(
        name = "com_google_mediapipe_classification_tensor_float_meta_json",
        sha256 = "1d10b1c9c87eabac330651136804074ddc134779e94a73cf783207c3aa2a5619",
        urls = ["https://storage.googleapis.com/mediapipe-assets/classification_tensor_float_meta.json?generation=1665422803073223"],
    )

    http_file(
        name = "com_google_mediapipe_classification_tensor_uint8_meta_json",
        sha256 = "74f4d64ee0017d11e0fdc975a88d974d73b72b889fd4d67992356052edde0f1e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/classification_tensor_uint8_meta.json?generation=1665422808178685"],
    )

    http_file(
        name = "com_google_mediapipe_classification_tensor_unsupported_meta_json",
        sha256 = "4810ad8a00f0078c6a693114d00f692aa70ff2d61030a6e516db1e654707e208",
        urls = ["https://storage.googleapis.com/mediapipe-assets/classification_tensor_unsupported_meta.json?generation=1665422813312699"],
    )

    http_file(
        name = "com_google_mediapipe_coco_efficientdet_lite0_v1_1_0_quant_2021_09_06_tflite",
        sha256 = "dee1b4af055a644804d5594442300ecc9e4f7080c25b7c044c98f527eeabb6cf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/coco_efficientdet_lite0_v1_1.0_quant_2021_09_06.tflite?generation=1661875692679200"],
    )

    http_file(
        name = "com_google_mediapipe_coco_ssd_mobilenet_v1_1_0_quant_2018_06_29_tflite",
        sha256 = "61d598093ed03ed41aa47c3a39a28ac01e960d6a810a5419b9a5016a1e9c469b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite?generation=1666144700870810"],
    )

    http_file(
        name = "com_google_mediapipe_coco_ssd_mobilenet_v1_1_0_quant_2018_06_29_with_dummy_score_calibration_tflite",
        sha256 = "81b2681e3631c3813769396ff914a8f333b191fefcd8c61297fd165bc81e7e19",
        urls = ["https://storage.googleapis.com/mediapipe-assets/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_with_dummy_score_calibration.tflite?generation=1662653237233967"],
    )

    http_file(
        name = "com_google_mediapipe_corrupted_mobilenet_v1_0_25_224_1_default_1_tflite",
        sha256 = "f0cbeb8061f4c693e20de779ce255af923508492e8a24f6db320845a52facb51",
        urls = ["https://storage.googleapis.com/mediapipe-assets/corrupted_mobilenet_v1_0.25_224_1_default_1.tflite?generation=1661875706780536"],
    )

    http_file(
        name = "com_google_mediapipe_deeplabv3_tflite",
        sha256 = "9711334db2b01d5894feb8ed0f5cb3e97d125b8d229f8d8692f625801818f5ef",
        urls = ["https://storage.googleapis.com/mediapipe-assets/deeplabv3.tflite?generation=1661875711618421"],
    )

    http_file(
        name = "com_google_mediapipe_dummy_gesture_recognizer_task",
        sha256 = "18e54586bda33300d459ca140cd045f6daf43d897224ba215a16db3423eae18e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/dummy_gesture_recognizer.task?generation=1665707319890725"],
    )

    http_file(
        name = "com_google_mediapipe_empty_vocab_for_regex_tokenizer_txt",
        sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        urls = ["https://storage.googleapis.com/mediapipe-assets/empty_vocab_for_regex_tokenizer.txt?generation=1661875714907539"],
    )

    http_file(
        name = "com_google_mediapipe_expected_left_down_hand_landmarks_prototxt",
        sha256 = "ae9cb01035f18b0023fc12256c048666da76b41b327cec09c2d2820054b1295f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_left_down_hand_landmarks.prototxt?generation=1661875720230540"],
    )

    http_file(
        name = "com_google_mediapipe_expected_left_down_hand_rotated_landmarks_prototxt",
        sha256 = "c4dfdcc2e4cd366eb5f8ad227be94049eb593e3a528564611094687912463687",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_left_down_hand_rotated_landmarks.prototxt?generation=1666629474155924"],
    )

    http_file(
        name = "com_google_mediapipe_expected_left_up_hand_landmarks_prototxt",
        sha256 = "1353ba617c4f048083618587cd23a8a22115f634521c153d4e1bd1ebd4f49dd7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_left_up_hand_landmarks.prototxt?generation=1661875726008879"],
    )

    http_file(
        name = "com_google_mediapipe_expected_left_up_hand_rotated_landmarks_prototxt",
        sha256 = "7fb2d33cf69d2da50952a45bad0c0618f30859e608958fee95948a6e0de63ccb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_left_up_hand_rotated_landmarks.prototxt?generation=1666629476401757"],
    )

    http_file(
        name = "com_google_mediapipe_expected_right_down_hand_landmarks_prototxt",
        sha256 = "f281b745175aaa7f458def6cf4c89521fb56302dd61a05642b3b4a4f237ffaa3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_right_down_hand_landmarks.prototxt?generation=1661875730821226"],
    )

    http_file(
        name = "com_google_mediapipe_expected_right_up_hand_landmarks_prototxt",
        sha256 = "174cf5f7c3ab547f0affb666ee7be933b0758c60fbfe7b7e93795c5082555592",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_right_up_hand_landmarks.prototxt?generation=1661875733440313"],
    )

    http_file(
        name = "com_google_mediapipe_external_file_txt",
        sha256 = "ae0666f161fed1a5dde998bbd0e140550d2da0db27db1d0e31e370f2bd366a57",
        urls = ["https://storage.googleapis.com/mediapipe-assets/external_file.txt?generation=1661875736240688"],
    )

    http_file(
        name = "com_google_mediapipe_face_detection_full_range_sparse_tflite",
        sha256 = "671dd2f9ed11a78436fc21cc42357a803dfc6f73e9fb86541be942d5716c2dce",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_detection_full_range_sparse.tflite?generation=1661875739104017"],
    )

    http_file(
        name = "com_google_mediapipe_face_detection_full_range_tflite",
        sha256 = "99bf9494d84f50acc6617d89873f71bf6635a841ea699c17cb3377f9507cfec3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite?generation=1661875742733283"],
    )

    http_file(
        name = "com_google_mediapipe_face_detection_short_range_tflite",
        sha256 = "3bc182eb9f33925d9e58b5c8d59308a760f4adea8f282370e428c51212c26633",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite?generation=1661875748538815"],
    )

    http_file(
        name = "com_google_mediapipe_face_landmark_tflite",
        sha256 = "c603fa6149219a3e9487dc9abd7a0c24474c77263273d24868378cdf40aa26d1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite?generation=1662063817995673"],
    )

    http_file(
        name = "com_google_mediapipe_face_landmark_with_attention_tflite",
        sha256 = "883b7411747bac657c30c462d305d312e9dec6adbf8b85e2f5d8d722fca9455d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_landmark_with_attention.tflite?generation=1661875751615925"],
    )

    http_file(
        name = "com_google_mediapipe_feature_tensor_meta_json",
        sha256 = "b2c30ddfd495956ce81085f8a143422f4310b002cfbf1c594ff2ee0576e29d6f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/feature_tensor_meta.json?generation=1665422818797346"],
    )

    http_file(
        name = "com_google_mediapipe_fist_jpg",
        sha256 = "43fa1cabf3f90d574accc9a56986e2ee48638ce59fc65af1846487f73bb2ef24",
        urls = ["https://storage.googleapis.com/mediapipe-assets/fist.jpg?generation=1666999359066679"],
    )

    http_file(
        name = "com_google_mediapipe_fist_landmarks_pbtxt",
        sha256 = "76d6489e6163211ce5e9080e51983165bb9b24ff50146cc7487bd629f011c598",
        urls = ["https://storage.googleapis.com/mediapipe-assets/fist_landmarks.pbtxt?generation=1666999360561864"],
    )

    http_file(
        name = "com_google_mediapipe_general_meta_json",
        sha256 = "b95363e4bae89b9c2af484498312aaad4efc7ff57c7eadcc4e5e7adca641445f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/general_meta.json?generation=1665422822603848"],
    )

    http_file(
        name = "com_google_mediapipe_gesture_embedder_tflite",
        sha256 = "927e4f6cbe6451da6b4fd1485e2576a6f8dbd95062666661cbd9dea893c41d01",
        urls = ["https://storage.googleapis.com/mediapipe-assets/gesture_embedder.tflite?generation=1668550476472972"],
    )

    http_file(
        name = "com_google_mediapipe_gesture_recognizer_task",
        sha256 = "a966b1d4e774e0423c19c8aa71f070e5a72fe7a03c2663dd2f3cb0b0095ee3e1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/gesture_recognizer.task?generation=1668100501451433"],
    )

    http_file(
        name = "com_google_mediapipe_golden_json_json",
        sha256 = "55c0c88748d099aa379930504df62c6c8f1d8874ea52d2f8a925f352c4c7f09c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/golden_json.json?generation=1664340169675228"],
    )

    http_file(
        name = "com_google_mediapipe_hair_segmentation_tflite",
        sha256 = "d2c940c4fd80edeaf38f5d7387d1b4235ee320ed120080df67c663e749e77633",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite?generation=1661875756623461"],
    )

    http_file(
        name = "com_google_mediapipe_hand_detector_result_one_hand_pbtxt",
        sha256 = "4b2deb84992bbfe68e3409d2b76914960d1c65aa6edd4524ff3455ca489df5f1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_detector_result_one_hand.pbtxt?generation=1662745351291628"],
    )

    http_file(
        name = "com_google_mediapipe_hand_detector_result_one_hand_rotated_pbtxt",
        sha256 = "555079c274ea91699757a0b9888c9993a8ab450069103b1bcd4ebb805a8e023c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_detector_result_one_hand_rotated.pbtxt?generation=1666629478777955"],
    )

    http_file(
        name = "com_google_mediapipe_hand_detector_result_two_hands_pbtxt",
        sha256 = "2589cb08b0ee027dc24649fe597adcfa2156a21d12ea2480f83832714ebdf95f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_detector_result_two_hands.pbtxt?generation=1662745353586157"],
    )

    http_file(
        name = "com_google_mediapipe_hand_landmarker_task",
        sha256 = "2ed44f10872e87a5834b9b1130fb9ada30e107af2c6fcc4562ad788aca4e7bc4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task?generation=1666153732577904"],
    )

    http_file(
        name = "com_google_mediapipe_hand_landmark_full_tflite",
        sha256 = "11c272b891e1a99ab034208e23937a8008388cf11ed2a9d776ed3d01d0ba00e3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite?generation=1661875760968579"],
    )

    http_file(
        name = "com_google_mediapipe_hand_landmark_lite_tflite",
        sha256 = "048edd3645c9bf7397d19a9f6e3a42957d6e414c9bea6598030a2e9b624156e6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite?generation=1661875766398729"],
    )

    http_file(
        name = "com_google_mediapipe_hand_landmark_tflite",
        sha256 = "bad88ac1fd144f034e00f075afcade4f3a21d0d09c41bee8dd50504dacd70efd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_landmark.tflite?generation=1666153735814956"],
    )

    http_file(
        name = "com_google_mediapipe_hand_recrop_tflite",
        sha256 = "67d996ce96f9d36fe17d2693022c6da93168026ab2f028f9e2365398d8ac7d5d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_recrop.tflite?generation=1661875770633070"],
    )

    http_file(
        name = "com_google_mediapipe_image_tensor_meta_json",
        sha256 = "aad86fde3defb379c82ff7ee48e50493a58529cdc0623cf0d7bf135c3577060e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/image_tensor_meta.json?generation=1665422826106636"],
    )

    http_file(
        name = "com_google_mediapipe_input_image_tensor_float_meta_json",
        sha256 = "426ecf5c3ace61db3936b950c3709daece15827ea21905ddbcdc81b1c6e70232",
        urls = ["https://storage.googleapis.com/mediapipe-assets/input_image_tensor_float_meta.json?generation=1665422829230563"],
    )

    http_file(
        name = "com_google_mediapipe_input_image_tensor_uint8_meta_json",
        sha256 = "dc7ff86b606641e480c7d154b5f467e1f8c895f85733c73ba47a259a66ed187b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/input_image_tensor_uint8_meta.json?generation=1665422832572887"],
    )

    http_file(
        name = "com_google_mediapipe_input_image_tensor_unsupported_meta_json",
        sha256 = "443d436c2068df8201b9822c35e724acfd8004a788d388e7d74c38a2425c55df",
        urls = ["https://storage.googleapis.com/mediapipe-assets/input_image_tensor_unsupported_meta.json?generation=1665422835757143"],
    )

    http_file(
        name = "com_google_mediapipe_input_text_tensor_default_meta_json",
        sha256 = "9723e59960b0e6ca60d120494c32e798b054ea6e5a441b359c84f759bd2b3a36",
        urls = ["https://storage.googleapis.com/mediapipe-assets/input_text_tensor_default_meta.json?generation=1667855382021347"],
    )

    http_file(
        name = "com_google_mediapipe_input_text_tensor_meta_json",
        sha256 = "c6782f676220e2cc89b70bacccb649fc848c18e33bedc449bf49f5d839b3cc6c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/input_text_tensor_meta.json?generation=1667855384891533"],
    )

    http_file(
        name = "com_google_mediapipe_iris_and_gaze_tflite",
        sha256 = "b6dcb860a92a3c7264a8e50786f46cecb529672cdafc17d39c78931257da661d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/iris_and_gaze.tflite?generation=1661875774291949"],
    )

    http_file(
        name = "com_google_mediapipe_iris_landmark_tflite",
        sha256 = "d1744d2a09c25f501d39eba4faff47e53ecca8852c5ce19bce8eeac39357521f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite?generation=1662065468653224"],
    )

    http_file(
        name = "com_google_mediapipe_knift_float_1k_tflite",
        sha256 = "5dbfa98c7a3caae97840576a278a1d1fe37c86bad4007d1acdffec094242837c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/knift_float_1k.tflite?generation=1661875777483362"],
    )

    http_file(
        name = "com_google_mediapipe_knift_float_400_tflite",
        sha256 = "3ee576050f3d5d45ea19a19dbd67267cb345b0348efde00952eddb8b7aabe2e5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/knift_float_400.tflite?generation=1661875782481333"],
    )

    http_file(
        name = "com_google_mediapipe_knift_float_tflite",
        sha256 = "40567854c2c1022c98cd2c55a7eef1c60999580ce67db118c1274000d0e22ace",
        urls = ["https://storage.googleapis.com/mediapipe-assets/knift_float.tflite?generation=1661875785348544"],
    )

    http_file(
        name = "com_google_mediapipe_knift_index_pb",
        sha256 = "2c2b57a846e0adbf1e3f25bd20c7878ac9399460a1ad5d8147e3231ace8eb3dc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/knift_index.pb?generation=1661875789855286"],
    )

    http_file(
        name = "com_google_mediapipe_knift_labelmap_txt",
        sha256 = "40f9f5bd76a8574478299af93fcab96f5cdc71273f4e20c5899c248a33970cff",
        urls = ["https://storage.googleapis.com/mediapipe-assets/knift_labelmap.txt?generation=1661875792821628"],
    )

    http_file(
        name = "com_google_mediapipe_labels_txt",
        sha256 = "536feacc519de3d418de26b2effb4d75694a8c4c0063e36499a46fa8061e2da9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/labels.txt?generation=1667892497527642"],
    )

    http_file(
        name = "com_google_mediapipe_left_hands_jpg",
        sha256 = "4b5134daa4cb60465535239535f9f74c2842aba3aa5fd30bf04ef5678f93d87f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/left_hands.jpg?generation=1661875796949017"],
    )

    http_file(
        name = "com_google_mediapipe_left_hands_rotated_jpg",
        sha256 = "8609c6202bca43a99bbf23fa8e687e49fa525e89481152e4c0987f46d60d7931",
        urls = ["https://storage.googleapis.com/mediapipe-assets/left_hands_rotated.jpg?generation=1666037068103465"],
    )

    http_file(
        name = "com_google_mediapipe_mobilebert_embedding_with_metadata_tflite",
        sha256 = "fa47142dcc6f446168bc672f2df9605b6da5d0c0d6264e9be62870282365b95c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilebert_embedding_with_metadata.tflite?generation=1664516086197724"],
    )

    http_file(
        name = "com_google_mediapipe_mobilebert_vocab_txt",
        sha256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilebert_vocab.txt?generation=1661875800701493"],
    )

    http_file(
        name = "com_google_mediapipe_mobilebert_with_metadata_tflite",
        sha256 = "5984e86eb5d4cb95f004ff78e6f44d5f59b17120575c6313955d95afbb843ca3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilebert_with_metadata.tflite?generation=1661875806733025"],
    )

    http_file(
        name = "com_google_mediapipe_mobile_ica_8bit-with-metadata_tflite",
        sha256 = "4afa3970d3efd6726d147d505e28c7ff1e4fe1c24be7bcda6b5429eb099777a5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_ica_8bit-with-metadata.tflite?generation=1661875810860490"],
    )

    http_file(
        name = "com_google_mediapipe_mobile_ica_8bit-without-model-metadata_tflite",
        sha256 = "407d7b11da4b9e3f56f0cff7075e86a3d70813c74a15cf11975176912c65cbde",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_ica_8bit-without-model-metadata.tflite?generation=1661875814428283"],
    )

    http_file(
        name = "com_google_mediapipe_mobile_ica_8bit-with-unsupported-metadata-version_tflite",
        sha256 = "5ea0341c481367df51741d7aa2fab4e3ba59f67ab366b18f6dcd50cb859ed548",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_ica_8bit-with-unsupported-metadata-version.tflite?generation=1661875819091013"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v1_0_25_192_quantized_1_default_1_tflite",
        sha256 = "f80999b6324c6f101300c3ee38fbe7e11e74a743b5e0be7350602087fe7430a3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v1_0.25_192_quantized_1_default_1.tflite?generation=1661875821863721"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v1_0_25_224_1_default_1_tflite",
        sha256 = "446ec673881cd46371a8726075b714194ada39d144762260cb76d15318597df7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v1_0.25_224_1_default_1.tflite?generation=1661875824782010"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v1_0_25_224_1_metadata_1_tflite",
        sha256 = "348cc1221740b9fe1f609c964eff5bf09650bda76341c30aa27800b3da6171f4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v1_0.25_224_1_metadata_1.tflite?generation=1661875828385370"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v1_0_25_224_quant_tflite",
        sha256 = "e480eb15572f86d3d5f1be6e83e35b3c7d509ab2bcec353707d1f614e14edca2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v1_0.25_224_quant.tflite?generation=1661875831485992"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v1_0_25_224_quant_with_dummy_score_calibration_tflite",
        sha256 = "1fc6578a8f85f1f0454af6d908fba897fe17500c921e4d79434395abfb0e92f1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v1_0.25_224_quant_with_dummy_score_calibration.tflite?generation=1662650659741978"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v1_0_25_224_quant_without_subgraph_metadata_tflite",
        sha256 = "78f8b9bb5c873d3ad53ffc03b27651213016e45b6a2df42010c93191293bf694",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v1_0.25_224_quant_without_subgraph_metadata.tflite?generation=1661875836078124"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v2_1_0_224_json",
        sha256 = "94613ea9539a20a3352604004be6d4d64d4d76250bc9042fcd8685c9a8498517",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v2_1.0_224.json?generation=1666633416316646"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v2_1_0_224_quant_json",
        sha256 = "3703eadcf838b65bbc2b2aa11dbb1f1bc654c7a09a7aba5ca75a26096484a8ac",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v2_1.0_224_quant.json?generation=1666633418665507"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v2_1_0_224_quant_tflite",
        sha256 = "f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v2_1.0_224_quant.tflite?generation=1664340173966530"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v2_1_0_224_quant_without_metadata_tflite",
        sha256 = "f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v2_1.0_224_quant_without_metadata.tflite?generation=1665988405130772"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v2_1_0_224_tflite",
        sha256 = "ff5cb7f9e62c92ebdad971f8a98aa6b3106d82a64587a7787c6a385c9e791339",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v2_1.0_224.tflite?generation=1661875840611150"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v2_1_0_224_without_metadata_tflite",
        sha256 = "9f3bc29e38e90842a852bfed957dbf5e36f2d97a91dd17736b1e5c0aca8d3303",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v2_1.0_224_without_metadata.tflite?generation=1665988408360823"],
    )

    http_file(
        name = "com_google_mediapipe_mobilenet_v3_small_100_224_embedder_tflite",
        sha256 = "f7b9a563cb803bdcba76e8c7e82abde06f5c7a8e67b5e54e43e23095dfe79a78",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenet_v3_small_100_224_embedder.tflite?generation=1664184739429109"],
    )

    http_file(
        name = "com_google_mediapipe_mobile_object_classifier_v0_2_3-metadata-no-name_tflite",
        sha256 = "27fdb2dce68b8bd9a0f16583eefc4df13605808c1417cec268d1e838920c1a81",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_object_classifier_v0_2_3-metadata-no-name.tflite?generation=1661875843557142"],
    )

    http_file(
        name = "com_google_mediapipe_mobile_object_labeler_v1_tflite",
        sha256 = "9400671e04685f5277edd3052a311cc51533de9da94255c52ebde1e18484c77c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_object_labeler_v1.tflite?generation=1666144701839813"],
    )

    http_file(
        name = "com_google_mediapipe_model_without_metadata_tflite",
        sha256 = "05c5aea7ae00aeed0053a85f2b2e896b4ea272c5219052d32c06b655fbf5cc9b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/model_without_metadata.tflite?generation=1661875850966737"],
    )

    http_file(
        name = "com_google_mediapipe_movie_review_json",
        sha256 = "c09b88af05844cad5133b49744fed3a0bd514d4a1c75b9d2f23e9a40bd7bc04e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/movie_review.json?generation=1667892501695336"],
    )

    http_file(
        name = "com_google_mediapipe_movie_review_labels_txt",
        sha256 = "4b9b26392f765e7a872372131cd4cee8ad7c02e496b5a1228279619b138c4b7a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/movie_review_labels.txt?generation=1667892504334882"],
    )

    http_file(
        name = "com_google_mediapipe_movie_review_tflite",
        sha256 = "3935ee73b13d435327d05af4d6f37dc3c146e117e1c3d572ae4d2ae0f5f412fe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/movie_review.tflite?generation=1667855395736217"],
    )

    http_file(
        name = "com_google_mediapipe_mozart_square_jpg",
        sha256 = "4feb4dadc5d6f853ade57b8c9d4c9a1f5ececd6469616c8e505f9a14823392b6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mozart_square.jpg?generation=1661875853838871"],
    )

    http_file(
        name = "com_google_mediapipe_multi_objects_jpg",
        sha256 = "ada6e36b40519cf0a4fbdf1b535de7fa7d0c472f2c0a08ada8ee5728e16c0c68",
        urls = ["https://storage.googleapis.com/mediapipe-assets/multi_objects.jpg?generation=1663251779213308"],
    )

    http_file(
        name = "com_google_mediapipe_multi_objects_rotated_jpg",
        sha256 = "175f6c572ffbab6554e382fd5056d09720eef931ccc4ed79481bdc47a8443911",
        urls = ["https://storage.googleapis.com/mediapipe-assets/multi_objects_rotated.jpg?generation=1665065847969523"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_3d_camera_tflite",
        sha256 = "f66e92e81ed3f4698f74d565a7668e016e2288ea92fb42938e33b778bd1e110d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_3d_camera.tflite?generation=1661875857210211"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_3d_chair_1stage_tflite",
        sha256 = "694af9bdcea270f2bad488beb4e5ef89aad819489d5d9aa4a774d2fad2a91ae9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_3d_chair_1stage.tflite?generation=1661875860251330"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_3d_chair_tflite",
        sha256 = "190e4ea49ba891ed242ddc73703e03d70164c27f3da07492d7010379e24f2a6b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_3d_chair.tflite?generation=1661875863685724"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_3d_cup_tflite",
        sha256 = "c4f4ea8def16bd191d11279f754e6f3f2a9d94839a956b975e5697e943157ac7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_3d_cup.tflite?generation=1661875867924057"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_3d_sneakers_1stage_tflite",
        sha256 = "ef052353e882d93429ee90a8e8e5e781f04acdf44c0cef4d961d8cbfa89aad8c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_3d_sneakers_1stage.tflite?generation=1661875871321513"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_3d_sneakers_tflite",
        sha256 = "4eb1633d646a43ae979ba497487e95dbf89f97406ed02200ae39ae46b0a0543d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_3d_sneakers.tflite?generation=1661875875616135"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_ssd_mobilenetv2_oidv4_fp16_tflite",
        sha256 = "d0a5255bf8c4f5a0bc4240741a76c41d5e939f7655078f945f50ab53a9375da6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_ssd_mobilenetv2_oidv4_fp16.tflite?generation=1661875879063676"],
    )

    http_file(
        name = "com_google_mediapipe_palm_detection_full_tflite",
        sha256 = "1b14e9422c6ad006cde6581a46c8b90dd573c07ab7f3934b5589e7cea3f89a54",
        urls = ["https://storage.googleapis.com/mediapipe-assets/palm_detection_full.tflite?generation=1662745358034050"],
    )

    http_file(
        name = "com_google_mediapipe_palm_detection_lite_tflite",
        sha256 = "e9a4aaddf90dda56a87235303cf00e4c2d3fb28725f68fd88772997dac905c18",
        urls = ["https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite?generation=1661875885885770"],
    )

    http_file(
        name = "com_google_mediapipe_pointing_up_jpg",
        sha256 = "ecf8ca2611d08fa25948a4fc10710af9120e88243a54da6356bacea17ff3e36e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pointing_up.jpg?generation=1662650662527717"],
    )

    http_file(
        name = "com_google_mediapipe_pointing_up_landmarks_pbtxt",
        sha256 = "a3cd7f088a9e997dbb8f00d91dbf3faaacbdb262c8f2fde3c07a9d0656488065",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pointing_up_landmarks.pbtxt?generation=1665174976408451"],
    )

    http_file(
        name = "com_google_mediapipe_pointing_up_rotated_jpg",
        sha256 = "50ff66f50281207072a038e5bb6648c43f4aacbfb8204a4d2591868756aaeff1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pointing_up_rotated.jpg?generation=1666037072219697"],
    )

    http_file(
        name = "com_google_mediapipe_pointing_up_rotated_landmarks_pbtxt",
        sha256 = "5ec37218d8b613436f5c10121dc689bf9ee69af0656a6ccf8c2e3e8b652e2ad6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pointing_up_rotated_landmarks.pbtxt?generation=1666629486774022"],
    )

    http_file(
        name = "com_google_mediapipe_pose_detection_tflite",
        sha256 = "a63c614bef30d35947f13be361820b1e4e3bec9cfeebf4d11216a18373108e85",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite?generation=1661875889147923"],
    )

    http_file(
        name = "com_google_mediapipe_pose_landmark_full_tflite",
        sha256 = "e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_landmark_full.tflite?generation=1661875894245786"],
    )

    http_file(
        name = "com_google_mediapipe_pose_landmark_heavy_tflite",
        sha256 = "59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite?generation=1661875897944151"],
    )

    http_file(
        name = "com_google_mediapipe_pose_landmark_lite_tflite",
        sha256 = "f17bfbecadb61c3be1baa8b8d851cc6619c870a87167b32848ad20db306b9d61",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite?generation=1661875901231143"],
    )

    http_file(
        name = "com_google_mediapipe_README_md",
        sha256 = "a96d08c9c70cd9717207ed72c926e02e5eada751f00bdc5d3a7e82e3492b72cb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/README.md?generation=1668124179156767"],
    )

    http_file(
        name = "com_google_mediapipe_regex_one_embedding_with_metadata_tflite",
        sha256 = "b8f5d6d090c2c73984b2b92cd2fda27e5630562741a93d127b9a744d60505bc0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/regex_one_embedding_with_metadata.tflite?generation=1667888045310541"],
    )

    http_file(
        name = "com_google_mediapipe_regex_vocab_txt",
        sha256 = "b1134b10927a53ce4224bbc30ccf075c9969c94ebf40c368966d1dcf445ca923",
        urls = ["https://storage.googleapis.com/mediapipe-assets/regex_vocab.txt?generation=1667892507770551"],
    )

    http_file(
        name = "com_google_mediapipe_right_hands_jpg",
        sha256 = "240c082e80128ff1ca8a83ce645e2ba4d8bc30f0967b7991cf5fa375bab489e1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/right_hands.jpg?generation=1661875908672404"],
    )

    http_file(
        name = "com_google_mediapipe_right_hands_rotated_jpg",
        sha256 = "b3bdf692f0d54b86c8b67e6d1286dd0078fbe6e9dfcd507b187e3bd8b398c0f9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/right_hands_rotated.jpg?generation=1666037076873345"],
    )

    http_file(
        name = "com_google_mediapipe_score_calibration_file_meta_json",
        sha256 = "6a3c305620371f662419a496f75be5a10caebca7803b1e99d8d5d22ba51cda94",
        urls = ["https://storage.googleapis.com/mediapipe-assets/score_calibration_file_meta.json?generation=1665422841236117"],
    )

    http_file(
        name = "com_google_mediapipe_score_calibration_tensor_meta_json",
        sha256 = "24cbde7f76dd6a09a55d07f30493c2f254d61154eb2e8d18ed947ff56781186d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/score_calibration_tensor_meta.json?generation=1665422844327992"],
    )

    http_file(
        name = "com_google_mediapipe_score_calibration_txt",
        sha256 = "34b0c51a8c79b4515bdd24e440c4b76a9f0fd01ef6385b36af983036e7be6271",
        urls = ["https://storage.googleapis.com/mediapipe-assets/score_calibration.txt?generation=1665422847392804"],
    )

    http_file(
        name = "com_google_mediapipe_score_thresholding_meta_json",
        sha256 = "7bb74f21c2d7f0237675ed7c09d7b7afd3507c8373f51dc75fa0507852f6ee19",
        urls = ["https://storage.googleapis.com/mediapipe-assets/score_thresholding_meta.json?generation=1667273953630766"],
    )

    http_file(
        name = "com_google_mediapipe_segmentation_golden_rotation0_png",
        sha256 = "9ee993919b753118928ba2d14f7c5c83a6cfc23355e6943dac4ad81eedd73069",
        urls = ["https://storage.googleapis.com/mediapipe-assets/segmentation_golden_rotation0.png?generation=1661875911319083"],
    )

    http_file(
        name = "com_google_mediapipe_segmentation_input_rotation0_jpg",
        sha256 = "5bf58d8af1f1c33224f3f3bc0ce451c8daf0739cc15a86d59d8c3bf2879afb97",
        urls = ["https://storage.googleapis.com/mediapipe-assets/segmentation_input_rotation0.jpg?generation=1661875914048401"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segm_128_128_3_expected_mask_jpg",
        sha256 = "a295f3ab394a5e0caff2db5041337da58341ec331f1413ef91f56e0d650b4a1e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segm_128_128_3_expected_mask.jpg?generation=1661875916766416"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segm_128_128_3_tflite",
        sha256 = "bb154f248543c0738e32f1c74375245651351a84746dc21f10bdfaabd8fae4ca",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segm_128_128_3.tflite?generation=1661875919964123"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segm_144_256_3_expected_mask_jpg",
        sha256 = "cfc699db9670585c04414d0d1a07b289a027ba99d6903d2219f897d34e2c9952",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segm_144_256_3_expected_mask.jpg?generation=1661875922646736"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segm_144_256_3_tflite",
        sha256 = "5c770b8834ad50586599eae7710921be09d356898413fc0bf37a9458da0610eb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segm_144_256_3.tflite?generation=1661875925519713"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segmentation_landscape_tflite",
        sha256 = "4aafe6223bb8dac6fac8ca8ed56852870a33051ef3f6238822d282a109962894",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segmentation_landscape.tflite?generation=1661875928328455"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segmentation_tflite",
        sha256 = "8d13b7fae74af625c641226813616a2117bd6bca19eb3b75574621fc08557f27",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite?generation=1661875931201364"],
    )

    http_file(
        name = "com_google_mediapipe_sentence_piece_tokenizer_meta_json",
        sha256 = "416bfe231710502e4a93e1b1950c0c6e5db49cffb256d241ef3d3f2d0d57718b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/sentence_piece_tokenizer_meta.json?generation=1667948375508564"],
    )

    http_file(
        name = "com_google_mediapipe_speech_16000_hz_mono_wav",
        sha256 = "71caf50b8757d6ab9cad5eae4d36669d3c20c225a51660afd7fe0dc44cdb74f6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/speech_16000_hz_mono.wav?generation=1661875934539524"],
    )

    http_file(
        name = "com_google_mediapipe_speech_48000_hz_mono_wav",
        sha256 = "04d4590b61d0519170d7aa0686ab2ff5da2b8487d192e40413dd36d9c1a24304",
        urls = ["https://storage.googleapis.com/mediapipe-assets/speech_48000_hz_mono.wav?generation=1661875938066405"],
    )

    http_file(
        name = "com_google_mediapipe_ssdlite_object_detection_labelmap_txt",
        sha256 = "c7e79c855f73cbba9f33d649d60e1676eb0a974021a41696d1ac0d4b7f7e0211",
        urls = ["https://storage.googleapis.com/mediapipe-assets/ssdlite_object_detection_labelmap.txt?generation=1661875940778557"],
    )

    http_file(
        name = "com_google_mediapipe_ssdlite_object_detection_tflite",
        sha256 = "8e10a2e2f5db85d8f90628f00752a89ff241c5b2ca82f3b92fc496c7bda122ef",
        urls = ["https://storage.googleapis.com/mediapipe-assets/ssdlite_object_detection.tflite?generation=1661875944118759"],
    )

    http_file(
        name = "com_google_mediapipe_ssd_mobilenet_v1_tflite",
        sha256 = "cbdecd08b44c5dea3821f77c5468e2936ecfbf43cde0795a2729fdb43401e58b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/ssd_mobilenet_v1.tflite?generation=1661875947436302"],
    )

    http_file(
        name = "com_google_mediapipe_test_jpg",
        sha256 = "798a12a466933842528d8438f553320eebe5137f02650f12dd68706a2f94fb4f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/test.jpg?generation=1664672140191116"],
    )

    http_file(
        name = "com_google_mediapipe_test_model_add_op_tflite",
        sha256 = "298300ca8a9193b80ada1dca39d36f20bffeebde09e85385049b3bfe7be2272f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/test_model_add_op.tflite?generation=1661875950076192"],
    )

    http_file(
        name = "com_google_mediapipe_test_model_text_classifier_bool_output_tflite",
        sha256 = "09877ac6d718d78da6380e21fe8179854909d116632d6d770c12f8a51792e310",
        urls = ["https://storage.googleapis.com/mediapipe-assets/test_model_text_classifier_bool_output.tflite?generation=1664904110313163"],
    )

    http_file(
        name = "com_google_mediapipe_test_model_text_classifier_with_regex_tokenizer_tflite",
        sha256 = "cb12618d084b813cb7b90ceb39c9fe4b18dae4de9880b912cdcd4b577cd65b4f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/test_model_text_classifier_with_regex_tokenizer.tflite?generation=1663009546758456"],
    )

    http_file(
        name = "com_google_mediapipe_test_model_with_custom_op_tflite",
        sha256 = "bafff7c8508ac24846e089ab70dcf48943a483a3e20290ff60e7740d073d7653",
        urls = ["https://storage.googleapis.com/mediapipe-assets/test_model_with_custom_op.tflite?generation=1661875957061036"],
    )

    http_file(
        name = "com_google_mediapipe_test_model_without_custom_op_tflite",
        sha256 = "e17f0a1a22bc9242d9f825fe1edce07d2f90eb2a57e8b29a996244f194ee08a0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/test_model_without_custom_op.tflite?generation=1661875959757731"],
    )

    http_file(
        name = "com_google_mediapipe_thumb_up_jpg",
        sha256 = "5d673c081ab13b8a1812269ff57047066f9c33c07db5f4178089e8cb3fdc0291",
        urls = ["https://storage.googleapis.com/mediapipe-assets/thumb_up.jpg?generation=1662650667349746"],
    )

    http_file(
        name = "com_google_mediapipe_thumb_up_landmarks_pbtxt",
        sha256 = "b129ae0536be4e25d6cdee74aabe9dedf1bcfe87430a40b68be4079db3a4d926",
        urls = ["https://storage.googleapis.com/mediapipe-assets/thumb_up_landmarks.pbtxt?generation=1665174979747784"],
    )

    http_file(
        name = "com_google_mediapipe_thumb_up_rotated_landmarks_pbtxt",
        sha256 = "6645bbd98ea7f90b3e1ba297e16ea5280847fc5bf5400726d98c282f6c597257",
        urls = ["https://storage.googleapis.com/mediapipe-assets/thumb_up_rotated_landmarks.pbtxt?generation=1666629489421733"],
    )

    http_file(
        name = "com_google_mediapipe_two_heads_16000_hz_mono_wav",
        sha256 = "a291a9c22c39bba30138a26915e154a96286ba6ca3b413053123c504a58cce3b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/two_heads_16000_hz_mono.wav?generation=1661875962421337"],
    )

    http_file(
        name = "com_google_mediapipe_two_heads_44100_hz_mono_wav",
        sha256 = "1bf525ad7b7bac2da65addb5593b49adaba52ec3a9ed891f70afe0b392db02cd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/two_heads_44100_hz_mono.wav?generation=1661875965316595"],
    )

    http_file(
        name = "com_google_mediapipe_two_heads_tflite",
        sha256 = "bfa6ee4ccaf9180b69b39fa579b26b74bbf7758ae398e1d2265a58d323ca3d84",
        urls = ["https://storage.googleapis.com/mediapipe-assets/two_heads.tflite?generation=1661875968723352"],
    )

    http_file(
        name = "com_google_mediapipe_universal_sentence_encoder_qa_with_metadata_tflite",
        sha256 = "82c2d0450aa458adbec2f78eff33cfbf2a41b606b44246726ab67373926e32bc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/universal_sentence_encoder_qa_with_metadata.tflite?generation=1665445919252005"],
    )

    http_file(
        name = "com_google_mediapipe_victory_jpg",
        sha256 = "84cb8853e3df614e0cb5c93a25e3e2f38ea5e4f92fd428ee7d867ed3479d5764",
        urls = ["https://storage.googleapis.com/mediapipe-assets/victory.jpg?generation=1666999364225126"],
    )

    http_file(
        name = "com_google_mediapipe_victory_landmarks_pbtxt",
        sha256 = "b25ab4f222674489f543afb6454396ecbc1437a7ae6213dbf0553029ae939ab0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/victory_landmarks.pbtxt?generation=1666999366036622"],
    )

    http_file(
        name = "com_google_mediapipe_vocab_for_regex_tokenizer_txt",
        sha256 = "b1134b10927a53ce4224bbc30ccf075c9969c94ebf40c368966d1dcf445ca923",
        urls = ["https://storage.googleapis.com/mediapipe-assets/vocab_for_regex_tokenizer.txt?generation=1661875971574893"],
    )

    http_file(
        name = "com_google_mediapipe_vocab_txt",
        sha256 = "a125f531f48943ac4c3f117112150b91825aed560d890718dd96dc764a2bc141",
        urls = ["https://storage.googleapis.com/mediapipe-assets/vocab.txt?generation=1661875974626008"],
    )

    http_file(
        name = "com_google_mediapipe_vocab_with_index_txt",
        sha256 = "664d78a2835bba781c23f9b556886bfcd8eef3d2a7414cf31d5c6963d9669379",
        urls = ["https://storage.googleapis.com/mediapipe-assets/vocab_with_index.txt?generation=1661875977280658"],
    )

    http_file(
        name = "com_google_mediapipe_yamnet_audio_classifier_with_metadata_tflite",
        sha256 = "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de",
        urls = ["https://storage.googleapis.com/mediapipe-assets/yamnet_audio_classifier_with_metadata.tflite?generation=1661875980774466"],
    )

    http_file(
        name = "com_google_mediapipe_yamnet_embedding_metadata_tflite",
        sha256 = "7baa72708e3919bae5a5dc78d932847bc28008af14febd083eff62d28af9c72a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/yamnet_embedding_metadata.tflite?generation=1668295071595506"],
    )

    http_file(
        name = "com_google_mediapipe_gesture_embedder_keras_metadata_pb",
        sha256 = "c76b856101e2284293a5e5963b7c445e407a0b3e56ec63eb78f64d883e51e3aa",
        urls = ["https://storage.googleapis.com/mediapipe-assets/gesture_embedder/keras_metadata.pb?generation=1668550482128410"],
    )

    http_file(
        name = "com_google_mediapipe_gesture_embedder_saved_model_pb",
        sha256 = "0082d37c5b85487fbf553e00a63f640945faf3da2d561a5f5a24c3194fecda6a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/gesture_embedder/saved_model.pb?generation=1668550484904822"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_saved_model_model_ckpt_data-00000-of-00001",
        sha256 = "ad2f733f271dd5000a8c7f926bfea1083e6408b34d4f3b60679e5a6f96251c97",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_saved_model/model.ckpt.data-00000-of-00001?generation=1661875984176294"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_saved_model_model_ckpt_index",
        sha256 = "283816fcab228e6246d1c03b596f50dd40e4fe3e04c52a522a5b9d6f2cc43273",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_saved_model/model.ckpt.index?generation=1661875987100245"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_saved_model_model_ckpt_meta",
        sha256 = "9d80696ab76a492a23f6ce1d0d33b2d13c26e118b86d3ef61b691ad67d0f1f5a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_saved_model/model.ckpt.meta?generation=1661875990332395"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_saved_model_pipeline_config",
        sha256 = "995aff0b28af5f66eb98d0734494395710ae84c843aee207755e7bc5025c9abb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_saved_model/pipeline.config?generation=1661875993079273"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_saved_model_README_md",
        sha256 = "fe163cf12fbd017738a2fd360c03d223e964ba6404ac75c635f5918784e9c34d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_saved_model/README.md?generation=1661875995856372"],
    )

    http_file(
        name = "com_google_mediapipe_object_detection_saved_model_saved_model_pb",
        sha256 = "f29606cf218397d5580c496e50fd28cddf66e2f59b819ab9c761b72270a5adf3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_saved_model/saved_model.pb?generation=1661875999264354"],
    )

    http_file(
        name = "com_google_mediapipe_gesture_embedder_variables_variables_data-00000-of-00001",
        sha256 = "c156c9654c9ffb1091bb9f06c71080bd1e428586276d3f39c33fbab27fe0522d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/gesture_embedder/variables/variables.data-00000-of-00001?generation=1668550487965052"],
    )

    http_file(
        name = "com_google_mediapipe_gesture_embedder_variables_variables_index",
        sha256 = "76ea482b8da6bdb3d65d3b2ea989c1699c9fa0d6df0cb6d80863d1dc6fe7c4bd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/gesture_embedder/variables/variables.index?generation=1668550490691823"],
    )
