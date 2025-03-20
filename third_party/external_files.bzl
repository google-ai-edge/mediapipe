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
        name = "com_google_mediapipe_average_word_classifier_tflite",
        sha256 = "13bf6f7f35964f1e85d6cc762ee7b1952009b532b233baa5bdb4bf7441097f34",
        urls = ["https://storage.googleapis.com/mediapipe-assets/average_word_classifier.tflite?generation=1699635086044360"],
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
        sha256 = "cfbc1404ba18ee9eb0f08e9ee66d5b51f3fac47f683a5fa0cc23b46f30e05a1f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/BUILD?generation=1686332366306166"],
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
        name = "com_google_mediapipe_category_tensor_float_meta_json",
        sha256 = "d0cbe95a99ffc57046d7e66cf194600d12899216a4d3bf1a3851811648005e38",
        urls = ["https://storage.googleapis.com/mediapipe-assets/category_tensor_float_meta.json?generation=1677522730922512"],
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
        name = "com_google_mediapipe_cats_and_dogs_mask_dog1_png",
        sha256 = "2ab37d56ba1e46e70b3ddbfe35dac51b18b597b76904c68d7d34c7c74c677d4c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cats_and_dogs_mask_dog1.png?generation=1678840350058498"],
    )

    http_file(
        name = "com_google_mediapipe_cats_and_dogs_mask_dog2_png",
        sha256 = "2010850e2dd7f520fe53b9086d70913b6fb53b178cae15a373e5ee7ffb46824a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/cats_and_dogs_mask_dog2.png?generation=1678840352961684"],
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
        name = "com_google_mediapipe_coco_ssd_mobilenet_v1_1_0_quant_2018_06_29_no_metadata_tflite",
        sha256 = "e4b118e5e4531945de2e659742c7c590f7536f8d0ed26d135abcfe83b4779d13",
        urls = ["https://storage.googleapis.com/mediapipe-assets/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_no_metadata.tflite?generation=1677522735292070"],
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
        name = "com_google_mediapipe_coco_ssd_mobilenet_v1_score_calibration_json",
        sha256 = "a850674f9043bfc775527fee7f1b639f7fe0fb56e8d3ed2b710247967c888b09",
        urls = ["https://storage.googleapis.com/mediapipe-assets/coco_ssd_mobilenet_v1_score_calibration.json?generation=1682456086898538"],
    )

    http_file(
        name = "com_google_mediapipe_conv2d_input_channel_1_tflite",
        sha256 = "ccb667092f3aed3a35a57fb3478fecc0c8f6360dbf477a9db9c24e5b3ec4273e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/conv2d_input_channel_1.tflite?generation=1683252905577703"],
    )

    http_file(
        name = "com_google_mediapipe_corrupted_mobilenet_v1_0_25_224_1_default_1_tflite",
        sha256 = "f0cbeb8061f4c693e20de779ce255af923508492e8a24f6db320845a52facb51",
        urls = ["https://storage.googleapis.com/mediapipe-assets/corrupted_mobilenet_v1_0.25_224_1_default_1.tflite?generation=1661875706780536"],
    )

    http_file(
        name = "com_google_mediapipe_deeplabv3_json",
        sha256 = "f299835bd9ea1cceb25fdf40a761a22716cbd20025cd67c365a860527f178b7f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/deeplabv3.json?generation=1678818040715103"],
    )

    http_file(
        name = "com_google_mediapipe_deeplabv3_tflite",
        sha256 = "5faed2c653905d3e22a8f6f29ee198da84e9b0e7936a207bf431f17f6b4d87ff",
        urls = ["https://storage.googleapis.com/mediapipe-assets/deeplabv3.tflite?generation=1678775085237701"],
    )

    http_file(
        name = "com_google_mediapipe_deeplabv3_with_activation_json",
        sha256 = "a7633476d02f970db3cc30f5f027bcb608149e02207b2ccae36a4b69d730c82c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/deeplabv3_with_activation.json?generation=1678818047050984"],
    )

    http_file(
        name = "com_google_mediapipe_deeplabv3_without_labels_json",
        sha256 = "7d045a583a4046f17a52d2078b0175607a45ed0cc187558325f9c66534c08401",
        urls = ["https://storage.googleapis.com/mediapipe-assets/deeplabv3_without_labels.json?generation=1678818050191996"],
    )

    http_file(
        name = "com_google_mediapipe_deeplabv3_without_metadata_tflite",
        sha256 = "68a539782c2c6a72f8aac3724600124a85ed977162b44e84cbae5db717c933c6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/deeplabv3_without_metadata.tflite?generation=1678818053623010"],
    )

    http_file(
        name = "com_google_mediapipe_dense_tflite",
        sha256 = "6795e7c3a263f44e97be048a5e1166e0921b453bfbaf037f4f69ac5c059ee945",
        urls = ["https://storage.googleapis.com/mediapipe-assets/dense.tflite?generation=1683252907920466"],
    )

    http_file(
        name = "com_google_mediapipe_dummy_face_stylizer_tflite",
        sha256 = "c44a32a673790aac4aca63ca4b4192b9870c21045241e69d9fe09b7ad1a38d65",
        urls = ["https://storage.googleapis.com/mediapipe-assets/dummy_face_stylizer.tflite?generation=1682960595073526"],
    )

    http_file(
        name = "com_google_mediapipe_dummy_gesture_recognizer_task",
        sha256 = "76de8c58d206d098557959d574953c2db3a4363fa52922ca198450d5d696814d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/dummy_gesture_recognizer.task?generation=1728509644439822"],
    )

    http_file(
        name = "com_google_mediapipe_dynamic_input_classifier_tflite",
        sha256 = "c5499daf5773cef89ce984df329c6324194a83bea7c7cf83159bf660a58de85c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/dynamic_input_classifier.tflite?generation=1693433004555536"],
    )

    http_file(
        name = "com_google_mediapipe_efficientdet_lite0_fp16_no_nms_anchors_csv",
        sha256 = "284475a0f16e34afcc6c0fe68b05bd871aca5b20c83db0870c6a36dd63827176",
        urls = ["https://storage.googleapis.com/mediapipe-assets/efficientdet_lite0_fp16_no_nms_anchors.csv?generation=1682456090001817"],
    )

    http_file(
        name = "com_google_mediapipe_efficientdet_lite0_fp16_no_nms_json",
        sha256 = "dc3b333e41c43fb49ace048c25c18d0e34df78fb5ee77edbe169264368f78b92",
        urls = ["https://storage.googleapis.com/mediapipe-assets/efficientdet_lite0_fp16_no_nms.json?generation=1682456092938505"],
    )

    http_file(
        name = "com_google_mediapipe_efficientdet_lite0_fp16_no_nms_tflite",
        sha256 = "237a58389081333e5cf4154e42b593ce7dd357445536fcaf4ca5bc51c2c50f1c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/efficientdet_lite0_fp16_no_nms.tflite?generation=1730305296514873"],
    )

    http_file(
        name = "com_google_mediapipe_efficientdet_lite0_v1_json",
        sha256 = "ef9706696a3ea5d87f4324ac56e877a92033d33e522c4b7d5a416fbcab24d8fc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/efficientdet_lite0_v1.json?generation=1682456098581704"],
    )

    http_file(
        name = "com_google_mediapipe_efficientdet_lite0_v1_tflite",
        sha256 = "f97efd21d6009a7b4b94b3e57baaeb77ec3225b42d32477f5003736a8084c081",
        urls = ["https://storage.googleapis.com/mediapipe-assets/efficientdet_lite0_v1.tflite?generation=1677522750449279"],
    )

    http_file(
        name = "com_google_mediapipe_empty_vocab_for_regex_tokenizer_txt",
        sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        urls = ["https://storage.googleapis.com/mediapipe-assets/empty_vocab_for_regex_tokenizer.txt?generation=1661875714907539"],
    )

    http_file(
        name = "com_google_mediapipe_expected_left_down_hand_landmarks_prototxt",
        sha256 = "f281b745175aaa7f458def6cf4c89521fb56302dd61a05642b3b4a4f237ffaa3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_left_down_hand_landmarks.prototxt?generation=1692121979089068"],
    )

    http_file(
        name = "com_google_mediapipe_expected_left_up_hand_landmarks_prototxt",
        sha256 = "174cf5f7c3ab547f0affb666ee7be933b0758c60fbfe7b7e93795c5082555592",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_left_up_hand_landmarks.prototxt?generation=1692121981605963"],
    )

    http_file(
        name = "com_google_mediapipe_expected_pose_landmarks_prototxt",
        sha256 = "eed8dfa169b0abee60cde01496599b0bc75d91a82594a1bdf59be2f76f45d7f5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_pose_landmarks.prototxt?generation=16812442325229901681244235071100"],
    )

    http_file(
        name = "com_google_mediapipe_expected_right_down_hand_landmarks_prototxt",
        sha256 = "ae9cb01035f18b0023fc12256c048666da76b41b327cec09c2d2820054b1295f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_right_down_hand_landmarks.prototxt?generation=1692121986324450"],
    )

    http_file(
        name = "com_google_mediapipe_expected_right_down_hand_rotated_landmarks_prototxt",
        sha256 = "c4dfdcc2e4cd366eb5f8ad227be94049eb593e3a528564611094687912463687",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_right_down_hand_rotated_landmarks.prototxt?generation=1692121989028161"],
    )

    http_file(
        name = "com_google_mediapipe_expected_right_up_hand_landmarks_prototxt",
        sha256 = "1353ba617c4f048083618587cd23a8a22115f634521c153d4e1bd1ebd4f49dd7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_right_up_hand_landmarks.prototxt?generation=1692121991596258"],
    )

    http_file(
        name = "com_google_mediapipe_expected_right_up_hand_rotated_landmarks_prototxt",
        sha256 = "7fb2d33cf69d2da50952a45bad0c0618f30859e608958fee95948a6e0de63ccb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/expected_right_up_hand_rotated_landmarks.prototxt?generation=1692121994043161"],
    )

    http_file(
        name = "com_google_mediapipe_external_file_txt",
        sha256 = "ae0666f161fed1a5dde998bbd0e140550d2da0db27db1d0e31e370f2bd366a57",
        urls = ["https://storage.googleapis.com/mediapipe-assets/external_file.txt?generation=1661875736240688"],
    )

    http_file(
        name = "com_google_mediapipe_face_detection_full_range_sparse_tflite",
        sha256 = "2c3728e6da56f21e21a320433396fb06d40d9088f2247c05e5635a688d45dfe1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_detection_full_range_sparse.tflite?generation=1674261618323821"],
    )

    http_file(
        name = "com_google_mediapipe_face_detection_full_range_tflite",
        sha256 = "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite?generation=1674261620964007"],
    )

    http_file(
        name = "com_google_mediapipe_face_detection_short_range_tflite",
        sha256 = "bbff11cebd1eb27a1e004cae0b0e63ec8c551cbf34a4451148b4908b8db3eca8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite?generation=1677044301978921"],
    )

    http_file(
        name = "com_google_mediapipe_face_geometry_expected_out_pbtxt",
        sha256 = "c23c55c14b24523e7fe51ee9ff90b9d4d32d82852ab3e452af9064e60c91c4d1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_geometry_expected_out.pbtxt?generation=1737065432962469"],
    )

    http_file(
        name = "com_google_mediapipe_face_landmarker_task",
        sha256 = "7cf2bbf1842c429e9defee38e7f1c4238978d8a6faf2da145bb19846f86bd2f4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_landmarker.task?generation=1678323583183024"],
    )

    http_file(
        name = "com_google_mediapipe_face_landmarker_v2_task",
        sha256 = "af23fc7c1ff21d034deaa2b7fc1d56bb670ce69a4cbdc9579b6f1afd680835f4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task?generation=1681322464758457"],
    )

    http_file(
        name = "com_google_mediapipe_face_landmarker_v2_with_blendshapes_task",
        sha256 = "b261925d4aad812b47a0e8d58c1baa1223270a5d1f663d78338bc881c003879d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2_with_blendshapes.task?generation=1681322467931433"],
    )

    http_file(
        name = "com_google_mediapipe_face_landmarker_with_blendshapes_task",
        sha256 = "b44e4cae6f5822456d60f33e7c852640d78c7e342aee7eacc22589451a0b9dc2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_landmarker_with_blendshapes.task?generation=1678504998301299"],
    )

    http_file(
        name = "com_google_mediapipe_face_landmark_tflite",
        sha256 = "1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite?generation=1676316347980492"],
    )

    http_file(
        name = "com_google_mediapipe_face_landmark_with_attention_tflite",
        sha256 = "e06a804e0144f9929eda782122916b35d60c697c3c9344013ca2bbe76a6ce2b4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_landmark_with_attention.tflite?generation=1676415468821650"],
    )

    http_file(
        name = "com_google_mediapipe_facemesh2_lite_iris_faceflag_2023_02_14_tflite",
        sha256 = "bc5ee5de06d8c3a5465c3155227615b164480a52105a2b3df5748250ab4d914f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/facemesh2_lite_iris_faceflag_2023_02_14.tflite?generation=1681322470818178"],
    )

    http_file(
        name = "com_google_mediapipe_face_stylization_dummy_tflite",
        sha256 = "f57fd2d5638def25466f6fec142eb3397d8ad99a9bd0a9344b622bad7c3f0376",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_stylization_dummy.tflite?generation=1678323589048063"],
    )

    http_file(
        name = "com_google_mediapipe_face_stylizer_color_ink_task",
        sha256 = "887a490b74046ecb2b1d092cc0173a961b4ed3640aaadeafa852b1122ca23b2a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_stylizer_color_ink.task?generation=1697732437695259"],
    )

    http_file(
        name = "com_google_mediapipe_face_stylizer_json",
        sha256 = "ad89860d5daba6a1c4163a576428713fc3ddab76d6bbaf06d675164423ae159f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_stylizer.json?generation=1682960598942694"],
    )

    http_file(
        name = "com_google_mediapipe_face_stylizer_task",
        sha256 = "423f350aab236123818adb7b39e0a14e14708a9a019fb2fe00a015a2561fd0c8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/face_stylizer.task?generation=1693433010526766"],
    )

    http_file(
        name = "com_google_mediapipe_falcon_rw_1b_test_weight_pt",
        sha256 = "62972530d362e881747f0f309573f32421a13b787603ab89874a23f4a5d44f44",
        urls = ["https://storage.googleapis.com/mediapipe-assets/falcon_rw_1b_test_weight.pt?generation=1728509646701969"],
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
        sha256 = "4b0ad2b00d5f2d140450f9f168af0f7422ecf6b630b7d64a213bcf6f04fb078b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/fist_landmarks.pbtxt?generation=1692121997451835"],
    )

    http_file(
        name = "com_google_mediapipe_fist_png",
        sha256 = "4397b3d3f590c88a8de7d21c08d73a0df4a97fd93f92cbd086eef37fd246daaa",
        urls = ["https://storage.googleapis.com/mediapipe-assets/fist.png?generation=1672952068696274"],
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
        sha256 = "d48562f535fd4ecd3cfea739d9663dd818eeaf6a8afb1b5e6f8f4747661f73d9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/gesture_recognizer.task?generation=1677051715043311"],
    )

    http_file(
        name = "com_google_mediapipe_golden_json_json",
        sha256 = "55c0c88748d099aa379930504df62c6c8f1d8874ea52d2f8a925f352c4c7f09c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/golden_json.json?generation=1664340169675228"],
    )

    http_file(
        name = "com_google_mediapipe_hair_segmentation_tflite",
        sha256 = "7cbddcfe6f6e10c3e0a509eb2e14225fda5c0de6c35e2e8c6ca8e3971988fc17",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite?generation=1678775089064550"],
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
        sha256 = "32d1eab97e80a9a20edb29231e15301ce65abfd0fa9d41cf1757e0ecc8078a4e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task?generation=1677051718270846"],
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
        name = "com_google_mediapipe_handrecrop_2020_07_21_v0_f16_tflite",
        sha256 = "d40b15e15f93f6c909a3cfb881ce16c9ff9aa6d57417a0c906a6624f1f60b60c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/handrecrop_2020_07_21_v0.f16.tflite?generation=1695159193831748"],
    )

    http_file(
        name = "com_google_mediapipe_hand_recrop_tflite",
        sha256 = "67d996ce96f9d36fe17d2693022c6da93168026ab2f028f9e2365398d8ac7d5d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_recrop.tflite?generation=1661875770633070"],
    )

    http_file(
        name = "com_google_mediapipe_hand_roi_refinement_generated_graph_pbtxt",
        sha256 = "a2304dedc6f1d167996d7261cd5b5e5db843c3b5657a367d9341388f350bdcc2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/hand_roi_refinement_generated_graph.pbtxt?generation=1695159196033618"],
    )

    http_file(
        name = "com_google_mediapipe_holistic_hand_tracking_left_hand_graph_pbtxt",
        sha256 = "c964589b448471c0cd9e0f68c243e232e6f8a4c0959b41a3cd1cbb14e9efa6b1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/holistic_hand_tracking_left_hand_graph.pbtxt?generation=1697732440362430"],
    )

    http_file(
        name = "com_google_mediapipe_holistic_landmarker_task",
        sha256 = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/holistic_landmarker.task?generation=1699635090585884"],
    )

    http_file(
        name = "com_google_mediapipe_holistic_pose_tracking_graph_pbtxt",
        sha256 = "1d36d014d38c09fea73042471d5d1a616f3cc9f22c8ca625deabc38efd63f6aa",
        urls = ["https://storage.googleapis.com/mediapipe-assets/holistic_pose_tracking_graph.pbtxt?generation=1697732442566093"],
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
        name = "com_google_mediapipe_labelmap_txt",
        sha256 = "f8803ef7900160c629d570848dfda4175e21667bf7b71f73f8ece4938c9f2bf2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/labelmap.txt?generation=1677522772140291"],
    )

    http_file(
        name = "com_google_mediapipe_labels_txt",
        sha256 = "536feacc519de3d418de26b2effb4d75694a8c4c0063e36499a46fa8061e2da9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/labels.txt?generation=1667892497527642"],
    )

    http_file(
        name = "com_google_mediapipe_language_detector_tflite",
        sha256 = "5f64d821110dd2a3280546e8cd59dff09547e25d5f5c9711ec3f03416414dbb2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/language_detector.tflite?generation=1678323592870401"],
    )

    http_file(
        name = "com_google_mediapipe_left_hands_jpg",
        sha256 = "240c082e80128ff1ca8a83ce645e2ba4d8bc30f0967b7991cf5fa375bab489e1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/left_hands.jpg?generation=1692122001487742"],
    )

    http_file(
        name = "com_google_mediapipe_left_hands_rotated_jpg",
        sha256 = "b3bdf692f0d54b86c8b67e6d1286dd0078fbe6e9dfcd507b187e3bd8b398c0f9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/left_hands_rotated.jpg?generation=1692122004272021"],
    )

    http_file(
        name = "com_google_mediapipe_leopard_bg_removal_result_512x512_png",
        sha256 = "30be22e89fdd1d7b985294498ec67509b0caa1ca941fe291fa25f43a3873e4dd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/leopard_bg_removal_result_512x512.png?generation=1690239134617707"],
    )

    http_file(
        name = "com_google_mediapipe_leopard_bg_removal_result_png",
        sha256 = "afd33f2058fd58d189cda86ec931647741a6139970c9bcbc637cdd151ec657c5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/leopard_bg_removal_result.png?generation=1685997278308542"],
    )

    http_file(
        name = "com_google_mediapipe_leopard_jpg",
        sha256 = "d66fda0aa655f87c9fe87965a642e7b33ec990a3d9ed5812f1e5513da9d7d744",
        urls = ["https://storage.googleapis.com/mediapipe-assets/leopard.jpg?generation=1685997280368627"],
    )

    http_file(
        name = "com_google_mediapipe_libc___shared_so",
        sha256 = "816d497229b6678db485b5dc16ae7d2ac63dc015691b1828bc35c4aa2ed6eed4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/libc++_shared.so?generation=1730305298946708"],
    )

    http_file(
        name = "com_google_mediapipe_libimagegenerator_gpu_so",
        sha256 = "e4407c7c0a2559b168a0f76cda6eb23ce2d167fa757a0d4887ccf57af70c0179",
        urls = ["https://storage.googleapis.com/mediapipe-assets/libimagegenerator_gpu.so?generation=1728509649298093"],
    )

    http_file(
        name = "com_google_mediapipe_libopencv_core_3_4_darwin_a",
        sha256 = "1e4355f5a3813f7656cf7c2d64b6c5d42d4c20688c2c4a08b51572b714e3795a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/libopencv_core.3.4.darwin.a?generation=1728509651551637"],
    )

    http_file(
        name = "com_google_mediapipe_libopencv_core_3_4_darwin_arm64_a",
        sha256 = "4a836f880d86123cb8bc1d838e53bf44053c683951c4e87921dddca7434d8be3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/libopencv_core.3.4.darwin_arm64.a?generation=1728509653713172"],
    )

    http_file(
        name = "com_google_mediapipe_libopencv_imgproc_3_4_darwin_a",
        sha256 = "6501109ed42eec78f54330ec14b99cae86c5f0db52651636ccf2a44297808eb7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/libopencv_imgproc.3.4.darwin.a?generation=1728509656218122"],
    )

    http_file(
        name = "com_google_mediapipe_libopencv_imgproc_3_4_darwin_arm64_a",
        sha256 = "4e8019ca2732c07c173188dbf156b8321e350873ce2115327642788001ead246",
        urls = ["https://storage.googleapis.com/mediapipe-assets/libopencv_imgproc.3.4.darwin_arm64.a?generation=1728509658402901"],
    )

    http_file(
        name = "com_google_mediapipe_living_room_jpg",
        sha256 = "8d74535dfe58e7d62dee99df5ab7741ad373a456797cf4d99048dfd17ccb0d6c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/living_room.jpg?generation=1699635092884512"],
    )

    http_file(
        name = "com_google_mediapipe_male_full_height_hands_jpg",
        sha256 = "8a7fe5be8b90d6078b09913ca28f7e5d342f8d3cde856ab4e3327d2970b887f8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg?generation=1692651585540897"],
    )

    http_file(
        name = "com_google_mediapipe_male_full_height_hands_result_cpu_pbtxt",
        sha256 = "f4a53dec51b621abeb1dbd854087dd0b02a3d40472f5fdbc5e836315bcb704f3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/male_full_height_hands_result_cpu.pbtxt?generation=1699635094875663"],
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
        name = "com_google_mediapipe_mobile_bg_removal_tflite",
        sha256 = "f85797391cd1ef03988441710781342a77a980665965771fba603e5aee940ee8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_bg_removal.tflite?generation=1685997284190857"],
    )

    http_file(
        name = "com_google_mediapipe_mobile_ica_8bit-with-custom-metadata_tflite",
        sha256 = "31f34f0dd0dc39e69e9c3deb1e3f3278febeb82ecf57c235834348a75df8fb51",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_ica_8bit-with-custom-metadata.tflite?generation=1677906531317767"],
    )

    http_file(
        name = "com_google_mediapipe_mobile_ica_8bit-with-large-min-parser-version_tflite",
        sha256 = "53d0ea047682539964820fcfc5dc81f4928957470f453f2065f4c2ab87406803",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_ica_8bit-with-large-min-parser-version.tflite?generation=1677906534624784"],
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
        name = "com_google_mediapipe_mobilenetsweep_dptrigmqn384_unit_384_384_fp16quant_fp32input_opt_tflite",
        sha256 = "3c4c7e36b35fc903ecfb51b351b4849b23c57cc18d1416cf6cabaa1522d84760",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilenetsweep_dptrigmqn384_unit_384_384_fp16quant_fp32input_opt.tflite?generation=1690302146106240"],
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
        name = "com_google_mediapipe_mobile_raid_det_nms_max_detections_40_max_labels_per_box_5_norm_coord_tflite",
        sha256 = "7df0e6fa124c6f30e5c5661f244e6c98bb8470ce2b487597541996b610c7cc87",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_raid_det_nms_max_detections_40_max_labels_per_box_5_norm_coord.tflite?generation=1728509661004483"],
    )

    http_file(
        name = "com_google_mediapipe_mobile_raid_one_stage_v2_1_uint8_tflite",
        sha256 = "2e397b750d8f270e3f41731c1ec1f5b7811f93bc3a39fb81a4c47dd5e9055915",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobile_raid_one_stage_v2_1_uint8.tflite?generation=1699635098658451"],
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
        name = "com_google_mediapipe_mraid_v2_2_multiclass_nms_200_40_qat_tflite",
        sha256 = "1507a03e3c0e2567d2384b159dc17e21fb67f86c6390b10aab5d74275640ebda",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mraid_v2_2_multiclass_nms_200_40_qat.tflite?generation=1737065441446005"],
    )

    http_file(
        name = "com_google_mediapipe_mraid_v25_max_boxes_40_max_classes_per_box_5_tflite",
        sha256 = "1c4667b1a3caf90cc6971517df2525c0f8b6807d6598cf4554a9ca224faf42c5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mraid_v25_max_boxes_40_max_classes_per_box_5.tflite?generation=1699635101036991"],
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
        name = "com_google_mediapipe_ocr_text_jpg",
        sha256 = "88052e93aa910330433741f5cef140f8f9ec463230a332aef7038b5457b06482",
        urls = ["https://storage.googleapis.com/mediapipe-assets/ocr_text.jpg?generation=1681244241009078"],
    )

    http_file(
        name = "com_google_mediapipe_opencv2_xcframework_zip",
        sha256 = "6b625f564b72fd7c0946de2ae61507bed5daf84f3dbb5296ef4fef19da491160",
        urls = ["https://storage.googleapis.com/mediapipe-assets/opencv2.xcframework.zip?generation=1728573757773068"],
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
        name = "com_google_mediapipe_penguins_large_jpg",
        sha256 = "3a7a74bf946b3e2b53a3953516a552df854b2854c91b3372d2d6343497ca2160",
        urls = ["https://storage.googleapis.com/mediapipe-assets/penguins_large.jpg?generation=1686332378707665"],
    )

    http_file(
        name = "com_google_mediapipe_penguins_large_mask_png",
        sha256 = "8f78486266dabb1a3f28bf52750c0d005f96233fe505d5e8dcba02c6ee3a13cb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/penguins_large_mask.png?generation=1686332381154669"],
    )

    http_file(
        name = "com_google_mediapipe_penguins_small_jpg",
        sha256 = "708ca356d8be4fbf5b76d4f2fcd094e97122cc24934cfcca22ac3ab0f13c4632",
        urls = ["https://storage.googleapis.com/mediapipe-assets/penguins_small.jpg?generation=1686332383656645"],
    )

    http_file(
        name = "com_google_mediapipe_penguins_small_mask_png",
        sha256 = "65523dd7ed468ee4be3cd0cfed5badcfa41eaa5cd06444c9ab9b71b2d5951abe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/penguins_small_mask.png?generation=1686332385707707"],
    )

    http_file(
        name = "com_google_mediapipe_pointing_up_jpg",
        sha256 = "ecf8ca2611d08fa25948a4fc10710af9120e88243a54da6356bacea17ff3e36e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pointing_up.jpg?generation=1662650662527717"],
    )

    http_file(
        name = "com_google_mediapipe_pointing_up_landmarks_pbtxt",
        sha256 = "6bfcd360c0caa82559396d387ac30e1d59efab3b3d96b5512f4f018d0abae7c4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pointing_up_landmarks.pbtxt?generation=1692122010006268"],
    )

    http_file(
        name = "com_google_mediapipe_pointing_up_rotated_jpg",
        sha256 = "50ff66f50281207072a038e5bb6648c43f4aacbfb8204a4d2591868756aaeff1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pointing_up_rotated.jpg?generation=1666037072219697"],
    )

    http_file(
        name = "com_google_mediapipe_pointing_up_rotated_landmarks_pbtxt",
        sha256 = "cc58cbe1ead8c5051e643d2b90b77d00843cab2f1227af3489513d2b02359dd1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pointing_up_rotated_landmarks.pbtxt?generation=1692122012510778"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_expected_blendshapes_pbtxt",
        sha256 = "3f8f698d8ed81346c6f13d1cc85190fd4a58b021e664d336997d29818b8ffbb6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_expected_blendshapes.pbtxt?generation=1681322480981015"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_expected_detection_pbtxt",
        sha256 = "ace755f0fd0ba3b2d75e4f8bb1b08d2f65975fd5570898004540dfef735c1c3d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_expected_detection.pbtxt?generation=1677044311581104"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_expected_face_geometry_pbtxt",
        sha256 = "f1045ae7a479248d5c6729102401308c042068304f393934370be53587ccec9a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_expected_face_geometry.pbtxt?generation=1681322483632218"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_expected_face_landmarks_pbtxt",
        sha256 = "dae959456f001015278f3a1535bd03c9fa0990a3df951135645ce23293be0613",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_expected_face_landmarks.pbtxt?generation=1681322486192872"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_hair_expected_mask_jpg",
        sha256 = "d9ffc4f2ed0ee2d551d9239942e4dfceebf0c33a56858c84410f32ea4f0c1b2c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_hair_expected_mask.jpg?generation=1678218370120178"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_jpg",
        sha256 = "a6f11efaa834706db23f275b6115058fa87fc7f14362681e6abe14e82749de3e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait.jpg?generation=1674261630039907"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_rotated_expected_detection_pbtxt",
        sha256 = "7e680fe0918d1e8409b0e0e4576a982e20afa871e6af9c13b7a626de1d5341a2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_rotated_expected_detection.pbtxt?generation=1677194677875312"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_rotated_jpg",
        sha256 = "f91ca0e4f827b06e9ac037cf58d95f1f3ffbe34238119b7d47eda35456007f33",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_rotated.jpg?generation=1677194680138164"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_selfie_segmentation_expected_category_mask_jpg",
        sha256 = "1400c6fccf3805bfd1644d7ed9be98dfa4f900e1720838c566963f8d9f10f5d0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_selfie_segmentation_expected_category_mask.jpg?generation=1683332555306471"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_selfie_segmentation_expected_confidence_mask_jpg",
        sha256 = "25b723e90608edaf6ed92f382da703dc904a59c87525b6d271e60d9eed7a90e9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_selfie_segmentation_expected_confidence_mask.jpg?generation=1678606937358235"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_selfie_segmentation_landscape_expected_category_mask_jpg",
        sha256 = "a208aeeeb615fd40046d883e2c7982458e1b12edd6526e88c305c4053b0a9399",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_selfie_segmentation_landscape_expected_category_mask.jpg?generation=1683332557473435"],
    )

    http_file(
        name = "com_google_mediapipe_portrait_small_jpg",
        sha256 = "873a1a5e4cc86c040101362c5dea6a71cf524563b0700640175e5c3763a4246a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/portrait_small.jpg?generation=1682627845552867"],
    )

    http_file(
        name = "com_google_mediapipe_pose_detection_tflite",
        sha256 = "9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite?generation=1678737489600422"],
    )

    http_file(
        name = "com_google_mediapipe_pose_expected_detection_pbtxt",
        sha256 = "16866c8dd4fbee60f6972630d73baed219b45824c055c7fbc7dc9a91c4b182cc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_expected_detection.pbtxt?generation=1681244244235448"],
    )

    http_file(
        name = "com_google_mediapipe_pose_expected_expanded_rect_pbtxt",
        sha256 = "b0a41d25ed115757606dfc034e9d320a93a52616d92d745150b6a886ddc5a88a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_expected_expanded_rect.pbtxt?generation=1681244246786802"],
    )

    http_file(
        name = "com_google_mediapipe_pose_jpg",
        sha256 = "c8a830ed683c0276d713dd5aeda28f415f10cd6291972084a40d0d8b934ed62b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose.jpg?generation=1678737494661975"],
    )

    http_file(
        name = "com_google_mediapipe_pose_landmarker_task",
        sha256 = "fb9cc326c88fc2a4d9a6d355c28520d5deacfbaa375b56243b0141b546080596",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task?generation=1681244249587900"],
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
        sha256 = "1150dc68a713b80660b90ef46ce4e85c1c781bb88b6e3512cc64e6a685ba5588",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite?generation=1681244252454799"],
    )

    http_file(
        name = "com_google_mediapipe_pose_landmarks_pbtxt",
        sha256 = "69c79cdf3964d7819776eab1172e47e70684139d72a6d7edcbdd62dbb2ca5527",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_landmarks.pbtxt?generation=1681425322701589"],
    )

    http_file(
        name = "com_google_mediapipe_pose_segmentation_mask_golden_png",
        sha256 = "62ee418e18f317327572da5fcc988af703eb31e6f0b9e0bf3d55e6f4797d6953",
        urls = ["https://storage.googleapis.com/mediapipe-assets/pose_segmentation_mask_golden.png?generation=1682541414235372"],
    )

    http_file(
        name = "com_google_mediapipe_ptm_512_hdt_ptm_woid_tflite",
        sha256 = "2baa1c9783d03dd26f91e3c49efbcab11dd1361ff80e40e7209e81f84f281b6a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/ptm_512_hdt_ptm_woid.tflite?generation=1678323604771164"],
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
        sha256 = "4b5134daa4cb60465535239535f9f74c2842aba3aa5fd30bf04ef5678f93d87f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/right_hands.jpg?generation=1692122016203904"],
    )

    http_file(
        name = "com_google_mediapipe_right_hands_rotated_jpg",
        sha256 = "8609c6202bca43a99bbf23fa8e687e49fa525e89481152e4c0987f46d60d7931",
        urls = ["https://storage.googleapis.com/mediapipe-assets/right_hands_rotated.jpg?generation=1692122018668162"],
    )

    http_file(
        name = "com_google_mediapipe_score_calibration_csv",
        sha256 = "3ff4962162387ab8851940d2f063ce2b3a4734a8893c007a3c92d11170b020c3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/score_calibration.csv?generation=1677522780749449"],
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
        name = "com_google_mediapipe_segmentation_mask_meta_json",
        sha256 = "4294d53b309c1fbe38a5184de4057576c3dec14e07d16491f1dd459ac9116ab3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/segmentation_mask_meta.json?generation=1678818065134737"],
    )

    http_file(
        name = "com_google_mediapipe_segmenter_labelmap_txt",
        sha256 = "d9efa78274f1799ddbcab1f87263e19dae338c1697de47a5b270c9526c45d364",
        urls = ["https://storage.googleapis.com/mediapipe-assets/segmenter_labelmap.txt?generation=1678818068181025"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segm_128_128_3_expected_mask_jpg",
        sha256 = "1a2a068287d8bcd4184492485b3dbb95a09b763f4653fd729d14a836147eb383",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segm_128_128_3_expected_mask.jpg?generation=1678606942616777"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segm_128_128_3_tflite",
        sha256 = "8322982866488b063af6531b1d16ac27c7bf404135b7905f20aaf5e6af7aa45b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segm_128_128_3.tflite?generation=1678775097370282"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segm_144_256_3_expected_mask_jpg",
        sha256 = "2de433b6e8adabec2aaf80135232db900903ead4f2811c0c9378a6792b2a68b5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segm_144_256_3_expected_mask.jpg?generation=1678606945085676"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segm_144_256_3_tflite",
        sha256 = "f16a9551a408edeadd53f70d1d2911fc20f9f9de7a394129a268ca9faa2d6a08",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segm_144_256_3.tflite?generation=1678775099616375"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segmentation_landscape_tflite",
        sha256 = "a77d03f4659b9f6b6c1f5106947bf40e99d7655094b6527f214ea7d451106edd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segmentation_landscape.tflite?generation=1683332561312022"],
    )

    http_file(
        name = "com_google_mediapipe_selfie_segmentation_tflite",
        sha256 = "9ee168ec7c8f2a16c56fe8e1cfbc514974cbbb7e434051b455635f1bd1462f5c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite?generation=1683332563830600"],
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
        name = "com_google_mediapipe_ssd_mobilenet_v1_no_metadata_json",
        sha256 = "ae5a5971a1c3df705307448ef97c854d846b7e6f2183fb51015bd5af5d7deb0f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/ssd_mobilenet_v1_no_metadata.json?generation=1682456117002011"],
    )

    http_file(
        name = "com_google_mediapipe_ssd_mobilenet_v1_no_metadata_tflite",
        sha256 = "e4b118e5e4531945de2e659742c7c590f7536f8d0ed26d135abcfe83b4779d13",
        urls = ["https://storage.googleapis.com/mediapipe-assets/ssd_mobilenet_v1_no_metadata.tflite?generation=1677522790838583"],
    )

    http_file(
        name = "com_google_mediapipe_ssd_mobilenet_v1_tflite",
        sha256 = "cbdecd08b44c5dea3821f77c5468e2936ecfbf43cde0795a2729fdb43401e58b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/ssd_mobilenet_v1.tflite?generation=1661875947436302"],
    )

    http_file(
        name = "com_google_mediapipe_stablelm_3b_4e1t_test_weight_safetensors",
        sha256 = "c732deb063697cb46ad55013ed87372d57fd22b9e1cdf913a5e563601f50b7ec",
        urls = ["https://storage.googleapis.com/mediapipe-assets/stablelm_3b_4e1t_test_weight.safetensors?generation=1728509663906850"],
    )

    http_file(
        name = "com_google_mediapipe_tensor_group_meta_json",
        sha256 = "eea454ae15b0c4f7e1f84aad9700bc936627fe22a085d335a40269740bc33c69",
        urls = ["https://storage.googleapis.com/mediapipe-assets/tensor_group_meta.json?generation=1677522794324300"],
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
        sha256 = "feddaa81e188b9bceae12a96766f71e8ff3b2b316b4a31d64054d8a329e6015e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/thumb_up_landmarks.pbtxt?generation=1692122022310696"],
    )

    http_file(
        name = "com_google_mediapipe_thumb_up_rotated_landmarks_pbtxt",
        sha256 = "f0e90db82890ad2e0304af5e6e88b2e64f3774eec4d43e56b634a296553b7196",
        urls = ["https://storage.googleapis.com/mediapipe-assets/thumb_up_rotated_landmarks.pbtxt?generation=1692122024789637"],
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
        sha256 = "73fb59741872bc66b79982d4c9765a4128d6308cc5d919100615080c0f4c0c55",
        urls = ["https://storage.googleapis.com/mediapipe-assets/victory_landmarks.pbtxt?generation=1692122027459905"],
    )

    http_file(
        name = "com_google_mediapipe_vit_multiclass_256x256-2022_10_14-xenoformer_f32_tflite",
        sha256 = "768b7dd613c5b9f263b289cdbe1b9bc716f65e92c51e7ae57fae01f97f9658cd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/vit_multiclass_256x256-2022_10_14-xenoformer.f32.tflite?generation=1679955093986149"],
    )

    http_file(
        name = "com_google_mediapipe_vit_multiclass_512x512-2022_12_02_f32_tflite",
        sha256 = "7ac7f0a037cd451b9be8eb25da86339aaba54fa821a0bd44e18768866ed0205a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/vit_multiclass_512x512-2022_12_02.f32.tflite?generation=1679955096856280"],
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
        name = "com_google_mediapipe_mobilebert_tiny_keras_metadata_pb",
        sha256 = "cef8131a414c602b9d4742ac57f4f90bc5d8a42baec36b65deece884e2d0cf0f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilebert_tiny/keras_metadata.pb?generation=1673297965144159"],
    )

    http_file(
        name = "com_google_mediapipe_mobilebert_tiny_saved_model_pb",
        sha256 = "323c997cd3e17df1b2e3bdebe3cfe2b17c5ffd9488a26a4afb59ee819196837a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilebert_tiny/saved_model.pb?generation=1673297968138825"],
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
        sha256 = "acc23dee09f69210717ac060035c844ba902e8271486f1086f29fb156c236690",
        urls = ["https://storage.googleapis.com/mediapipe-assets/object_detection_saved_model/README.md?generation=1678737498915254"],
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

    http_file(
        name = "com_google_mediapipe_mobilebert_tiny_assets_vocab_txt",
        sha256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilebert_tiny/assets/vocab.txt?generation=1673297970948751"],
    )

    http_file(
        name = "com_google_mediapipe_mobilebert_tiny_variables_variables_data-00000-of-00001",
        sha256 = "c3857370046cd3a2f345657cf1bb259a4e7e09185d7f0808e57803e9d41ebba4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilebert_tiny/variables/variables.data-00000-of-00001?generation=1673297975132568"],
    )

    http_file(
        name = "com_google_mediapipe_mobilebert_tiny_variables_variables_index",
        sha256 = "4df4d7c0fefe99903ab6ebf44b7478196ce613082d2ca692a5a37a7f24e562ed",
        urls = ["https://storage.googleapis.com/mediapipe-assets/mobilebert_tiny/variables/variables.index?generation=1673297977586840"],
    )
