# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

package(
    default_visibility = ["//mediapipe/tasks:internal"],
    licenses = ["notice"],  # Apache 2.0
)

mediapipe_files(srcs = [
    "30k-clean.model",
    "bert_text_classifier_no_metadata.tflite",
    "category_tensor_float_meta.json",
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_no_metadata.tflite",
    "coco_ssd_mobilenet_v1_score_calibration.json",
    "deeplabv3.json",
    "deeplabv3_with_activation.json",
    "deeplabv3_without_labels.json",
    "deeplabv3_without_metadata.tflite",
    "dummy_face_stylizer.tflite",
    "efficientdet_lite0_fp16_no_nms.tflite",
    "efficientdet_lite0_v1.tflite",
    "face_stylizer.json",
    "labelmap.txt",
    "mobile_ica_8bit-with-custom-metadata.tflite",
    "mobile_ica_8bit-with-large-min-parser-version.tflite",
    "mobile_ica_8bit-with-metadata.tflite",
    "mobile_ica_8bit-with-unsupported-metadata-version.tflite",
    "mobile_ica_8bit-without-model-metadata.tflite",
    "mobile_object_classifier_v0_2_3-metadata-no-name.tflite",
    "mobilebert_vocab.txt",
    "mobilenet_v1_0.25_224_1_default_1.tflite",
    "mobilenet_v2_1.0_224_quant.tflite",
    "mobilenet_v2_1.0_224_quant_without_metadata.tflite",
    "mobilenet_v2_1.0_224_without_metadata.tflite",
    "movie_review.tflite",
    "score_calibration.csv",
    "segmentation_mask_meta.json",
    "segmenter_labelmap.txt",
    "ssd_mobilenet_v1_no_metadata.json",
    "ssd_mobilenet_v1_no_metadata.tflite",
    "tensor_group_meta.json",
])

exports_files([
    "external_file",
    "general_meta.json",
    "golden_json.json",
    "associated_file_meta.json",
    "bounding_box_tensor_meta.json",
    "classification_tensor_float_meta.json",
    "classification_tensor_uint8_meta.json",
    "classification_tensor_unsupported_meta.json",
    "efficientdet_lite0_fp16_no_nms_anchors.csv",
    "efficientdet_lite0_fp16_no_nms.json",
    "feature_tensor_meta.json",
    "image_tensor_meta.json",
    "input_image_tensor_float_meta.json",
    "input_image_tensor_uint8_meta.json",
    "input_image_tensor_unsupported_meta.json",
    "score_calibration.txt",
    "score_calibration_file_meta.json",
    "score_calibration_tensor_meta.json",
    "score_thresholding_meta.json",
    "labels.txt",
    "mobilenet_v2_1.0_224.json",
    "mobilenet_v2_1.0_224_quant.json",
    "input_text_tensor_meta.json",
    "input_text_tensor_default_meta.json",
    "movie_review_labels.txt",
    "regex_vocab.txt",
    "movie_review.json",
    "bert_tokenizer_meta.json",
    "bert_text_classifier_with_sentence_piece.json",
    "sentence_piece_tokenizer_meta.json",
    "bert_text_classifier_with_bert_tokenizer.json",
])

filegroup(
    name = "model_files",
    srcs = [
        "30k-clean.model",
        "bert_text_classifier_no_metadata.tflite",
        "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29_no_metadata.tflite",
        "deeplabv3_without_metadata.tflite",
        "dummy_face_stylizer.tflite",
        "efficientdet_lite0_fp16_no_nms.tflite",
        "efficientdet_lite0_v1.tflite",
        "mobile_ica_8bit-with-custom-metadata.tflite",
        "mobile_ica_8bit-with-large-min-parser-version.tflite",
        "mobile_ica_8bit-with-metadata.tflite",
        "mobile_ica_8bit-with-unsupported-metadata-version.tflite",
        "mobile_ica_8bit-without-model-metadata.tflite",
        "mobile_object_classifier_v0_2_3-metadata-no-name.tflite",
        "mobilenet_v1_0.25_224_1_default_1.tflite",
        "mobilenet_v2_1.0_224_quant.tflite",
        "mobilenet_v2_1.0_224_quant_without_metadata.tflite",
        "mobilenet_v2_1.0_224_without_metadata.tflite",
        "movie_review.tflite",
        "ssd_mobilenet_v1_no_metadata.tflite",
    ],
)

filegroup(
    name = "data_files",
    srcs = [
        "associated_file_meta.json",
        "bert_text_classifier_with_bert_tokenizer.json",
        "bert_text_classifier_with_sentence_piece.json",
        "bert_tokenizer_meta.json",
        "bounding_box_tensor_meta.json",
        "category_tensor_float_meta.json",
        "classification_tensor_float_meta.json",
        "classification_tensor_uint8_meta.json",
        "classification_tensor_unsupported_meta.json",
        "coco_ssd_mobilenet_v1_score_calibration.json",
        "deeplabv3.json",
        "deeplabv3_with_activation.json",
        "deeplabv3_without_labels.json",
        "efficientdet_lite0_fp16_no_nms_anchors.csv",
        "efficientdet_lite0_v1.json",
        "external_file",
        "face_stylizer.json",
        "feature_tensor_meta.json",
        "general_meta.json",
        "golden_json.json",
        "image_tensor_meta.json",
        "input_image_tensor_float_meta.json",
        "input_image_tensor_uint8_meta.json",
        "input_image_tensor_unsupported_meta.json",
        "input_text_tensor_default_meta.json",
        "input_text_tensor_meta.json",
        "labelmap.txt",
        "labels.txt",
        "mobilebert_vocab.txt",
        "mobilenet_v2_1.0_224.json",
        "mobilenet_v2_1.0_224_quant.json",
        "movie_review.json",
        "movie_review_labels.txt",
        "regex_vocab.txt",
        "score_calibration.csv",
        "score_calibration.txt",
        "score_calibration_file_meta.json",
        "score_calibration_tensor_meta.json",
        "score_thresholding_meta.json",
        "segmentation_mask_meta.json",
        "segmenter_labelmap.txt",
        "sentence_piece_tokenizer_meta.json",
        "ssd_mobilenet_v1_no_metadata.json",
        "tensor_group_meta.json",
    ],
)
