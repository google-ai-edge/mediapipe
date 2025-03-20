package(
    default_visibility = ["//visibility:public"],
    features = [
        "layering_check",
        "parse_headers",
    ],
)

licenses(["notice"])  # Apache 2, BSD, MIT

proto_library(
    name = "sentencepiece_proto",
    srcs = ["sentencepiece/src/sentencepiece.proto"],
)

cc_proto_library(
    name = "sentencepiece_cc_proto",
    deps = [":sentencepiece_proto"],
)

proto_library(
    name = "sentencepiece_model_proto",
    srcs = ["sentencepiece/src/sentencepiece_model.proto"],
)

cc_proto_library(
    name = "sentencepiece_model_cc_proto",
    deps = [":sentencepiece_model_proto"],
)

genrule(
    name = "config_h",
    srcs = ["sentencepiece/config.h.in"],
    outs = ["sentencepiece/config.h"],
    cmd = "cp $< $@",
)

cc_library(
    name = "common",
    hdrs = [
        "sentencepiece/config.h",
        "sentencepiece/src/common.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "sentencepiece_processor",
    srcs = [
        "sentencepiece/src/bpe_model.cc",
        "sentencepiece/src/char_model.cc",
        "sentencepiece/src/error.cc",
        "sentencepiece/src/filesystem.cc",
        "sentencepiece/src/model_factory.cc",
        "sentencepiece/src/model_interface.cc",
        "sentencepiece/src/normalizer.cc",
        "sentencepiece/src/sentencepiece_processor.cc",
        "sentencepiece/src/unigram_model.cc",
        "sentencepiece/src/util.cc",
        "sentencepiece/src/word_model.cc",
    ],
    hdrs = [
        "sentencepiece/src/bpe_model.h",
        "sentencepiece/src/char_model.h",
        "sentencepiece/src/filesystem.h",
        "sentencepiece/src/freelist.h",
        "sentencepiece/src/model_factory.h",
        "sentencepiece/src/model_interface.h",
        "sentencepiece/src/normalizer.h",
        "sentencepiece/src/sentencepiece_processor.h",
        "sentencepiece/src/trainer_interface.h",
        "sentencepiece/src/unigram_model.h",
        "sentencepiece/src/util.h",
        "sentencepiece/src/word_model.h",
    ],
    defines = ["_USE_TF_STRING_VIEW"],
    includes = [
        "sentencepiece/",
        "sentencepiece/src",
    ],
    linkstatic = 1,
    deps =
        [
            ":common",
            ":sentencepiece_cc_proto",
            ":sentencepiece_model_cc_proto",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/container:flat_hash_set",
            "@com_google_absl//absl/memory",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/strings:str_format",
            "@darts_clone",
        ],
)

cc_library(
    name = "sentencepiece_trainer",
    srcs = [
        "sentencepiece/src/bpe_model_trainer.cc",
        "sentencepiece/src/builder.cc",
        "sentencepiece/src/char_model_trainer.cc",
        "sentencepiece/src/sentencepiece_trainer.cc",
        "sentencepiece/src/trainer_factory.cc",
        "sentencepiece/src/trainer_interface.cc",
        "sentencepiece/src/unicode_script.cc",
        "sentencepiece/src/unigram_model_trainer.cc",
        "sentencepiece/src/word_model_trainer.cc",
    ],
    hdrs = [
        "sentencepiece/src/bpe_model_trainer.h",
        "sentencepiece/src/builder.h",
        "sentencepiece/src/char_model_trainer.h",
        "sentencepiece/src/normalization_rule.h",
        "sentencepiece/src/sentencepiece_trainer.h",
        "sentencepiece/src/spec_parser.h",
        "sentencepiece/src/trainer_factory.h",
        "sentencepiece/src/trainer_interface.h",
        "sentencepiece/src/unicode_script.h",
        "sentencepiece/src/unicode_script_map.h",
        "sentencepiece/src/unigram_model_trainer.h",
        "sentencepiece/src/word_model_trainer.h",
        "sentencepiece/third_party/esaxx/esa.hxx",
        "sentencepiece/third_party/esaxx/sais.hxx",
    ],
    includes = [
        "sentencepiece/",
        "sentencepiece/src",
        "sentencepiece/third_party/esaxx",
    ],
    deps = [
        ":common",
        ":pretokenizer_for_training",
        ":sentencepiece_cc_proto",
        ":sentencepiece_model_cc_proto",
        ":sentencepiece_processor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@darts_clone",
    ],
)

cc_library(
    name = "pretokenizer_for_training",
    srcs = ["sentencepiece/src/pretokenizer_for_training.cc"],
    hdrs = ["sentencepiece/src/pretokenizer_for_training.h"],
    includes = [
        "sentencepiece/",
        "sentencepiece/src",
    ],
    deps = [
        ":common",
        ":sentencepiece_cc_proto",
        ":sentencepiece_processor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
