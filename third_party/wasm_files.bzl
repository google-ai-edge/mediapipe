"""
WASM dependencies for MediaPipe.

This file is auto-generated.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

# buildifier: disable=unnamed-macro
def wasm_files():
    """WASM dependencies for MediaPipe."""

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_js",
        sha256 = "fda8a6c1512ddfa63880990bfc14bca9e47d2f309cde337868c27124c2044516",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1781557373929320"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "a6a38c9f9b057a75bfa16173ee2dafb060bd3c23ebf311f39e27e7a323f9f235",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1781557378343964"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_js",
        sha256 = "ad8466731e04c8d55514150c20959d1ebd9634b5fe45cf0ad115c28d68e7b466",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.js?generation=1781557382472138"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_wasm",
        sha256 = "a5b464c00086578eedc6d05da1cbe3d6c0c117746fcbd3edf3f598c166dd14ab",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.wasm?generation=1781557386706122"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "af4923c2e1b2552f8b4400f98ca02481127c4a684f74f7a8006f1f370316c48a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1781557391201162"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "78c205cee57a56157bb8fbc2e6aeecfba7789dc8991f1685fc56b00f97f34356",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1781557395522489"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "01371bd099a788c036016174a6fb9ec99931490daaaa68142d781698b0c2cee4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1781557399623494"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "4c49c39ce65546e4774e8215823413213ba1ab444189d2b89549d82c085cfa1f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1781557404160361"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_js",
        sha256 = "5dd3ff13e0a9f1da07c75eeb02b13c368c32d681a8ce7edbfc5b2531e8fb6810",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.js?generation=1781557408667855"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_wasm",
        sha256 = "a94ffe8ace90a82b4417896a99fd5baaabbe2e5021fad093805285510adae0c2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.wasm?generation=1781557413039171"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "7ef03a0428323660ee5c8089737888a641a349f0562a5707d5b869569ea0d24a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1781557417048030"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "6db89f6ff32e1dd8955ec4dc3fd6965d11079e8da2f0ecf204fbfa00049fca20",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1781557421363705"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "c8f7b513ca6608dbaef8f321c0a18c18c1dcadc804c5fa2193399d846dbdd8a2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1781557425429857"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "9bac065e3a9955d13a1e5130bba34fb7869bda0e0470c0b02922a29291300476",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1781557429618888"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_js",
        sha256 = "77c4509bd6dbd2f751f2b6d183bd3602ce82303e30c1750f5c07b5ccce8aa63f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.js?generation=1781557433765099"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_wasm",
        sha256 = "bdd3b1e43fef2e8bad80d5a14d983db418d3663a0f271055ecbde0466b4edf94",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.wasm?generation=1781557438102677"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "de8c234dd628db76ce3a4c0cfbca2579d9de5784724876575c3b2118a1b82d08",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1781557442083183"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "05096f89aa099c7e21e2e70a0bc6b06d0a63ac5f90b20c7a61ea9504b11a94a3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1781557446260400"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "4f2ab9632bba06f168386ce3e38a3a0250460890c48c12891a8cea13690c4d98",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1781557450288455"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "1aeef7bc05036eab5d9c575ad9396381109b61cb98abb1d4ae7c146277fad8ed",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1781557454757187"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_js",
        sha256 = "e02387f0021cd4cd8d98f5e3eb9cbf9e00fd7abcffd0ad1ff4e7551cc3ef27e0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.js?generation=1781557458666946"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_wasm",
        sha256 = "4371ac4c70741a439e392de8e05cac5584514c5f5f5ad1216114ff3cfe0f853a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.wasm?generation=1781557462868728"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "9c0697b78769ab751d1a243a1c3c682953ae3c234a93ce0e0dce24231dab191e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1781557466924790"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "1e2cd3dd75d3bd93df303fb67c07a59d6e53742c12d90814bb63375db4b0c79f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1781557471208801"],
    )
