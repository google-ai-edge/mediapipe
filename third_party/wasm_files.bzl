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
        sha256 = "941b66518727f28f1afb3178a85a19bd494932700e803f0a612820af64b00365",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1747275568136480"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "f27751f01ffc5c3e3e942f048185b5e1c7aab710173e8c41803761a024c8e0da",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1747275570492470"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "8bb1880a37f87d374bee540f615c1ec7cc7e9222cdf7f54f5f431aaa3cd3bc37",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1747275572540382"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "f5a3afccbc04b3049bdaa63755c9f3c0fc2e0f4db27959d24c74b60d422c3d09",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1747275574905281"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_js",
        sha256 = "1f1c3e46898536b124f0da469ac2d9aa5b4954f6a4365266563255c2fb5559e8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.js?generation=1747275576965354"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_wasm",
        sha256 = "3a16fcd15fbe52c21a735055f3c8d054c92ca818fd239664313a9aa0c36bc89f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.wasm?generation=1747275579511311"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_js",
        sha256 = "00d9060f9c7b286ef1f31bebbca6e39e55ecb9157ee9d5b5933f8321b397636a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.js?generation=1747275581542191"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_wasm",
        sha256 = "aa9c0d916ca7e99b1b8dab57532640f8afcadac48abddc377dacacc817630193",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.wasm?generation=1747275583843570"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "5805496bf9b006664f6934eee13ce8120937e85807f0f4812324e55d49df9a41",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1747275585858717"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "777e96a3a4a7daa8ed8288f0b7f7c500a0523f0d753101dc6cc814113c96e0ec",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1747275588399493"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "afe95a4653d90b4abe31ff6c56a401dcd3090781f2c581e504b7a723d81cd762",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1747275590434871"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "d062fa835fe6c9eb5744a678baa368590771b6dffaedff7829f8ab677f785ff7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1747275592866651"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "9daa4197ce4cabb2c6bced8b23f2a59db8c829e06a32d005657e9644654650c7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1747275594913513"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "54ddd52a6695f3d95d7539a8ff27cf39a6c5225269caf3844ce72a34b77929c2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1747275597228622"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "45a609e4bf23d9c48af9b8474682b5c73bf8c228aa6719c53b8e1b27155d9fe4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1747275599316019"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "bd4b99614249e6f6b49e5cfafe8e713cdee9d18d125840ebc135976857b8fd9e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1747275601728329"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "6edca29f89293197186428ee71e0a7573d6dff843fc9d0617ed73b4bd356ea9f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1747275603896816"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "e90cf6af0b392b7b705e4c468fc7b82577dea0d78513418062e81607d9c5692a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1747275606379153"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "7ddac04088972cf641d9235b6f27497d24202de3a2ebdd3fba2225fb3b35d7b6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1747275608589791"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "2862ebb173981fd5872983878908d9d9d6e00c042919f7d4ec8d530b6e6cdfb2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1747275610863577"],
    )
