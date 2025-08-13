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
        sha256 = "c0705ec9ef42b0a9e72c23f71e6af0b524a0c403647be5f4ce15cbf486b1f12a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1755122368613769"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "a64c6d7baf37266452c7b8fc57a07e94abe5097b43e6b90e37a7de2e959bd638",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1755122371106917"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "3e4d0e18eb011b7ed2757a5adf3715c6b9b6459c3218b4d4934351d5697be974",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1755122373382926"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "d666e93d8f60b38864a547a24769dd93170d3f4e367dedb3c39e7e3c253396a7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1755122375704272"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_js",
        sha256 = "7720793391d1046129d2c6d88dbdeb6fefe76f150dfcb3b9488070c4da79178f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.js?generation=1755122377950524"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_wasm",
        sha256 = "78e703ef5cc854166ed113d703dbd4e99e3f9b7f6b0e270a02071c0253fb0dbe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.wasm?generation=1755122380219873"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_js",
        sha256 = "243ec59a6d2542145d08debe7b97b5f9e7a8dbacbd169629da5a19b3d94a03e1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.js?generation=1755122382557375"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_wasm",
        sha256 = "f75d33efd6608ec8c20a719b38ce7a17fddd45a81215951a3be7f7751b2d1d55",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.wasm?generation=1755122384990252"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "9b65af0b4b45274e6e4c99ad3a77e64f73eec7058b7ad980509df1f969fe0e58",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1755122387218148"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "5649baed1d3a22225f568e28041aa0729df48d87d782187eef923bf8b8222a4c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1755122389811567"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "e5efb3057b92fdef5be6d9bf6565b5928c6ac3af90d643dac121e764a340fbb1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1755122391946343"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "efd4054d52e03b320f9cbd819a0bee4c3e4ae31505d3964cb0e3bc75203544ee",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1755122394564053"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "09f4d1ccda541a90114e7934dffff650219aacf5a59ec5da4eee39ca9771bb30",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1755122396783579"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "d59377fc18e2f01e24e66b5dfa95fd945496fc47b050374585ae8da563d48ea0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1755122399257354"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "cf504cbac185225f044972a1c5c5d6a60b5c5eb05aa062f9c4bdedcb97e14e83",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1755122401510962"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "351aa03ad33c74bf5aaa65f85ab310bf664750ddbc52f2fff24b2632bce83c15",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1755122403991885"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "e351a1e5f684b0b5bc60164e49f0decd1957050ab460e777232844a6858196df",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1755122406172283"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "bc715c9c23f070e03481cf4c21d30aa8073ab5cad10f3185ea526d45ff39035e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1755122408665167"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "8f361faf34730551dea36f92894514b80307abc9594419855cadc8f7a1a218d7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1755122410969663"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "c135275f2b23043b44503e6c4760fff04991e449bc5d3c5e9e9016c70b977099",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1755122413523924"],
    )
