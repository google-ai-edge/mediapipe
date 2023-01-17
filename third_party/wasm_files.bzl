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
        sha256 = "d4d205d08e3e1b09662a9a358d0107e8a8023827ba9b6982a3777bb6c040f936",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1673996821002628"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "1b2ffe82b0a25d20188237a724a7cad68d068818a7738f91c69c782314f55965",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1673996823772372"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "1f367c2d667628b178251aec7fd464327351570edac4549450b11fb82f5f0fd4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1673996826132845"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "35c6ad888c06025dba1f9c8edb70e6c7be7e94e45dc2c0236a2fcfe61991dc44",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1673996828935550"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "68c0134e0b3cb986c3526cd645f74cc5a1f6ab19292276ca7d3558b89801e205",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1673996831356232"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "df82bb192ea852dc1bcc8f9f28fbd8c3d6b219dc4fec2b2a92451678d98ee1f0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1673996834657078"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "de1a4aabefb2e42ae4fee68b7e762e328623a163257a7ddc72365fc2502bd090",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1673996837104551"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "828dd1e73fa9478a97a62539117f92b813833ab35d37a986c466df15a8cfdc7b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1673996840120504"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "c146b68523c256d41132230e811fc224dafb6a0bce6fc318c29dad37dfac06de",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1673996842448396"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "8dbccaaf944ef1251cf78190450ab7074abea233e18ebb37d2c2ce0f18d14a0c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1673996845499070"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "705f9e3c2c62d12903ea2cadc22d2c328bc890f96fffc47b51f989471196ecea",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1673996847915731"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "c7ff6a7d8dc22380e2e8457a15a51b6bc1e70c6262fecca25825f54ecc593d1f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1673996850980344"],
    )
