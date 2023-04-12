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
        sha256 = "0eca68e2291a548b734bcab5db4c9e6b997e852ea7e19228003b9e2a78c7c646",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1681328323089931"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "69bc95af5b783b510ec1842d6fb9594254907d8e1334799c5753164878a7dcac",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1681328325829340"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "88a0176cc80d6a1eb175a5105df705cf8b8684cf13f6db0a264af0b67b65a22a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1681328328330829"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "1cc0c3db7d252801be4b090d8bbba61f308cc3dd5efe197319581d3af29495c7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1681328331085637"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "d9cd100b6d330d36f7749fe5fc64a2cdd0abb947a0376e6140784cfb0361a4e2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1681328333442454"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "30a2fcca630bdad6e99173ea7d0d8c5d7086aedf393d0159fa05bf9d08d4ff65",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1681328335803336"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "70ca2bd15c56e0ce7bb10ff2188b4a1f9eafbb657eb9424e4cab8d7b29179871",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1681328338162884"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "8221b385905f36a769d7731a0adbe18b681bcb873561890429ca84278c67c3fd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1681328340808115"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "07692acd8202adafebd35dbcd7e2b8e88a76d4a0e6b9229cb3cad59503eeddc7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1681328343147709"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "03bf553fa6a768b0d70103a5e7d835b6b37371ff44e201c3392f22e0879737c3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1681328345605574"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "36697be14f921985eac15d1447ec8a260817b05ade1c9bb3ca7e906e0f047ec0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1681328348025082"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "103fb145438d61cfecb2e8db3f06b43a5d77a7e3fcea940437fe272227cf2592",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1681328350709881"],
    )
