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
        sha256 = "15566c5600cbf7c6f61170b8017fcc682aa34c030215ab963049f0fe7c49f90d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1756258923439849"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "75e25a55ebd3acf3cccda069b243fe5aab05577adb7201f2e35c1e314b77395a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1756258926099465"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "f34289e0bf5b79c19d2d477f016acd7632fa8e10bfc6070e1dc276adb5af2443",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1756258928313592"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "6dcbf274d588ce22214b1a9f43925a37dfcc5ac9851b5df9878a7c4ef6c76df0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1756258930818788"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_js",
        sha256 = "87a3c0b2020dbed50732bf0b1640acdf98b13a3092fa148e353072bc26142fec",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.js?generation=1756258933023156"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_wasm",
        sha256 = "015fd14f511595770bbdd4c5fceff02168f3922de5d8ccfe120e2b13129ef11e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.wasm?generation=1756258935371997"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_js",
        sha256 = "7cdc5ea865654b2adad80deb094fa96760cd09062de20f5ce9519be87ddd4546",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.js?generation=1756258937678794"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_wasm",
        sha256 = "fa089aa353b2903980867fd4fc49cb478986724449e28b2df8bf6551ca0c248a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.wasm?generation=1756258940314943"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "da83c77f96d7de0dbb9de5b6e5267a193569f2e8f9d2d77765a55c22c64c4b4d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1756258942649748"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "e701adc07846951d8ffb3f28658bbc6af15aaf7230f1c85eb8349973200418a0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1756258945277286"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "3ee6cd5ac181c340db297dee87fe4475cb641f8dc5cd8fb5b1f3fdcb7cbb5ec3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1756258947625406"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "d92fa73ac208ffcdd28959ca6b90db2353ac1810f2a5e80a6de496f6ed5e89b2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1756258950306808"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "5845acbdd3439e0dd64c93bd588babb5ea475adf01b24046feb26d0505618492",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1756258952401166"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "ddc1c706a9c853a5c059a61bcde7b79309afd45e5d88809a63b6b394b30bccd6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1756258955192106"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "6ca5113240d745314c2f59158dcd5cff02bb576bc1776c0529d3952607f18862",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1756258957357136"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "ab57cff28ebdb81cfc1ebeeeb7e2153a0661ca1c9996338c453fa88a2331e7d4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1756258959966918"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "f62a5da396e1eb77028a993715679d380baa93d0d30470d495b16d93958d2e28",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1756258962100880"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "72a8e48a6043ccbdf3cc9d6254fe6dbb198c09d05c7076dceea0c3d0cd8bb2d8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1756258964605384"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "90e16389b5dc0f78a49ce9da5b0a3c4cb4ab3ef614652418a82429e39f180774",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1756258966923041"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "c178d58be1a9e9a4fac91ab2f9f44e62e2a43793fdd8f7e42a2ce8a9e53aa53c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1756258969590559"],
    )
