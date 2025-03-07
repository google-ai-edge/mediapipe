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
        sha256 = "03d111f6c2c1b4f2f78310ebaf5e224dda25eb6e3e4d84dc0ae6e3395da37f26",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1741382337224335"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "c6f967cde1b25420ac349bf1846ad0e5acc8713940c49a20b57fb7bcff01ac25",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1741382339492142"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "dd3ccf8e1cb526a1c51349232c5f9edf077d9983cfb114f48c42d923f223c6a2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1741382341463399"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "6621bcfb3bb727040b1f936c339d673c9d5df95899b2ccd0ddd2dc2ca546acdb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1741382343622086"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_js",
        sha256 = "2b7c16325b1f049fd860580d6291915ae4da2eb77c0e101ed998e583955db7f8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.js?generation=1741382345602227"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_wasm",
        sha256 = "dd69422e0f1d43f26e533e501e04e4bc6ecc77cd36fcb591488c4bb8f8bb8e79",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.wasm?generation=1741382347792494"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_js",
        sha256 = "c912343aa1454ff7cd5dbb6cd209256e53d1514226a44f224cec01a28b7d7872",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.js?generation=1741382349668830"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_wasm",
        sha256 = "cdad2078c39f60747aa60aa8995b5f2016d3aae983bea56d4644124c960c95fd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.wasm?generation=1741382351833418"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "4daeed5b60335fb8c9dcc89793e2f3e95ff58e26fa1163ae57423e39ecc9ab65",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1741382353743580"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "23f18fbafdbc4d4ce7d17ef27a0332113ebd397c0595a8e24ba3ac89a9fb2f6b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1741382355931837"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "e91e65c578a4b2d5a39fd0efa4f30a9b86724cc83cdd3ddb2eed1ce7d5c4f4df",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1741382357889791"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "1de0740afa5cdfe8b92edb70fb52916c4eedb98f33277a6ee7d693159d978c62",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1741382360033762"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "8ec3392ac489dc67fd411d6959d77f73a48473b5fb64c328c50f732b2dbc1f1b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1741382362011490"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "ff25abc78f4cf0a1b87156c88a87cb23fd7ab62d46be27100352f1b78c5cf318",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1741382364086370"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "6a4040131965fe441b9396f4be800bc045e422bedbebd2e4274e460c06c3a67b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1741382366053787"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "9818468e519145bcda719f443ed8079b34f3f7351b44684de68b52fc80c5f6a8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1741382368042528"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "acc5ad448e75078f81a07754241589e13e27dbf1fe7e343313bbaea2665b1393",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1741382369988928"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "bf85dc00fd2e3da1f170b36da41467b44156ca7f137310c08511963f5854386a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1741382372072673"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "f7992c8d6621098e58b39b8cac311733b7a2599b7c4695b2043bfcfa9344ccff",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1741382373995131"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "6cc46a5c69ec5b4d88a6df9f0fd94daecd3731d2adb5198ca4eeca21cf085421",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1741382376022583"],
    )
