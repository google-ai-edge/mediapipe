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
        sha256 = "f5e307789bb80892efa561cd25c6c8b52df16d508f88bf27d02561df620dfa59",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1773692449388786"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "f3d2cd1410bbb6d6ebcd45722376dba7691207a71685dc534dad32f5646f68ce",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1773692452382867"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_js",
        sha256 = "690966e178a58c2130f09ed0b0bcdbfa11bfeb9de464ef0a247fcc3ff9700933",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.js?generation=1773692455086030"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_wasm",
        sha256 = "8669180e1f323ee9ac4b7b21df703a6819e6b6d6113fb2cc7345b570a1c72308",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.wasm?generation=1773692458208506"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "87fe3776f37005e7a73a27675b7e711c71f22fec367bac5c05e41cf70675ef77",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1773692460995410"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "4f7d607ca94d9255eeefc808e0bc8fae54266247de97052658106242f6b5cc5d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1773692464104724"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "f0ceb4bf39f7bcbf2b79e0d0554874db3121dfa304bcd61abb50bd66e53e855f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1773692466937789"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "5e371744198dc16ecef4775aafde0405585daa90a7cdfc75a326408c692864ce",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1773692470366538"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_js",
        sha256 = "7a68b22765210e235cc1e1425b2ba1467a3f38d1e8206bea998709e9937629a3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.js?generation=1773692473096494"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_wasm",
        sha256 = "24c0651a37f0227a1b1ec0bd9d6b31e64eb741e55e4fad4d3214ca27c3ffe75e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.wasm?generation=1773692476224744"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "8281465403b3409fd64ce8d2890a74a0fafcb6fdda18723f58876ee5e533ac12",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1773692479190041"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "fa03103eeea1c9850f98f6b85c64a8dceb6abdca389b4509b7fca510ef0c1814",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1773692482221141"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "4b51fdf2338f38d216e7c9c9ace1250307afc7f34968339e6a8a721ef565196f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1773692485203168"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "ab2e13f037f5ed53a683dc379c78991e54d53360df980d199c34451067e8e10d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1773692488388298"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_js",
        sha256 = "d982f327627a870b4355ce9402800b6fe1b210ddfefaa12b00c26928e6648b8a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.js?generation=1773692491194214"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_wasm",
        sha256 = "81075debf15b7b4287e030b6b680060fba6517a2a97c72860cd4a8494e9070d8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.wasm?generation=1773692494296331"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "efe4a067a6ad4be357f6a5e1eca81fc899313f5b9e034c873648e055621f7224",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1773692496906850"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "e83311759e2216d38c0d3c64a445a1e9d8e08aaf82c129e5c6d5acd5c7a9ea1a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1773692499934670"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "c5edaa79ac6d1c0d7da35887a317724492f1ad4ee73a8945a072f2a931fbfa18",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1773692502732307"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "2ba284fbadd6a17adf404ea0ffdfe03d2ebd196eee297e39060233a49e53c367",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1773692505931446"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_js",
        sha256 = "f530dfcd275c75950563c37a078106820f0fdf51c13c11b712f2c629a496afdd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.js?generation=1773692508652455"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_wasm",
        sha256 = "3d138bdf81c28a337db6eb38e388eaee5bab55048cc205ccc82f41bb241950a9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.wasm?generation=1773692511913832"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "5f829d80bdfafef0c6857a3277671039e1a90b1ee0203452dae1c03d1432efce",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1773692518255667"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "b5a2226dbc3b4917dee0e64fe51f0c8e8d72452ec312409471204adb8cc4ce8b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1773692521221063"],
    )
