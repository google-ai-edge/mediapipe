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
        sha256 = "da368a61eb6f6688963cecdb2b1565c6c270dac766d9822486030620d2fc6602",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1756508430946499"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "9b1bc345aeed7b7b020a982a830c05ef777e7e7d1ecc47dd11fb1844f9fe4fa4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1756508433346511"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "dade90fdc0f0ba851275f7344303147c9c2c82708e098550614f445b2d1dceb8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1756508435729071"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "c4732e1d3dd285936eb06627cfe5eb23359d3c5d8004e6ef209682a6f45496aa",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1756508438195473"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_js",
        sha256 = "83550b35ca8a3e56bb1a95ca59e24fc734c9148f998bb6396cd2f16ea03684b6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.js?generation=1756508440318514"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_wasm",
        sha256 = "e10e40bd88074070404484e12756c83bd560e6b9e9d22b35ba68c360802ed714",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.wasm?generation=1756508442878231"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_js",
        sha256 = "d99d40d5806103db07787c8189524da27f491be7f013ed47857784438d656373",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.js?generation=1756508445216897"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_wasm",
        sha256 = "81875c19b521974e605c4bdc6bfaf7c6dd459834b81c13c10e4c5ed901d6a2e6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.wasm?generation=1756508447682982"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "bfcfe934d2afdf3b27f122abcc72d1ad3fbf1b4606ecce1a7e3f7747af1a657f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1756508449957406"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "685457c03d5d4ba9aa2329c5e750d4bc4824ceb710d3cd69b3dda96186b49a42",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1756508452530679"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "74c684eb804366c21be9f6afa982db4308911020a7fad63d0c8fe6510876d899",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1756508454882830"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "7db9a17f114bb21ea2d5e3ad1743c45f652eca37f94929b6cc9fbe6214e9efc7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1756508457348828"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "9dbc2c889ecb5efd2581d016fca19806692adae1f73fe9c69bf828533abc8630",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1756508459391727"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "4766da2e68c58d2b6145997ad74ef3abde8bb074116a4dedef9d44941a7bd440",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1756508461895267"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "ca263e909d0f5786d52dc6246e50395a2f72f365d4423afaa61300e4671d88ac",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1756508464369616"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "0cd44a749115046eb8093f4831a457898cf1a606d290588863c04f23714ef4ec",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1756508466644124"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "7e9249f278dd9919d08583c819add8aff657f6a323293018b1582900e2286bf3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1756508468870024"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "e2a53eb94fb6fdeb2decd72bfa68b681f9e96b670c6a2b87a1c497708a13cf8a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1756508471472096"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "df746a335c15e51bcc75e512230c030de7099b1233e2392b523798f3b5c74121",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1756508473806056"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "a3b7ec0991e015100acb6277cf12d89d06a92cbf9f174a10659ee7ba0d9a0e73",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1756508476082751"],
    )
