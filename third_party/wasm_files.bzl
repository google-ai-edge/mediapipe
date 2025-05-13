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
        sha256 = "62ff7e6f501b5d9faf85b2bc7a88802e54b461395904175b0bafa883c6ccb487",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1747160966694468"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "c57a8bc1b30958a9ba4fc0e863c843d88476e870e8e4d0e11c5e564232be6c6b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1747160969168960"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "9531f244a410cdb8985ce37c70d6b1c7563de66130669db074eccf5e6050958f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1747160971354334"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "f60254e26da39732e46d7166e6414600e43a9ca5f484742a76e00a3171c4fba8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1747160973944575"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_js",
        sha256 = "38498761e8db44321e05fb4ca9d1bee48265ce5086eddaf7f1356195cb048973",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.js?generation=1747160976216856"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_internal_wasm",
        sha256 = "edbe4c94e48640206107f4fa4c03d4536d7df3b9796d6ec872527b763d7b5388",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_internal.wasm?generation=1747160978486990"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_js",
        sha256 = "8cd2d5c958a3623b6376d90a62c3629b26c2f0d129384f92312aedde8a4a89eb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.js?generation=1747160980784897"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_experimental_wasm_nosimd_internal_wasm",
        sha256 = "fa77b3a865c60a5d8af85c9f09f227ecb087a323868f6a21fe4f81b8f12df75b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_experimental_wasm_nosimd_internal.wasm?generation=1747160983010772"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "0d4f999a67b34243f5a4b83924163e56a4fe6e20d146343099a83f62abb7853f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1747160985070341"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "314c3bce722b72d9d9c7e2311ddeae5e7749a70a92f9a84a52b5d51d706efd7a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1747160987441332"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "4d53f15cf9abf0d75d20351e0ea909aac894d93f8be56e5fda174e0a92eb8240",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1747160989514471"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "89023c90b6155c338caf7588fdce16a6f5effd389aca3c7eda5ecf8734006489",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1747160992010902"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "3207990ec4ee1ca24dd7674b56ce21d9385e40885f5ae0cc6b4ac49c9959fa09",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1747160994057034"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "f292619cd258ca010ce3d3dadb7043dd91107908bad2f5f85e6e715854b5abf2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1747160996375213"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "30ddf651fa462d634626d95ea630e6b7c2b091d72813e8cba5e3d35a042c9314",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1747160998346654"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "aadb43279a26c46cbe8c4a64ebb781538a4630946a9d09c9675db32ce6c012bc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1747161000617246"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "6255342e9aa07d2b1ed215f9c9c5469440a12591a930caa64157d37d713c88cc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1747161002644934"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "722cb23071efa21600293b53da88fdeba9d844f3657886639389044256b995f0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1747161004910472"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "6d801c26e48b080a1512921e32d8220251332379c774522bd98a3c8a203aca71",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1747161006946194"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "718520ef2ba02343a7be0ef9a889e286b4d4cd5d83a1b69e9e36ceaeba2dd2b9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1747161009256164"],
    )
