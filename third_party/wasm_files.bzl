"""
WASM dependencies for MediaPipe.

This file is auto-generated.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

# buildifier: disable=unnamed-macro
def wasm_files():
    """WASM dependencies for MediaPipe."""

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_wasm",
        sha256 = "17062f29969b38100baa76521decf4be75a3e8df1201ce4ae21d3e8ed319befa",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.wasm?generation=1782327827586589"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_js",
        sha256 = "62a15778221ae85cea1c59501c5b52e0f9a332f7c12a88875c5b38f0300ac5ce",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.js?generation=1782327831392996"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "a03feab54af4810811142cbff909b6c0df252a9fa478340bf8ae3a5a10f731a4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.wasm?generation=1782327835256334"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "c5ef1af8a54ad19e2664e1ec17008c3ad8ec1b5808565fd23d915d31a1aafb31",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.js?generation=1782327839171871"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_wasm",
        sha256 = "d75b4488b2008ad8fafbed30a243246e1d2f41f683a45759dce69b283607bc88",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.wasm?generation=1782327843053174"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_js",
        sha256 = "daa9b82ee8b0ef602bdb2cbe00bfa5c833322677fb18dde1352fa92ef44046d7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.js?generation=1782327846677923"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_wasm",
        sha256 = "ae0eb6fd74e0183d8f100c6d04844d1c14dd67ca1a131f978cc2d06232ac3238",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.wasm?generation=1782327850419974"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_js",
        sha256 = "a9d56ad2ac9fb2e8759aee4b4c0c114b05ec83f5327e935cce982f89b77feb80",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.js?generation=1782327854137327"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "03d34734479dd958ade455c2a6229541a4e78568ef143d7676eaf0b5be086648",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.wasm?generation=1782327857831322"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_js",
        sha256 = "42d327df07489abd53f8e68090d79c5945245ead3e98432ae05e1aabf7bc5651",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.js?generation=1782327861358301"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_wasm",
        sha256 = "d8a38b1a8e51df760d34b54f3113c9b3e6e8d9e1da120af17bb88d7e15afb826",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.wasm?generation=1782327864956050"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_js",
        sha256 = "036462d6c37ce3734a7060eca042b564ef9c023e7072f4845ff0e1a342b784d1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.js?generation=1782327868305124"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_wasm",
        sha256 = "5ad19a23121f8ee3a077c493453fb817080fc611ec88608fede890f5ac281dd1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.wasm?generation=1782327872118611"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_js",
        sha256 = "feb53fba16c0924d14b27dcd002ecf15cd3110474ac45a6de7f561200fa34a88",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.js?generation=1782327876076861"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "5dda0fb606f2e417ba1884ebe5ee8d755d2234fe087f64a259d0dc5c772b45d8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.wasm?generation=1782327880031247"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "8c4c9488b346ec98b79fb0ecb05be3a2e37eac7c08c517088ee93c04bc13defe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.js?generation=1782327883718640"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_wasm",
        sha256 = "d7665d1e08732e0fdc467a7fb399df740d0a239fd62c3baf2821483c832b739b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.wasm?generation=1782327887808866"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_js",
        sha256 = "af7f329789a24b85a60af4c7986ecef58c6df622dab5d204e4a2a873ecba3702",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.js?generation=1782327891402760"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_wasm",
        sha256 = "437318e0078f9a8dd609b1d71f81a0b7d0779fd4447a3e0a69138f447d3708be",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.wasm?generation=1782327894978721"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_js",
        sha256 = "c37f0a86d882da367aaf4a86c440f42acea06f1d024da96101626738b4e6ef5a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.js?generation=1782327898668662"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "6f5ec7043031411207fb3b0528e35c80be5685ba5f3d778a0b05d53aa2a1efea",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.wasm?generation=1782327902540619"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "69f2a397ce11cbe9d040b7983c6d44812c1e8703d0c6d263be487dadcae34f76",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.js?generation=1782327906303652"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_wasm",
        sha256 = "9e58bfc755889931d2729ae5880b85366352c63c37e5c74cbf033cf4c89bd81e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.wasm?generation=1782327910110208"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_js",
        sha256 = "5e41c131588096c71a99e113b2c0c6f543e4bd3c4b141ace752e03d53a4e5b98",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.js?generation=1782327913685863"],
    )
