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
        sha256 = "197780f10bb1dcc8ebea08910b08cc77e75d9f5b9e12faca3050d849bddaeaa7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.wasm?generation=1782416863272179"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_js",
        sha256 = "7f06bc6ef244c0fd4780db25e29d160e2e9b8d653f9d794c192bfdd19cd671b0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.js?generation=1782416866857920"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "d4dc25205303ac73b84eaad5c62bb3f6729327ed73108ef3887830451f3a194e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.wasm?generation=1782416870670616"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "195c31e2bbde1ec3c0dc13d34d43dc5e8ab8c6563b4b3f49fbcc9960e1900f0b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.js?generation=1782416874254707"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_wasm",
        sha256 = "47abbd8dacf0a87787abf7b53e661fc85f62c4c0ec022f69a238701f92e058b5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.wasm?generation=1782416877962668"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_js",
        sha256 = "5af0b8973651f377b0bfd39e17da7b91709261cef7ed8c97ccbd84eaa00039dc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.js?generation=1782416881746443"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_wasm",
        sha256 = "0d4e4d0531d8549001c71db6307299a9299c174f1994c7466cc39a6e3109e44f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.wasm?generation=1782416885444863"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_js",
        sha256 = "24cc366edfd4d89b675bfb65f98617babd23c5381c0339b29fba006beeb264c6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.js?generation=1782416889137092"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "a05c20e532d1638b2aab5da03a22868528e0edddc1f3dc7bcb28df847869c7e5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.wasm?generation=1782416893040999"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_js",
        sha256 = "2dd47ab3154fe6f9c8892922151476349dc3ec753d2cbe5bfc9be8d734e994b5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.js?generation=1782416896754141"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_wasm",
        sha256 = "83d706514ec3e76bcbda2b0daee57643aa90ba89ecf5f085f1ff145578668a40",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.wasm?generation=1782416900509889"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_js",
        sha256 = "a556d746a84ab4958e5c8ae379d49eacd72c59b7a4c1e392ccd3caaa0b6d7f6f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.js?generation=1782416904174929"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_wasm",
        sha256 = "1991e5b039dc9a6f326691f0a1643ab2996e6b2530fda6672ab58d2fc6425faa",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.wasm?generation=1782416908040501"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_js",
        sha256 = "bafcfe89f80be69fe56b7c5a1723e8346b9b70e45e3b44914083cd0dce33e8f1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.js?generation=1782416911752187"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "c52b0de079660ba6050f6ad5abd756bb2c9f63e0748d26b32e20b7ad43288be5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.wasm?generation=1782416915695905"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "195a9372e5c065fc6f6ccc824902ae65118e8c414307a4755fd8b1450ac8f3ac",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.js?generation=1782416919259089"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_wasm",
        sha256 = "02bf269c2a4e392b2c710696a9da0794096d0a039b18693182098430fb9009fb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.wasm?generation=1782416923340112"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_js",
        sha256 = "43082c237b81165b0a946a4bf738da5a8fc5ae743259c8507b1233d98036ec94",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.js?generation=1782416926934499"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_wasm",
        sha256 = "e502b27ddd1bfe456f4707395918c4a9be20234852947eb91c3779e23c61aa84",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.wasm?generation=1782416930762868"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_js",
        sha256 = "0a9e5d48f120fca6c80b5897ca39c57ddda535e61c8484244ac7cddb2daeca66",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.js?generation=1782416934368211"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "97821e3759a69ea0abec149362eec2a6b329c29cfd739952fe2329bbf9d24f41",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.wasm?generation=1782416938186180"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "9287bed9889b37ceb6fbc056578e9e16bc89423e9f79309ce1749f7c73ff4dd5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.js?generation=1782416941933518"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_wasm",
        sha256 = "901308fc0a09164c317c56fdba7cb5bd795cc8aa6382c6f612eba6412f1d1645",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.wasm?generation=1782416945697839"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_js",
        sha256 = "fabc397e0e2c0b4f363fed4a069d319421306086133048b4d15364376e584834",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.js?generation=1782416949269794"],
    )
