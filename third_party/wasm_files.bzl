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
        sha256 = "1be15a0b329df2ba70f5e07de00e2753451b3e79a2d88781534a2f984009ad9c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.wasm?generation=1792766333441935"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_js",
        sha256 = "3a94074f3d733097b371413b3dd610430735ed65e5c64c37aaf3d8ec83adb236",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.js?generation=1792766337161492"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_wasm",
        sha256 = "4f70c23b9348bfb3a637d65802afa49a52b69870265bdccc7ba4b205657c74c2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.wasm?generation=1792766348777390"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_js",
        sha256 = "ac149fc251f9ddf8758b3f4eb2f5f1900755608bed75423ba9704ebdb8aac37d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.js?generation=1792766352470732"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_wasm",
        sha256 = "7f489d03be75cc6e87bb9cf0b0f1309a889ce9684f2c9d96c02dcadc27e4afdf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.wasm?generation=1792766356471661"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_js",
        sha256 = "c132f37395c3cca69830d1796620e11e68558280a112cba31b71448b83907b47",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.js?generation=1792766360267989"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_wasm",
        sha256 = "c04527557088b0b758030b323ada7315d2e1ecd4bbc280fb83b8a6d90ec5f49d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.wasm?generation=1792766371994138"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_js",
        sha256 = "69b381050e25ce41702503acf4fe71169e3c6752184d39bb7f5481770e5a46af",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.js?generation=1792766375625180"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_wasm",
        sha256 = "bf4dd61f1f114df39ad33ae943dce26236dd2b3edad11b984fc78819f3a2cddd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.wasm?generation=1792766379631807"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_js",
        sha256 = "089b3c6779a99412b783b17fce349b834f653517368a8f54dacb2c2612826ba1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.js?generation=1792766383537056"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_wasm",
        sha256 = "8fb74bf6f247f8a95be6c20a7f26623255dead9c4b98bec22d808e1cc68316f0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.wasm?generation=1792766395515614"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_js",
        sha256 = "35e876439c3178bdebf702c018de2d7bda4b890296603b2f1a946dd128dadf0e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.js?generation=1792766399314787"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_wasm",
        sha256 = "36808df5ece42b455764c5c8dc7bc57de9da4192afdf8a08b3ef60c10dbe4e9e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.wasm?generation=1792888857017427"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_js",
        sha256 = "7cf1532f6ada705ae0a331988a858c7b58b00979128ee1b69ae2bf7680d2085e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.js?generation=1792888860755454"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_wasm",
        sha256 = "0f4103d32f7df0e75bd2894e7d7e521f203a58794d79202d18ac40207a522509",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.wasm?generation=1792888872777689"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_js",
        sha256 = "07ab98c446aa082c1c258a593bcb7bd61532df011395eaa4182572cc8b542f56",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.js?generation=1792888876496721"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_module_internal_wasm",
        sha256 = "c1d2f89017772648ab3a9f0dd4dd3a3d1780877b5a81e589c4014a4a69c05827",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_module_internal.wasm?generation=1792766427609220"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_module_internal_js",
        sha256 = "eae69107f80628c922d1ca2178b504a269915e08527cbad44a486fd598cab077",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_module_internal.js?generation=1792766431310563"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_internal_wasm",
        sha256 = "71d611daf78acdd68bd8a2ce1c1b263a02d5dd3666b53c5c96c36ab3bbac3032",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_internal.wasm?generation=1792766443392721"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_internal_js",
        sha256 = "2971bf9c3f084a75a026dc2228f154224b16d94b7391265d3126c3e2287363d6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_internal.js?generation=1792766447050359"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_wasm",
        sha256 = "66e798aea84578b5f26a75f6351c9b0655574c9439ddc57e9bdc66c17cb854d6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.wasm?generation=1792766450957911"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_js",
        sha256 = "16a0a6ab39c3b26190e12e7bdc2304ab641defb1d562bd0ee1d4d2434a78c535",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.js?generation=1792766454624167"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_wasm",
        sha256 = "314e7f13c8633c36ea59ae93da371ef590f6d32f3a6856d84a52f77594205906",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.wasm?generation=1792766466661322"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_js",
        sha256 = "3b7c8f2baebd391cd1c5465e87652102eb5080cb2831f2294f1ab96feb06a658",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.js?generation=1792766470503629"],
    )
