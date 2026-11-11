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
        sha256 = "7514e847aa907db79d0af5a17a097a2a27c21afd3df51c08b5ba034af70aac1c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.wasm?generation=1794415939596840"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_js",
        sha256 = "f345b33360f905d5b57b4d8bf4f7142457332a90840b0b8274507aa4e08cfacf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.js?generation=1794415940574087"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_wasm",
        sha256 = "ebf238bbdeab3e9e5d76c1782f44368980b8c14361c6186260d26c0ccc66da84",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.wasm?generation=1794415941932543"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_js",
        sha256 = "85082ad989c33a68ef17e58d48ba3c65aa3e10ce2f11d1bb894b9418ded61fea",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.js?generation=1794415942976304"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_wasm",
        sha256 = "851fc1ce561668c4df0092704e4a50c15b0f5dff694d68ae9bc17bee0e7418cf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.wasm?generation=1794415944067304"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_js",
        sha256 = "d3a0541e08e2e96ebfba84a8e4fd9f473439463db8404b4489c9c3584ba3423c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.js?generation=1794415945089490"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_wasm",
        sha256 = "5c7bebd47d48d49d89946055688865c1847e5cd3ef9583c82cdd43a667e1ffe1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.wasm?generation=1794415946389767"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_js",
        sha256 = "61e294940064b8ccea2eab4e7fd3f3970ce7db5b9bcabbe00a950101aebe7d28",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.js?generation=1794415947342173"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_wasm",
        sha256 = "78e4c936e4d5b03db251e86608ba10e1bc1f77b3ec5a1d71db54e5c6816c0a0b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.wasm?generation=1794415949369129"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_js",
        sha256 = "292326376c08ed730c47b0e462ebf405708a372ed4289425064457e7426eceb4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.js?generation=1794415950349274"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_wasm",
        sha256 = "ea3f9c25f48d41e16821ac7c0a7f0fb4675f057eab575d361e88076f34665022",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.wasm?generation=1794415952319168"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_js",
        sha256 = "42280b766094c80c86b8442beafd2dad5846406b2f99a49dcadef611d88112c8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.js?generation=1794415953331577"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_wasm",
        sha256 = "f89a5065bf5034cab5a36481cb1fbb14aca418b88440ed3657b1d139afcbff02",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.wasm?generation=1794415955408319"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_js",
        sha256 = "bcbf8d56a89b0e7679fb84f06ac134ad84cdd50d81de1509fc59536a59206c59",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.js?generation=1794415956312584"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_wasm",
        sha256 = "752c21be9e1573c3e207b153f4a6f2753c6753bce3a1642d4cbdf64107110120",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.wasm?generation=1794415958082173"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_js",
        sha256 = "8a54a53fb3aa3699c6e3075fee2e1a47934886fd3edb8b7c37eb94f44f0a4984",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.js?generation=1794415958988886"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_module_internal_wasm",
        sha256 = "d9ce1dfb0ebbda34b3caf1d1036d3da3f29a313103cf295bd3a23136d0ab4765",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_module_internal.wasm?generation=1794415960875045"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_module_internal_js",
        sha256 = "8c0b06c9878a1b531aef8fdb3c0c5344f098bd18eb62e497847235a04b786b9c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_module_internal.js?generation=1794415961799415"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_internal_wasm",
        sha256 = "68493f2f4d17f66a7d84ef08494d23719610a71a0403323ca6d742433a3d9d85",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_internal.wasm?generation=1794415963883964"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_internal_js",
        sha256 = "d77afcb7302d33cf0327772822d31b1207083b60e118e30bf5117bfae14ae2bc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_internal.js?generation=1794415965186657"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_wasm",
        sha256 = "174d5b55d6e1ec5bbcbc26c132c6d2bd77af1ee211e52fc48663989da2aad285",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.wasm?generation=1794415967038935"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_js",
        sha256 = "327c1b025bcb766cd94dd54b65df38d1ef30ffeb2aa3fa2c41d9bd307194c4df",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.js?generation=1794415968001284"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_wasm",
        sha256 = "b3fc2b562b79a6a3773ee14afd9cbcd648c368112d70272b08312af201d212db",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.wasm?generation=1794415969690583"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_js",
        sha256 = "605a235930d461f0ab0548cc26bdc720cc12b92bbf667b3b27020e6438175fc9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.js?generation=1794415970590315"],
    )
