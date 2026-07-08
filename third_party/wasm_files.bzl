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
        sha256 = "825d25976e02e3a27253afdcf385c6d63a6e19bb74329c8e65798e68232308fe",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.wasm?generation=1783483216032488"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_js",
        sha256 = "e8683e14b90192cc9f5fad3b23da174b526fdfcbc7019de1c755277961b925d6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.js?generation=1783483219744436"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "5045fae29303628f8cadc842eace3c599bc58f604781feb7bd69accccf88d8d6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.wasm?generation=1783483223591267"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "14bfe4dc97219e9f745ea1503f9b8c6c41b294b86de302509411c0cf60378130",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.js?generation=1783483227293036"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_wasm",
        sha256 = "0fdf3cdb5523dee1dbee6b52ff4e77e0168c67e5c6044b34544340bcfdf1f7ac",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.wasm?generation=1783483231712356"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_js",
        sha256 = "12df5ac0c0e3ca8b00293e9d42864e0aee89b4a8a23acaff1742e99269626537",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.js?generation=1783483235489460"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_wasm",
        sha256 = "3565ef913827381f29f5e44828a4579e0ee47d09e9d2bccea06684b5a24da25c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.wasm?generation=1783483239173895"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_js",
        sha256 = "27778b8dc86012aae0c11b89389a9948616e0e02a3a8ffc8096c1ba004da240b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.js?generation=1783483243085133"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "1be74fa0aca6182b247b639b0ec95ebf12c018ca1eee2325418a4145eb9bd952",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.wasm?generation=1783483246924828"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_js",
        sha256 = "faed0096ed8eaea70258a7db7dcd1a0f5e0f65938dbff94ba455fa6f29861a86",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.js?generation=1783483250514490"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_wasm",
        sha256 = "3e5c299cf04d1ba288611fe9c323f384f01c799fab425203fec503cc43224859",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.wasm?generation=1783483254272230"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_js",
        sha256 = "b2e2f29c70ef0823eca528bb8c28b35f33e5d6af9db24982286d717c72249152",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.js?generation=1783483257848072"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_wasm",
        sha256 = "cdaac67a8c0d6143e54372678a80f86f8a4c124f82a7daa6a8ce2cf9923008ee",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.wasm?generation=1783483261706364"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_js",
        sha256 = "0ca0cd9f8a5662561428fa886b83352866823714337da25a4248f3202db689bf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.js?generation=1783483265218396"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "18ff88bb07219690754f2e5d7ff0e5bf1a136c6390f7f53b01be418cc922a762",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.wasm?generation=1783483269311854"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "aaf17bf91e69c5c14ff227eba3ccbae68f3bcaad99382fd8504e74e861c2a086",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.js?generation=1783483272801892"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_wasm",
        sha256 = "c3605c5d7569b869d94638548f71ca72e611befb43bacf316d42f7fc8ebced70",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.wasm?generation=1783483276774777"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_js",
        sha256 = "5e59cd6ca1df6be6814291e16f97ed718768a4b8cdb4b020e1734acaca86be95",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.js?generation=1783483280323273"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_wasm",
        sha256 = "05683d493803478e56db1c8b021c1d78eb0f05894477ea038eaa946a8a492972",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.wasm?generation=1783483285387442"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_js",
        sha256 = "1915a8e47daf456b8e99bc7568f8cf9be63a8b4899658cf3dacf78246e7094da",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.js?generation=1783483288900432"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "e82192c7a18ef5fc292f5c561fc8291be8db9f80112a87c4521a8ad8e09486df",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.wasm?generation=1783483292658135"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "4e791158074e2a917268e4dee15a985bf06dd5f660eba7b3cdebfa9a0e6ba4b2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.js?generation=1783483296304313"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_wasm",
        sha256 = "86433652bc8a81c8c2bf40cf8d83057c7faaeaac7b398577de8ab48fc4d2d25c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.wasm?generation=1783483299926258"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_js",
        sha256 = "c0eee62a59e1903e52f3c531de03b8ae7071fc812fe48db6f79771c26a546186",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.js?generation=1783483303505510"],
    )
