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
        sha256 = "82fa2bc2a41dfa2ba2294e1832a64c32739fd7172f51fb97b61cd22d33de9850",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.wasm?generation=1794424637669628"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_js",
        sha256 = "6148324499281407575b9e5a3245b14bb43f61b221ee990cb81112c284061562",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.js?generation=1794424732581992"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_wasm",
        sha256 = "6022b86eede08a57d598a58c1a06c7f063cbf5d153a0d06125fd012f0910d161",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.wasm?generation=1794424733849357"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_js",
        sha256 = "89f8f2340f96b108d896e725366d5c93e25a3bc8947e289c8661fec217a070b0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.js?generation=1794424735140985"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_wasm",
        sha256 = "e08eb1405d35a6b55db90d01cbb83dbfc5cb1e8205f6460c09ed79d434743fd6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.wasm?generation=1794424736639659"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_js",
        sha256 = "c5bfb14ca78bba6e36d466fc865c33415386306c079df30858b0a378e0015f35",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.js?generation=1794424738075798"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_wasm",
        sha256 = "25b72398e0ee8924471c9efbfe4749855e7c3620940e30a44a7cdddd4dd3d7e4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.wasm?generation=1794424739295236"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_js",
        sha256 = "7c5baef83c1fd6f320001a71f736a098d0949d1293af80beeb0dfce333465beb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.js?generation=1794424740276722"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_wasm",
        sha256 = "bddeb24dbb610fbaf10cf78ab123247f70365e9006dabb1da0c136caec9aaaf6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.wasm?generation=1794424743374971"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_js",
        sha256 = "94680d35db681c3f3128f6f81fede1968ce13b7ac4eb8d122b07687a8d7d03c7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.js?generation=1794424744401587"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_wasm",
        sha256 = "c09cf55200ca6522433a0e97932a05988e3d5d3995e108aec63d7eae42b61b61",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.wasm?generation=1794424745894025"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_js",
        sha256 = "ac64f623fc762fc2bf77e5e3fc298d357615fc94db7b90f8c4933f69d34965e6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.js?generation=1794424746859447"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_wasm",
        sha256 = "c34fadfc9c3aea953392099311c9aae45b194b412310e56f16970ea5f6935032",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.wasm?generation=1794424749410795"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_js",
        sha256 = "b90b6820252fea450936f949fed74dd03f9256c5aa81b4a979c60dc72ec25556",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.js?generation=1794424750502923"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_wasm",
        sha256 = "8dd823dc9b41dda64fc871dec1f77e64458cf3fd079d738ca29e0a8df1270c79",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.wasm?generation=1794424751761296"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_js",
        sha256 = "423154ab0c56564695c3aec63c56c40e42d83e437997c4fb9fb6763cb8c58584",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.js?generation=1794424752622113"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_module_internal_wasm",
        sha256 = "ce92ca80f62449ace668dbb2f512f4ed1d2979eb4f4853cc42cc165b694f1419",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_module_internal.wasm?generation=1794424754210355"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_module_internal_js",
        sha256 = "5a7421a2ebd860347862f763f49270e42a3fd49b44384b70893ba3dee0708e7d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_module_internal.js?generation=1794424755062642"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_internal_wasm",
        sha256 = "d2d8a9f729bf814650da17d419576451c65d9d68c16b364e3ffb9b9b9aa05e03",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_internal.wasm?generation=1794424757420493"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_internal_js",
        sha256 = "7e9c9d64a20555678e60def3b0c3f85879dbe24978a76a04447db2820f1f04cc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_internal.js?generation=1794424758321670"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_wasm",
        sha256 = "07d89e11db7821877c29a4981f34923db414a65aace4b7343b68d8a51b4a4cc7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.wasm?generation=1794424759908435"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_js",
        sha256 = "e89947c8c428b30884214e2910e59b2a6904b3454d4d02e8a81a2a3e9cf239e2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.js?generation=1794424760855755"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_wasm",
        sha256 = "5efd3e119e6b66a2ee40e87fd46dab4e91c869f550d78e1a793a961eff8d8b3b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.wasm?generation=1794424762410412"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_js",
        sha256 = "3a38d265895f85763a253d1029e0301a1057d63a1354be03ab1fd81fd9afeb4d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.js?generation=1794424763484656"],
    )
