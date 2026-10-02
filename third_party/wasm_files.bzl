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
        sha256 = "afa84a85f7c79dce2f2e6cdf4b9a10b9909c9c5f54b63dbb85df07c6a8fe23da",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.wasm?generation=1790967781565588"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_js",
        sha256 = "f38d9239c23e3591dc1fdd4774b3a51cbcda327cb2a895885f0129664471d102",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.js?generation=1790967785436037"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "6415c6ecfe324337519bfd99f9d3c36f39fc543a60a1d4085ffc5eb04428f144",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.wasm?generation=1790967789509287"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "4070819754c76a0b37460cc3eb2631271a0c8af6c480511b272a15240b51650b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.js?generation=1790967793357644"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_wasm",
        sha256 = "751f7d1504ffb5a5b9e862e3b3aec95a20f2649583c431f1fcb82914db7249c8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.wasm?generation=1790967797321530"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_js",
        sha256 = "cda4257fdead1c2eedf831c1d8cd19bbcd7be5436c34530fb77e1cf2de6d01f8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.js?generation=1790967801156787"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_wasm",
        sha256 = "941c2a9f2d7c35458d3225dde8b51803f0f617914f4ea883f839b39685afee1e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.wasm?generation=1790967805395260"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_js",
        sha256 = "d4c893c90ca62e224fcea64a42c3148516886661bdf54f78a25c81cec6725855",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.js?generation=1790967809417371"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "92178066bb38d68e5f390f54806a16c9c429a59f2da8aa975bf5808792f0750c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.wasm?generation=1790967813409745"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_js",
        sha256 = "54d28f77530d460439df8e281dbc0cf80f6cc662eca06fc8e627eec85ef68c78",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.js?generation=1790967817261464"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_wasm",
        sha256 = "6a60f3303ce0daf16f5a61aceab85b4575e9921dfa3e73167cb026e392b58668",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.wasm?generation=1790967821451543"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_js",
        sha256 = "7b76f95a8aeb99b75c42343b31ff7e4e7014ec7b0c601f19f8e53f2f9ee50635",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.js?generation=1790967825348807"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_wasm",
        sha256 = "a192db613b8ad82b600f85c6c05c8bd5e48f80176e8055dc332c27b6a6109564",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.wasm?generation=1790967829431554"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_js",
        sha256 = "d123075de7ff66eb83a1e28abc4625e872d18b1c2771384875cf5c2375e6ec75",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.js?generation=1790967833368587"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_nosimd_internal_wasm",
        sha256 = "e49dc7df15417a9f3c115b71f9fc8645ed8f72ba5e464b25fcba6e5099fa2feb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_nosimd_internal.wasm?generation=1790967837714973"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_nosimd_internal_js",
        sha256 = "430f1532801fed7bda736b82e3c6d05da87d5e21ced8bf1a51a7358334dff122",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_nosimd_internal.js?generation=1790967841609159"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_wasm",
        sha256 = "a54607d9690beef5de3fd08282dee29084d0d344cf4572f17d96f42510019aab",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.wasm?generation=1790967845667153"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_js",
        sha256 = "f49ee01d71e6d169f195f4cd815987205f04ccaddc6dfa832a592a01b4629619",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.js?generation=1790967849496349"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_wasm",
        sha256 = "d1b7cbcaa4e67ef4c9c4725520801f61fe0fd1b20e2e07ebf18379596430d96d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.wasm?generation=1790967853737425"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_js",
        sha256 = "d6a24b9c3164c7ed500a58bb06b8bc2936f99f1148f40aeea3998c1791b96f1f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.js?generation=1790967857623376"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "e48f236121464ea630ed827541a375ecdcb6def8c4e9711e930fd4f8826ed9f9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.wasm?generation=1790967862150114"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "7a8555bcd9c0220405ca968847d0212aafb11387f8606c59a8358a5f6c3e55ef",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.js?generation=1790967866108827"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_wasm",
        sha256 = "6077f3f1c74c93d1acf7631d855c37962b986926d05dfdb859347c6df4de47e5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.wasm?generation=1790967870367787"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_js",
        sha256 = "d1ef13a2cbd184e6b9605b455edcd25490a039a01af5e6e7bdb8de4972cb6ae6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.js?generation=1790967874279708"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_module_internal_wasm",
        sha256 = "093ead701102301cb2fc73755d4e51b51baf21fa08ad7f22c82ec22f62109f87",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_module_internal.wasm?generation=1790967878625000"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_module_internal_js",
        sha256 = "5f10493d0aae8ad785e6b9f89c6f8737f03326327a8bb0e2364923d166af5c1d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_module_internal.js?generation=1790967882779745"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_nosimd_internal_wasm",
        sha256 = "2bd09c5ffaec46af4a350a1a3af704c1e3bcad257f75aef4430a0e96ad99f3e7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_nosimd_internal.wasm?generation=1790967887042585"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_nosimd_internal_js",
        sha256 = "52d84e9eac577797c4e7fdd76f99bc17d5e10c2f72ca319c1166e7d3ff391846",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_nosimd_internal.js?generation=1790967890987536"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_internal_wasm",
        sha256 = "298370df699b81e506e963ca4153242140befce7439df4271e6c5aea98661c25",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_internal.wasm?generation=1790967895349019"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_decision_wasm_decision_wasm_internal_js",
        sha256 = "cae234e221a7f168ba8947823157e581cc0942d7e193db2b630dc5ba4cd6cd8c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/decision/wasm/decision_wasm_internal.js?generation=1790967899418426"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_wasm",
        sha256 = "613dd5c2b3391586a913d0e4776bfcc1e11a44472e5585375c4ce0d3f1ef4ae3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.wasm?generation=1790967903597630"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_js",
        sha256 = "55a71d3ab11affe15286bcbe8b7d1a92e217327056ebce94a0ed2f17c8ae8556",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.js?generation=1790967908264865"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "161fcc9c43f6108fcc2452ac448e5b85b7992264c252cb3d98ffc7ed9cc8da4e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.wasm?generation=1790967912383511"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "683f1e892f0dadac73fa4dd28bfc536730be26ad9d59cee5f2e0e83672710329",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.js?generation=1790967916189644"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_wasm",
        sha256 = "e0be94e123112fa65f60f8863382600f3feca5a8bd24d78ea867c476c0256943",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.wasm?generation=1790967920259943"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_js",
        sha256 = "fba3513c6c16c3815879d973d1f10ff9147e7e95ca8b0fa9d7d4c6e7bf333adf",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.js?generation=1790967924092657"],
    )
