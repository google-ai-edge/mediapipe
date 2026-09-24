"""
WASM dependencies for MediaPipe.

This file is auto-generated.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

# buildifier: disable=unnamed-macro
def wasm_files():
    """WASM dependencies for MediaPipe."""

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_wasm",
        sha256 = "ea4dd54b41912ee17d9a89750674dfd5af0cf679b370c9d45d51b6a01f5dbb7b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.wasm?generation=1790196785868891"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_module_internal_js",
        sha256 = "136a8c1bee1bcafdc75ec2bdb38faf117ec0c338f31dd43ebd1c9b74889b5748",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_module_internal.js?generation=1790196790039362"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_nosimd_internal_wasm",
        sha256 = "8864d10b54c55d3652c5850371412f6e80b224c66466bfd0039949735c519018",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_nosimd_internal.wasm?generation=1790196794424068"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_nosimd_internal_js",
        sha256 = "241710f3f543da0f8a61f55d9de1f768f4071c4342215756d65aec25227a4e24",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_nosimd_internal.js?generation=1790196798582974"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_wasm",
        sha256 = "77a0d598cc07907ab14f922d7410ccc2cac92198d60ccc661d0933d4b37406b2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.wasm?generation=1790196803077172"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_retrieval_wasm_retrieval_wasm_internal_js",
        sha256 = "615dedbd9517c46e3e50ab1230bce3ee97dfc32148cf8db47a117d47dafec5a4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/retrieval/wasm/retrieval_wasm_internal.js?generation=1790196807400050"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_wasm",
        sha256 = "601cbe872419585211417d33718839cd5cd983fe2d2660fbce2278cbfb76b2dd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.wasm?generation=1790196811537304"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_module_internal_js",
        sha256 = "6dc5a08de0d2623e906f8817ffe2cf80c3f4d33da36fa0bd0a6a0c8f676e50dc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_module_internal.js?generation=1790196815543195"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "1ba2e2c8c6717f7621956adba9d51bb79dcf51e183c4da41e5f77ffd0e0a9ea3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.wasm?generation=1790196819716097"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "bd73365cf87ac5176e66d31955624b2dfeaa26a50e68e69ce3e19d0a0dfe3216",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_nosimd_internal.js?generation=1790196823612580"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_wasm",
        sha256 = "42f7514267f8db4ed8252a3de46b9f4f0cd9dbb1de2278d76a11476e705e4383",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.wasm?generation=1790196827601664"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_vision_wasm_vision_wasm_internal_js",
        sha256 = "ab92c6823921dbd19db29e1dd267b601e5c283825d6cff7295173643346e79f9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/vision/wasm/vision_wasm_internal.js?generation=1790196832384004"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_wasm",
        sha256 = "97f1ad29a9d5ac73c72c0c5c4740337c6684812ea3960f01df86cc55dfea7704",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.wasm?generation=1790196836625969"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_module_internal_js",
        sha256 = "7f6f7b7dba7e3d4759e62640235890e8648ea152a4cf8b88fc1aa08ab2de6702",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_module_internal.js?generation=1790196840477292"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "92b10cede68810c6c1266fcbfa21843be0e53b88bf3df85da2a8f77dfbd6034e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.wasm?generation=1790196844522866"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_nosimd_internal_js",
        sha256 = "4fe0bad4ad1f00e084913c8311a2060dfa4f621eba7a38c6f774b4622008fa2d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_nosimd_internal.js?generation=1790196848466300"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_wasm",
        sha256 = "23380d00d1787e1e20ec1c9fc61e487018f020b94e246319bdc623c84413c7de",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.wasm?generation=1790196852737387"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_text_wasm_text_wasm_internal_js",
        sha256 = "095a290cdd66eb36a392238db7042654af654bc9aeef64dc93a6b03262ce03a9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/text/wasm/text_wasm_internal.js?generation=1790196856518178"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_wasm",
        sha256 = "cf5c205fd2ce06e78efc3e63395a217037f0202f12f5d2ab1b8584391fcc7103",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.wasm?generation=1790196860885762"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_module_internal_js",
        sha256 = "c35cbb753bb2000c738e788f399df2fccfda1ee4011c8996b11e51a9c50dfe64",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_module_internal.js?generation=1790196864744110"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "28c1343ad948febdc89179e4d9c454af7af9b2a13f29d9bbd412e1ef4277b02a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.wasm?generation=1790196869201830"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "1a6146e672cb02b80f43b81625ea655dc2080b3f8e2641b2583f60e0a50a2777",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_nosimd_internal.js?generation=1790196873190965"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_wasm",
        sha256 = "140b174cf3df62911e515541131caf90a06a642ef2653f312ba1699587c350d4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.wasm?generation=1790196877594208"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_genai_wasm_genai_wasm_internal_js",
        sha256 = "753e6a9d1159661d624c29c6bc6d4b3cdca14b221c0793f1cb55cbffd6ec9aa1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/genai/wasm/genai_wasm_internal.js?generation=1790196881404756"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_wasm",
        sha256 = "62f0a9086741fae98e5d3d455854d6271989fa49d26b90727efd2a6c56ad9e00",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.wasm?generation=1790196885506697"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_module_internal_js",
        sha256 = "3ac5e0609a31da07c8a0f18f44be42d10d8a969fb0b403d228bd4daecfc29137",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_module_internal.js?generation=1790196889443466"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "4e6c74a0c171cd03f4e2cd6268d4613400852fe6e01ed29df4217d9acd90b9e1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.wasm?generation=1790196893540521"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "7403a23913cf7d5e99136aaaad4d1d7f726d037e0c871e480efa8e504471abd7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_nosimd_internal.js?generation=1790196897387489"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_wasm",
        sha256 = "9aeee5ab9d835b57ffac0616c76437ba1dbb74381cfca1d1ff7d8de87bd695ae",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.wasm?generation=1790196901445547"],
    )

    http_file(
        name = "com_google_mediapipe_tasks_web_audio_wasm_audio_wasm_internal_js",
        sha256 = "fa60525389886e0dbc999d65417b24b3e0e7b25f3f507a649214336e1f9b859d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/tasks/web/audio/wasm/audio_wasm_internal.js?generation=1790196905422340"],
    )
