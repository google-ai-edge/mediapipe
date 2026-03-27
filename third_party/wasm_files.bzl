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
        sha256 = "f00a5f6eda02ad667498a2f5ea092e32fb679b854a04ef09b42bf3481e3bb457",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1774585028125243"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "2c694239f4295863b1757ec16a83f55247eaed6d9b95546c0c73466db00e55c4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1774585032523431"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_js",
        sha256 = "13ab52fdda0f7620ce015ff9e222b3f2a88f6a70d666569e23de96be49754123",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.js?generation=1774585036695122"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_wasm",
        sha256 = "527887c5834bdb326d0ad75194cfdd4d5a5004ae428018124282a3f67ad1ba8a",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.wasm?generation=1774585041105879"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "5945e2f1879233bd46dc4df1356d87669c9c2ec6b6526430700226d65a15fb8c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1774585048218308"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "c64941b1bb42184cd4f74c106ab9d5e4d75a6b8df9284ac3e01e735332c62c02",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1774585052450818"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "0dbaa4c29a3322bcac9085d4cc16c51faff2da9fd2e3993a9b61bfb1566d107e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1774585056590563"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "ef7b38b8c39a3b2d1729d1511c53cb6d91e5c8451c43b4226a5b39e6b24e30f5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1774585066229724"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_js",
        sha256 = "0a0c1245d50b58715412af6e0c6a9945cddc4b569b16f24f2cedbff831cb0714",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.js?generation=1774585070513907"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_wasm",
        sha256 = "4c15be53848da4f969041c2c09556f26984ece9def1ff8a24498948604930d56",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.wasm?generation=1774585075007874"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "0193a3991fa12a3715e1883127e059d1dbbeb44e22bf19a60aee728a04fd1c8e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1774585079100754"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "56e900d7f3976b78a55eecfdc2f67d6d0360c02086ea06d55149f84fc6a43574",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1774585083705716"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "1d819d0597302d4ff5debb569497f8faea72137194e5fdbf1c06de85cac49e8d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1774585088025464"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "1f89425f282c9b90be7c43bfb808135bb23bb0aa2feb3611fe5ab2c2ee9f15d8",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1774585092330712"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_js",
        sha256 = "dac86f52998d465c624494dd060df6ffd5a2c78c985426a33528c3d4a8968cee",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.js?generation=1774585096553768"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_wasm",
        sha256 = "8f865e2c35befe4e286edf0db9fafbe3ea117731b78efde2496a2ca041861920",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.wasm?generation=1774585100880087"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "9d1b869bbe09a5c474fef57bf9edcf093abc5d307c595c3dd2c34a4e4abb6c08",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1774585105087080"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "faf85ea32b62e2367624e08c0753573b1c30adf0fe3a66a56df53a2060bd9123",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1774585109472860"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "1183307bc795978e8b2467f55fecdddc042feec9f9d038c72d4c865187abcc52",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1774585113494519"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "c2737ad1e076f2afc14d754066748577e6bb4595fd6f83a38a59c986cf431ad4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1774585117786012"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_js",
        sha256 = "5ad55ffb1f6ef14da0d82526370ce95f04fa6e8e0fc82967f2a4ee775c69e9a6",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.js?generation=1774585121944732"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_wasm",
        sha256 = "6d749023bd41509eaa5f3dc5557f993ad61dfa86d2efe0b124c87df4aadff009",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.wasm?generation=1774585126254859"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "0d2f912dfde00d1a98695511580bea2dd19527e860c2b8080bfe98978ed51a7f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1774585130445029"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "c7a3d523cc2772b2ef3d8f7e3bd73cc0cdc745f9ec68fed5ffa2f88cb12a2ca3",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1774585134863146"],
    )
