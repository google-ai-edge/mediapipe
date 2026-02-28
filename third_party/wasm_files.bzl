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
        sha256 = "fe8d48e8e307a59020c35a08088564bd28e481a29c518ac9d3071878d0ed6817",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1771786242380697"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "8dc308d028fc9782af69f895d73c0f878f6ccd1b953b3c567b73053612d2678f",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1771786244450553"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_js",
        sha256 = "ebab6d14a9005341c003f303597759493c053d7f5cdb3507781d8d00a9063112",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.js?generation=1771786246437397"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_wasm",
        sha256 = "4a7de664d3478a31d2697ffb88a7270b004c4c91bbab8620d9eb8f8af9cc3419",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.wasm?generation=1771786248454203"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "fdc9fb915026f3cf3baf1a026cc904f1d40e8218a7168ed99b3f81ebdeb081dd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1771786254422239"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "91dfb8d937460a564e36565039610a9e62cc5d678f543f74c5da060410dbdb51",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1771786256455057"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "9e38b10ec7605cb5163121289d71f37006967b2edd28dadd4d9507597b7a7afc",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1771786258579142"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "67ef11462f68f20bb90bc25ac7d53bc14759269dbc127ef451c142d8f508f920",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1771786261010431"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_js",
        sha256 = "10d3fdda3b2a359afd1c989c2ed5e78496063da9907e16577824de8e49ef5254",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.js?generation=1771786262970160"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_wasm",
        sha256 = "b548723f0604db0b79e06286b1a2d0f66ff284b5a9041d00ee17bf8b2aa34071",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.wasm?generation=1771786265164412"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "3c3bdd8554ccfedbfdfb34680af641fb59c83cff71b522f6a30c44b6080a9948",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1771786271333616"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "47884801489c9c7ef65e530f31d2799a4624097837020bc773e905779bdcbdd2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1771786273867795"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "dc282a68e2b40c57d98cf03b49c05f69cd2a60be1e518d52900bb55debf473ce",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1771786275942825"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "7532312dfec7e7ec63b56480d7dd89604fcb54b382988e4ca636059ed5ed3eea",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1771786278295064"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_js",
        sha256 = "80a31bc88d60483805f365581118b0b40d4739677ea8ab99d0fc06355709be7e",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.js?generation=1771786280107758"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_wasm",
        sha256 = "365f12187bb15ebf6c36e25348e4c2a503297fcb2ce00b6551ff2acd912e33c2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.wasm?generation=1771786282122973"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "9ad8793857376af5881733b38edde2b1310577765e58c13af3edc394de9f1851",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1771786288149449"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "4dc6f14ea2377e2d57ce8c93e2f940ba2e760cf86e4235bf9143f61986c1a82d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1771786290280586"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "45b0eeace17b28474f36b55899c04b8d59fe4ceb4899c4213193e9d048e7b90c",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1771786292295170"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "2b4803c9af4543d2e03c013fad19d7e48f070fbf45bc728881f3cd508db7ffc0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1771786294664628"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_js",
        sha256 = "145e36b64b62c4e756248333d7d3578a6f0ca6356cbaf13222587f3d5d278ae2",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.js?generation=1771786296597033"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_wasm",
        sha256 = "2fb2d1a1c8ac516036704bb7761aaa8da98c36df7c0726cef37b21f7891b2a46",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.wasm?generation=1771786298707275"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "9015fca405fb51eaf9c78c2c529fdf6dceb02d75087dbaee4a57bd28ca11c679",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1771786304421686"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "2da00e173ea1c76b9359c79d67e8ce55f68a753e2b4e5c6ad5bf1bc78b3a8567",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1771786306577936"],
    )
