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
        sha256 = "d6ee220f82ee3c7b3093b970754cfed1952a47f0db860d2183268f64e8d79099",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.js?generation=1782267336837410"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_internal_wasm",
        sha256 = "ead402c09a260eb46dbab452310acc05c2d264a24dc793d05114dd0384e859a1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_internal.wasm?generation=1782267340475266"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_js",
        sha256 = "1d97ea5dfa7c03d5811e95c2b6b239c9b966046b27c34ae9315f34e65384ca01",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.js?generation=1782267344027105"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_module_internal_wasm",
        sha256 = "daef62e608d7a045f56942433949db09f90db753cbfddbe86c8e2d3092e174fd",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_module_internal.wasm?generation=1782267347777678"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_js",
        sha256 = "da346bc2999ab3c5abf945929c0a64bc36925b8be1e83aaae8056d625687e509",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.js?generation=1782267351375088"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_audio_wasm_nosimd_internal_wasm",
        sha256 = "0f9d58de65f40303fd18149679bf7a247fa44bca9a2711debd1e25a820988d50",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/audio_wasm_nosimd_internal.wasm?generation=1782267355068853"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_js",
        sha256 = "17cf81ce1c6fc0f9c264dcada190e4a30d97106cf86f04daada44e167f41087d",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.js?generation=1782267358681133"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_internal_wasm",
        sha256 = "866027fe679a79808d59150af9a25980d783b788813dacf204eaee7f4d393d4b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_internal.wasm?generation=1782267362930354"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_js",
        sha256 = "9a9d1c3ab7d33dc676fce46e651c46558bdfccd1cdce18da493efaffc12e1810",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.js?generation=1782267366493185"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_module_internal_wasm",
        sha256 = "97fc557a24bae8e49793e4230355f0090719fb436af1a1ba6bf3c4f0ee6f8cd4",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_module_internal.wasm?generation=1782267370574007"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_js",
        sha256 = "f2aceccd502f3ecf3e2f9a367fea39eb7fcd2d5210fdd4a059726fb9ad4adbf1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.js?generation=1782267374086966"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_genai_wasm_nosimd_internal_wasm",
        sha256 = "fe9884e62166e955395f6031cd21b0d4f730b6f63807f4f73dee84c7ec4fdff5",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/genai_wasm_nosimd_internal.wasm?generation=1782267378025063"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_js",
        sha256 = "89a78acfeb9829a1002d472554f4dad1ff9f9396c765266dd089ba94619844f9",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.js?generation=1782267381820479"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_internal_wasm",
        sha256 = "b4d7a3be6f1c3a0c6239fcb63b0a44db64ce87a9a5d677ea91487c9eb1cc12a1",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_internal.wasm?generation=1782267385659099"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_js",
        sha256 = "a610a6697f2f7cb83d6b9d96af8660ad2e7ab83e124921929c78b8aea5abc45b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.js?generation=1782267389290732"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_module_internal_wasm",
        sha256 = "ab96c8697ef057147f010298f0e0b2d03378a0ea3629433fb6e070529175d3ae",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_module_internal.wasm?generation=1782267393002948"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_js",
        sha256 = "09fe825a4f16abcfef4a1bd0592144c0bd1616c2665aa3f42e49c074e096997b",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.js?generation=1782267396516198"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_text_wasm_nosimd_internal_wasm",
        sha256 = "1f84d123f75065b1d161943098031f4b5c2a5ddb8e5a534ae49c7d9fef4a89da",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/text_wasm_nosimd_internal.wasm?generation=1782267400345509"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_js",
        sha256 = "05d4d40da0e723d219ca82564586d2652c8b61312f0919b87c4842be8951ebeb",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.js?generation=1782267404063949"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_internal_wasm",
        sha256 = "b6e06924e075fc9febe87a48c19aa3ba2efc7401904000cc7ec8d363e0ace3b7",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_internal.wasm?generation=1782267407932874"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_js",
        sha256 = "834790aaad4403979219f8a8217996936b816a5011035606b77533d1dfb28c63",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.js?generation=1782267411475690"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_module_internal_wasm",
        sha256 = "33e9efc08b6581dc49f7ff8788d544fc3fdabc21d51f681f8614eddef70d9ef0",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_module_internal.wasm?generation=1782267415438448"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_js",
        sha256 = "da57f67b6ccedb9ab464ef602d0597143f2a75437bc022e5002ff48443458231",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.js?generation=1782267419084113"],
    )

    http_file(
        name = "com_google_mediapipe_wasm_vision_wasm_nosimd_internal_wasm",
        sha256 = "ea285d6f5ea8af086dfcb989bf70e22d25ad3aea769c24133ef0f8018e5d4882",
        urls = ["https://storage.googleapis.com/mediapipe-assets/wasm/vision_wasm_nosimd_internal.wasm?generation=1782267422841959"],
    )
