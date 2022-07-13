// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#ifndef BASE_ANDROID_BUILD_INFO_H_
#define BASE_ANDROID_BUILD_INFO_H_
#include <jni.h>
#include <string>
#include <vector>
#include "base_export.h"
#include "macros.h"
#include "base/memory/singleton.h"
namespace QImage {
    namespace android {
// This enumeration maps to the values returned by BuildInfo::sdk_int(),
// indicating the Android release associated with a given SDK version.
        enum SdkVersion {
            SDK_VERSION_JELLY_BEAN = 16,
            SDK_VERSION_JELLY_BEAN_MR1 = 17,
            SDK_VERSION_JELLY_BEAN_MR2 = 18,
            SDK_VERSION_KITKAT = 19,
            SDK_VERSION_KITKAT_WEAR = 20,
            SDK_VERSION_LOLLIPOP = 21,
            SDK_VERSION_LOLLIPOP_MR1 = 22,
            SDK_VERSION_MARSHMALLOW = 23,
            SDK_VERSION_NOUGAT = 24,
            SDK_VERSION_NOUGAT_MR1 = 25,
            SDK_VERSION_OREO = 26,
            SDK_VERSION_O_MR1 = 27,
            SDK_VERSION_P = 28,
            SDK_VERSION_Q = 29,
            SDK_VERSION_R = 30,
        };
// BuildInfo is a singleton class that stores android build and device
// information. It will be called from Android specific code and gets used
// primarily in crash reporting.
        class BASE_EXPORT BuildInfo {
                public:
                ~BuildInfo() {}
                // Static factory method for getting the singleton BuildInfo instance.
                // Note that ownership is not conferred on the caller and the BuildInfo in
                // question isn't actually freed until shutdown. This is ok because there
                // should only be one instance of BuildInfo ever created.
                static BuildInfo* GetInstance();
                // Const char* is used instead of std::strings because these values must be
                // available even if the process is in a crash state. Sadly
                // std::string.c_str() doesn't guarantee that memory won't be allocated when
                // it is called.
                const char* device() const {
                    return device_;
                }
                const char* manufacturer() const {
                    return manufacturer_;
                }
                const char* model() const {
                    return model_;
                }
                const char* brand() const {
                    return brand_;
                }
                const char* android_build_id() const {
                    return android_build_id_;
                }
                const char* android_build_fp() const {
                    return android_build_fp_;
                }
                const char* gms_version_code() const {
                    return gms_version_code_;
                }
                const char* host_package_name() const { return host_package_name_; }
                const char* host_version_code() const { return host_version_code_; }
                const char* host_package_label() const { return host_package_label_; }
                const char* package_version_code() const {
                    return package_version_code_;
                }
                const char* package_version_name() const {
                    return package_version_name_;
                }
                const char* package_name() const {
                    return package_name_;
                }
                // Will be empty string if no app id is assigned.
                const char* firebase_app_id() const { return firebase_app_id_; }
                const char* custom_themes() const { return custom_themes_; }
                const char* resources_version() const { return resources_version_; }
                const char* build_type() const {
                    return build_type_;
                }
                const char* board() const { return board_; }
                const char* installer_package_name() const { return installer_package_name_; }
                const char* abi_name() const { return abi_name_; }
                int sdk_int() const {
                    return sdk_int_;
                }
                // Returns the targetSdkVersion of the currently running app. If called from a
                // library, this returns the embedding app's targetSdkVersion.
                //
                // This can only be compared to finalized SDK versions, never against
                // pre-release Android versions. For pre-release Android versions, see the
                // targetsAtLeast*() methods in BuildInfo.java.
                int target_sdk_version() const { return target_sdk_version_; }
                bool is_debug_android() const { return is_debug_android_; }
                bool is_tv() const { return is_tv_; }
                const char* version_incremental() const { return version_incremental_; }
                private:
                friend struct BuildInfoSingletonTraits;
                explicit BuildInfo(const std::vector<std::string>& params);
                // Const char* is used instead of std::strings because these values must be
                // available even if the process is in a crash state. Sadly
                // std::string.c_str() doesn't guarantee that memory won't be allocated when
                // it is called.
                const char* const brand_;
                const char* const device_;
                const char* const android_build_id_;
                const char* const manufacturer_;
                const char* const model_;
                const int sdk_int_;
                const char* const build_type_;
                const char* const board_;
                const char* const host_package_name_;
                const char* const host_version_code_;
                const char* const host_package_label_;
                const char* const package_name_;
                const char* const package_version_code_;
                const char* const package_version_name_;
                const char* const android_build_fp_;
                const char* const gms_version_code_;
                const char* const installer_package_name_;
                const char* const abi_name_;
                const char* const firebase_app_id_;
                const char* const custom_themes_;
                const char* const resources_version_;
                // Not needed by breakpad.
                const int target_sdk_version_;
                const bool is_debug_android_;
                const bool is_tv_;
                const char* const version_incremental_;
                DISALLOW_COPY_AND_ASSIGN(BuildInfo);
        };
    }  // namespace android
}  // namespace base
#endif  // BASE_ANDROID_BUILD_INFO_H_