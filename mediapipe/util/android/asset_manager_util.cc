// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/util/android/asset_manager_util.h"

#include <fstream>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/filesystem.h"

namespace {

// Checks for, prints and clears any pending Java exceptions.
// Returns true if there was a pending exception.
inline bool ExceptionPrintClear(JNIEnv* env) {
  if (env->ExceptionCheck()) {
    env->ExceptionDescribe();
    env->ExceptionClear();
    return true;
  }
  return false;
}

}  // namespace

namespace mediapipe {

AAssetManager* AssetManager::GetAssetManager() { return asset_manager_; }

bool AssetManager::InitializeFromAssetManager(JNIEnv* env,
                                              jobject local_asset_manager) {
  return InitializeFromAssetManager(env, local_asset_manager, "");
}

bool AssetManager::InitializeFromAssetManager(
    JNIEnv* env, jobject local_asset_manager,
    const std::string& cache_dir_path) {
  cache_dir_path_ = cache_dir_path;
  // Create a global reference so that Java doesn't remove the object.
  jobject global_asset_manager = env->NewGlobalRef(local_asset_manager);

  // Finally get the pointer to the AAssetManager using native code.
  asset_manager_ = AAssetManager_fromJava(env, global_asset_manager);
  if (asset_manager_) {
    LOG(INFO) << "Created global reference to asset manager.";
    return true;
  }
  return false;
}

bool AssetManager::InitializeFromContext(JNIEnv* env, jobject context,
                                         const std::string& cache_dir_path) {
  if (!mediapipe::java::SetJavaVM(env)) {
    return false;
  }

  if (context_ != nullptr) {
    env->DeleteGlobalRef(context_);
  }
  context_ = env->NewGlobalRef(context);

  // Get the class of the Java activity that calls this JNI method.
  jclass context_class = env->GetObjectClass(context_);
  // Get the id of the getAssets method for the activity.
  jmethodID context_class_get_assets = env->GetMethodID(
      context_class, "getAssets", "()Landroid/content/res/AssetManager;");
  // Call activity.getAssets();
  jobject local_asset_manager =
      env->CallObjectMethod(context_, context_class_get_assets);

  // TODO: Don't swallow the exception
  if (ExceptionPrintClear(env)) {
    return false;
  }

  return InitializeFromAssetManager(env, local_asset_manager, cache_dir_path);
}

bool AssetManager::InitializeFromActivity(JNIEnv* env, jobject activity,
                                          const std::string& cache_dir_path) {
  return InitializeFromContext(env, activity, cache_dir_path);
}

bool AssetManager::FileExists(const std::string& filename, bool* is_dir) {
  if (!asset_manager_) {
    LOG(ERROR) << "Asset manager was not initialized from JNI";
    return false;
  }

  auto safe_set_is_dir = [is_dir](bool is_dir_value) {
    if (is_dir) {
      *is_dir = is_dir_value;
    }
  };

  AAsset* asset =
      AAssetManager_open(asset_manager_, filename.c_str(), AASSET_MODE_RANDOM);
  if (asset != nullptr) {
    AAsset_close(asset);
    safe_set_is_dir(false);
    return true;
  }

  // Check if it is a directory.
  AAssetDir* asset_dir =
      AAssetManager_openDir(asset_manager_, filename.c_str());
  if (asset_dir != nullptr) {
    // openDir always succeeds, so check if there are files in it. This won't
    // work if it's empty, but an empty assets manager directory is essentially
    // unusable (i.e. not considered a valid path).
    bool dir_exists = AAssetDir_getNextFileName(asset_dir) != nullptr;
    AAssetDir_close(asset_dir);
    safe_set_is_dir(dir_exists);
    return dir_exists;
  }

  return false;
}

bool AssetManager::ReadFile(const std::string& filename, std::string* output) {
  CHECK(output);
  if (!asset_manager_) {
    LOG(ERROR) << "Asset manager was not initialized from JNI";
    return false;
  }

  AAsset* asset =
      AAssetManager_open(asset_manager_, filename.c_str(), AASSET_MODE_RANDOM);
  if (asset == nullptr) {
    return false;
  } else {
    size_t size = AAsset_getLength(asset);
    output->resize(size);
    memcpy(static_cast<void*>(&output->at(0)), AAsset_getBuffer(asset), size);
    AAsset_close(asset);
  }
  return true;
}

absl::StatusOr<std::string> AssetManager::CachedFileFromAsset(
    const std::string& asset_path) {
  RET_CHECK(cache_dir_path_.size()) << "asset manager not initialized";

  std::string file_path =
      absl::StrCat(cache_dir_path_, "/mediapipe_asset_cache/", asset_path);

  // TODO: call the Java AssetCache, or make it call us.
  // For now, since we don't know the app version, we overwrite the cache file
  // unconditionally.

  std::string asset_data;
  RET_CHECK(ReadFile(asset_path, &asset_data))
      << "could not read asset: " << asset_path;

  std::string dir_path = File::StripBasename(file_path);
  MP_RETURN_IF_ERROR(file::RecursivelyCreateDir(dir_path, file::Defaults()));

  std::ofstream output_file(file_path);
  RET_CHECK(output_file.good()) << "could not open cache file: " << file_path;

  output_file << asset_data;
  RET_CHECK(output_file.good()) << "could not write cache file: " << file_path;

  return file_path;
}

absl::Status AssetManager::ReadContentUri(const std::string& content_uri,
                                          std::string* output) {
  RET_CHECK(mediapipe::java::HasJavaVM()) << "JVM instance not set";
  JNIEnv* env = mediapipe::java::GetJNIEnv();
  RET_CHECK(env != nullptr) << "Unable to retrieve JNIEnv";

  RET_CHECK(context_ != nullptr) << "Android context not initialized";

  // ContentResolver contentResolver = context.getContentResolver();
  jclass context_class = env->FindClass("android/content/Context");
  jmethodID context_get_content_resolver =
      env->GetMethodID(context_class, "getContentResolver",
                       "()Landroid/content/ContentResolver;");
  jclass content_resolver_class =
      env->FindClass("android/content/ContentResolver");
  jobject content_resolver =
      env->CallObjectMethod(context_, context_get_content_resolver);

  // Uri uri = Uri.parse(content_uri)
  jclass uri_class = env->FindClass("android/net/Uri");
  jmethodID uri_parse = env->GetStaticMethodID(
      uri_class, "parse", "(Ljava/lang/String;)Landroid/net/Uri;");
  jobject uri = env->CallStaticObjectMethod(
      uri_class, uri_parse, env->NewStringUTF(content_uri.c_str()));

  // AssetFileDescriptor descriptor =
  //          contentResolver.openAssetFileDescriptor(uri, "r");
  jmethodID content_resolver_open_file_descriptor =
      env->GetMethodID(content_resolver_class, "openAssetFileDescriptor",
                       "(Landroid/net/Uri;Ljava/lang/String;)"
                       "Landroid/content/res/AssetFileDescriptor;");
  jobject descriptor = env->CallObjectMethod(
      content_resolver, content_resolver_open_file_descriptor, uri,
      env->NewStringUTF("r"));

  RET_CHECK(!ExceptionPrintClear(env)) << "unable to open content URI";

  // long size = descriptor.getLength();
  jclass asset_file_descriptor_class =
      env->FindClass("android/content/res/AssetFileDescriptor");
  jmethodID get_length_method =
      env->GetMethodID(asset_file_descriptor_class, "getLength", "()J");
  jlong size = env->CallLongMethod(descriptor, get_length_method);

  // byte[] data = new byte[size];
  jbyteArray data = env->NewByteArray(size);

  // FileInputStream stream = descriptor.createInputStream();
  jmethodID create_input_stream_method =
      env->GetMethodID(asset_file_descriptor_class, "createInputStream",
                       "()Ljava/io/FileInputStream;");
  jobject stream =
      env->CallObjectMethod(descriptor, create_input_stream_method);

  RET_CHECK(!ExceptionPrintClear(env)) << "failed to create input stream";

  // stream.read(data);
  jclass input_stream_class = env->FindClass("java/io/InputStream");
  jmethodID read_method = env->GetMethodID(input_stream_class, "read", "([B)I");
  env->CallIntMethod(stream, read_method, data);

  RET_CHECK(!ExceptionPrintClear(env)) << "failed to read input stream";

  // stream.close();
  jmethodID close_method = env->GetMethodID(input_stream_class, "close", "()V");
  env->CallVoidMethod(stream, close_method);

  output->resize(size);
  env->GetByteArrayRegion(data, 0, size,
                          reinterpret_cast<jbyte*>(&output->at(0)));
  RET_CHECK(!ExceptionPrintClear(env)) << "failed to copy array data";

  return absl::OkStatus();
}

}  // namespace mediapipe
