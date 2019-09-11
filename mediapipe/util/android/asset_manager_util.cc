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
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/filesystem.h"

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

bool AssetManager::InitializeFromActivity(JNIEnv* env, jobject activity,
                                          const std::string& cache_dir_path) {
  // Get the class of the Java activity that calls this JNI method.
  jclass activity_class = env->GetObjectClass(activity);

  // Get the id of the getAssets method for the activity.
  jmethodID activity_class_getAssets = env->GetMethodID(
      activity_class, "getAssets", "()Landroid/content/res/AssetManager;");

  // Call activity.getAssets();
  jobject local_asset_manager =
      env->CallObjectMethod(activity, activity_class_getAssets);

  return InitializeFromAssetManager(env, local_asset_manager, cache_dir_path);
}

bool AssetManager::FileExists(const std::string& filename) {
  if (!asset_manager_) {
    LOG(ERROR) << "Asset manager was not initialized from JNI";
    return false;
  }

  AAsset* asset =
      AAssetManager_open(asset_manager_, filename.c_str(), AASSET_MODE_RANDOM);
  if (asset != nullptr) {
    AAsset_close(asset);
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
    return dir_exists;
  }

  return false;
}

bool AssetManager::ReadFile(const std::string& filename,
                            std::vector<uint8_t>* raw_bytes) {
  CHECK(raw_bytes);
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
    raw_bytes->resize(size);
    memcpy(static_cast<void*>(&raw_bytes->at(0)), AAsset_getBuffer(asset),
           size);
    AAsset_close(asset);
  }
  return true;
}

::mediapipe::StatusOr<std::string> AssetManager::CachedFileFromAsset(
    const std::string& asset_path) {
  RET_CHECK(cache_dir_path_.size()) << "asset manager not initialized";

  std::string file_path =
      absl::StrCat(cache_dir_path_, "/mediapipe_asset_cache/", asset_path);

  // TODO: call the Java AssetCache, or make it call us.
  // For now, since we don't know the app version, we overwrite the cache file
  // unconditionally.

  std::vector<uint8_t> asset_data;
  RET_CHECK(ReadFile(asset_path, &asset_data))
      << "could not read asset: " << asset_path;

  std::string dir_path = File::StripBasename(file_path);
  MP_RETURN_IF_ERROR(file::RecursivelyCreateDir(dir_path, file::Defaults()));

  std::ofstream output_file(file_path);
  RET_CHECK(output_file.good()) << "could not open cache file: " << file_path;

  output_file.write(reinterpret_cast<char*>(asset_data.data()),
                    asset_data.size());
  RET_CHECK(output_file.good()) << "could not write cache file: " << file_path;

  return file_path;
}

}  // namespace mediapipe
