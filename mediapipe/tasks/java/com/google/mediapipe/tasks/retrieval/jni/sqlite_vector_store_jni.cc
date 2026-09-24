// Copyright 2025 The Google AI Edge Authors.
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

#include "mediapipe/tasks/java/com/google/mediapipe/tasks/retrieval/jni/sqlite_vector_store_jni.h"

#include <jni.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/proto/vector_stores.pb.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/sqlite_memory_store.h"
#include "mediapipe/tasks/cc/retrieval/semantic_retriever/sqlite_vector_store.h"
#include "util/java/jni_helper.h"

namespace {

using ::mediapipe::tasks::retrieval::MemoryRecord;
using ::mediapipe::tasks::retrieval::TableConfig;

void ThrowException(const std::string& exception_class,
                    const std::string& message,
                    util::java::ThrowingJniHelper* jni_helper) {
  jni_helper->ThrowNew(jni_helper->FindClass(exception_class.c_str()).get(),
                       message.c_str());
}

absl::flat_hash_map<std::string, std::string> ParseMetadataFilter(
    JNIEnv* env, jobjectArray j_keys, jobjectArray j_values) {
  absl::flat_hash_map<std::string, std::string> filter;
  if (j_keys == nullptr || j_values == nullptr) {
    return filter;
  }
  int num_keys = env->GetArrayLength(j_keys);
  int num_values = env->GetArrayLength(j_values);
  if (num_keys != num_values) {
    util::java::ThrowingJniHelper jni_helper(env);
    ThrowException(
        "java/lang/IllegalArgumentException",
        "Keys and values array lengths must match in metadata filter",
        &jni_helper);
    return filter;
  }
  for (int i = 0; i < num_keys; ++i) {
    jstring j_key = static_cast<jstring>(env->GetObjectArrayElement(j_keys, i));
    jstring j_val =
        static_cast<jstring>(env->GetObjectArrayElement(j_values, i));
    if (j_key != nullptr && j_val != nullptr) {
      const char* key_chars = env->GetStringUTFChars(j_key, nullptr);
      const char* val_chars = env->GetStringUTFChars(j_val, nullptr);
      if (key_chars != nullptr && val_chars != nullptr) {
        filter[key_chars] = val_chars;
      }
      if (key_chars != nullptr) {
        env->ReleaseStringUTFChars(j_key, key_chars);
      }
      if (val_chars != nullptr) {
        env->ReleaseStringUTFChars(j_val, val_chars);
      }
    }
    if (j_key != nullptr) {
      env->DeleteLocalRef(j_key);
    }
    if (j_val != nullptr) {
      env->DeleteLocalRef(j_val);
    }
  }
  return filter;
}

MemoryRecord ParseMemoryRecord(JNIEnv* env, jbyteArray memory_record_bytes) {
  util::java::ThrowingJniHelper jni_helper(env);
  jbyte* request_ref = env->GetByteArrayElements(memory_record_bytes, nullptr);
  int request_size = env->GetArrayLength(memory_record_bytes);
  MemoryRecord memory_record;
  if (!memory_record.ParseFromArray(request_ref, request_size)) {
    ThrowException("java/lang/RuntimeException",
                   "Failed to parse memory record", &jni_helper);
  }
  env->ReleaseByteArrayElements(memory_record_bytes, request_ref, JNI_ABORT);
  return memory_record;
}

std::vector<float> ParseFloatArray(JNIEnv* env, jfloatArray float_array) {
  jfloat* float_array_ref = env->GetFloatArrayElements(float_array, nullptr);
  int float_array_size = env->GetArrayLength(float_array);
  std::vector<float> float_vector(float_array_ref,
                                  float_array_ref + float_array_size);
  env->ReleaseFloatArrayElements(float_array, float_array_ref, JNI_ABORT);
  return float_vector;
}

jobject ConvertRecordProtosToArrayListOfProtoBytes(
    JNIEnv* env, std::vector<MemoryRecord> records) {
  jclass array_list_class = env->FindClass("java/util/ArrayList");
  jmethodID array_list_ctor =
      env->GetMethodID(array_list_class, "<init>", "(I)V");
  jint initial_capacity = static_cast<jint>(records.size());
  jobject array_list_object =
      env->NewObject(array_list_class, array_list_ctor, initial_capacity);
  jmethodID array_list_add_method =
      env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");
  for (const MemoryRecord& record : records) {
    size_t record_size = record.ByteSizeLong();
    char* record_bytes = new char[record_size];
    record.SerializeToArray(record_bytes, record_size);
    jbyteArray byte_array_object = env->NewByteArray(record_size);
    env->SetByteArrayRegion(byte_array_object, 0, record_size,
                            reinterpret_cast<const jbyte*>(record_bytes));
    delete[] record_bytes;
    env->CallBooleanMethod(array_list_object, array_list_add_method,
                           byte_array_object);
    env->DeleteLocalRef(byte_array_object);
  }
  return array_list_object;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define SQLITE_METHOD(METHOD_NAME) \
  Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_##METHOD_NAME  // NOLINT

JNIEXPORT jlong JNICALL SQLITE_METHOD(nativeCreateSqliteVectorStore)(
    JNIEnv* env, jobject thiz, jint num_embedding_dimensions,
    jstring database_path) {
  util::java::ThrowingJniHelper jni_helper(env);

  // Initialize the vector store to the given database path.
  auto sqlite_vector_store =
      std::make_unique<mediapipe::tasks::retrieval::SqliteVectorStore>();
  auto path = jni_helper.JStringToString(database_path);
  auto statusVectorInit = sqlite_vector_store->Initialize(path);
  if (!statusVectorInit.ok()) {
    ThrowException("java/lang/RuntimeException", statusVectorInit.ToString(),
                   &jni_helper);
  }

  // Initialize the memory store with the vector store and the given config.
  auto sqlite_memory_store =
      std::make_unique<mediapipe::tasks::retrieval::SqliteMemoryStore>(
          std::move(sqlite_vector_store));
  auto config =
      mediapipe::tasks::retrieval::SqliteMemoryStore::GetDefaultConfig(
          num_embedding_dimensions);
  auto statusMemoryInit = sqlite_memory_store->Initialize(config);
  if (!statusMemoryInit.ok()) {
    ThrowException("java/lang/RuntimeException", statusMemoryInit.ToString(),
                   &jni_helper);
  }

  return reinterpret_cast<jlong>(sqlite_memory_store.release());
}

JNIEXPORT void JNICALL
SQLITE_METHOD(nativeInsert)(JNIEnv* env, jobject thiz, jlong jni_handle,
                            jbyteArray memory_record_bytes) {
  util::java::ThrowingJniHelper jni_helper(env);
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  mediapipe::tasks::retrieval::MemoryRecord memory_record =
      ParseMemoryRecord(env, memory_record_bytes);
  auto status = sqlite_memory_store->Insert(memory_record);
  if (!status.ok()) {
    ThrowException("java/lang/RuntimeException", status.ToString(),
                   &jni_helper);
  }
}

JNIEXPORT jobject JNICALL SQLITE_METHOD(nativeGetNearestRecords)(
    JNIEnv* env, jobject thiz, jlong jni_handle, jfloatArray query_embeddings,
    jint top_k, jfloat min_similarity_score) {
  util::java::ThrowingJniHelper jni_helper(env);
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  auto recordsStatus = sqlite_memory_store->GetNearestRecords(
      ParseFloatArray(env, query_embeddings), top_k, min_similarity_score);
  if (!recordsStatus.ok()) {
    ThrowException("java/lang/RuntimeException",
                   recordsStatus.status().ToString(), &jni_helper);
    return nullptr;
  }
  return ConvertRecordProtosToArrayListOfProtoBytes(env, recordsStatus.value());
}

JNIEXPORT jobject JNICALL SQLITE_METHOD(nativeGetRecords)(JNIEnv* env,
                                                          jobject thiz,
                                                          jlong jni_handle,
                                                          jobjectArray j_ids) {
  util::java::ThrowingJniHelper jni_helper(env);
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  std::vector<std::string> ids;
  int num_ids = env->GetArrayLength(j_ids);
  for (int i = 0; i < num_ids; ++i) {
    jstring j_id = static_cast<jstring>(env->GetObjectArrayElement(j_ids, i));
    const char* id_chars = env->GetStringUTFChars(j_id, nullptr);
    if (id_chars != nullptr) {
      ids.push_back(id_chars);
      env->ReleaseStringUTFChars(j_id, id_chars);
    }
    env->DeleteLocalRef(j_id);
  }

  auto recordsStatus = sqlite_memory_store->GetRecords(ids);
  if (!recordsStatus.ok()) {
    ThrowException("java/lang/RuntimeException",
                   recordsStatus.status().ToString(), &jni_helper);
    return nullptr;
  }
  return ConvertRecordProtosToArrayListOfProtoBytes(env, recordsStatus.value());
}

JNIEXPORT void JNICALL SQLITE_METHOD(nativeSqlQuery)(JNIEnv* env, jobject thiz,
                                                     jlong jni_handle,
                                                     jstring query) {
  util::java::ThrowingJniHelper jni_helper(env);
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  auto result = sqlite_memory_store->Execute(jni_helper.JStringToString(query));
  if (!result.ok()) {
    ThrowException("java/lang/RuntimeException", result.ToString(),
                   &jni_helper);
  }
}

JNIEXPORT void JNICALL SQLITE_METHOD(nativeDeleteRecords)(JNIEnv* env,
                                                          jobject thiz,
                                                          jlong jni_handle,
                                                          jobjectArray j_ids) {
  util::java::ThrowingJniHelper jni_helper(env);
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  if (j_ids == nullptr) {
    return;
  }
  int num_ids = env->GetArrayLength(j_ids);
  for (int i = 0; i < num_ids; ++i) {
    jstring j_id = static_cast<jstring>(env->GetObjectArrayElement(j_ids, i));
    if (j_id != nullptr) {
      const char* id_chars = env->GetStringUTFChars(j_id, nullptr);
      if (id_chars != nullptr) {
        auto status = sqlite_memory_store->DeleteById(id_chars);
        env->ReleaseStringUTFChars(j_id, id_chars);
        if (!status.ok()) {
          env->DeleteLocalRef(j_id);
          ThrowException("java/lang/RuntimeException", status.ToString(),
                         &jni_helper);
          return;
        }
      }
      env->DeleteLocalRef(j_id);
    }
  }
}

JNIEXPORT jobject JNICALL SQLITE_METHOD(nativeGetNearestRecordsWithFilter)(
    JNIEnv* env, jobject thiz, jlong jni_handle, jfloatArray query_embeddings,
    jint top_k, jfloat min_similarity_score, jobjectArray filter_keys,
    jobjectArray filter_values) {
  util::java::ThrowingJniHelper jni_helper(env);
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  auto filter = ParseMetadataFilter(env, filter_keys, filter_values);
  if (env->ExceptionCheck()) {
    return nullptr;
  }
  auto recordsStatus = sqlite_memory_store->GetNearestRecords(
      ParseFloatArray(env, query_embeddings), top_k, min_similarity_score,
      filter);
  if (!recordsStatus.ok()) {
    ThrowException("java/lang/RuntimeException",
                   recordsStatus.status().ToString(), &jni_helper);
    return nullptr;
  }
  return ConvertRecordProtosToArrayListOfProtoBytes(env, recordsStatus.value());
}

JNIEXPORT void JNICALL SQLITE_METHOD(nativeDeleteByMetadata)(
    JNIEnv* env, jobject thiz, jlong jni_handle, jobjectArray filter_keys,
    jobjectArray filter_values) {
  util::java::ThrowingJniHelper jni_helper(env);
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  auto filter = ParseMetadataFilter(env, filter_keys, filter_values);
  if (env->ExceptionCheck()) {
    return;
  }
  auto status = sqlite_memory_store->DeleteByMetadata(filter);
  if (!status.ok()) {
    ThrowException("java/lang/RuntimeException", status.ToString(),
                   &jni_helper);
  }
}

JNIEXPORT jobject JNICALL SQLITE_METHOD(nativeGetAllRecordIds)(
    JNIEnv* env, jobject thiz, jlong jni_handle) {
  util::java::ThrowingJniHelper jni_helper(env);
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  auto idsStatus = sqlite_memory_store->GetAllRecordIds();
  if (!idsStatus.ok()) {
    ThrowException("java/lang/RuntimeException", idsStatus.status().ToString(),
                   &jni_helper);
    return nullptr;
  }
  jclass array_list_class = env->FindClass("java/util/ArrayList");
  jmethodID array_list_constructor =
      env->GetMethodID(array_list_class, "<init>", "()V");
  jobject array_list_obj =
      env->NewObject(array_list_class, array_list_constructor);
  jmethodID array_list_add =
      env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");
  for (const std::string& id : idsStatus.value()) {
    jstring j_str = env->NewStringUTF(id.c_str());
    env->CallBooleanMethod(array_list_obj, array_list_add, j_str);
    env->DeleteLocalRef(j_str);
  }
  return array_list_obj;
}

JNIEXPORT void JNICALL SQLITE_METHOD(nativeClose)(JNIEnv* env, jobject thiz,
                                                  jlong jni_handle) {
  auto sqlite_memory_store =
      reinterpret_cast<mediapipe::tasks::retrieval::SqliteMemoryStore*>(
          jni_handle);
  delete sqlite_memory_store;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
