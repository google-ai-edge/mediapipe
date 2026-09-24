/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_RETRIEVAL_JNI_SQLITE_VECTOR_STORE_JNI_H_
#define MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_RETRIEVAL_JNI_SQLITE_VECTOR_STORE_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeCreateSqliteVectorStore(
    JNIEnv* env, jobject thiz, jint num_embedding_dimensions,
    jstring database_path);

JNIEXPORT void JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeInsert(
    JNIEnv* env, jobject thiz, jlong jni_handle,
    jbyteArray memory_record_bytes);

JNIEXPORT jobject JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeGetNearestRecords(
    JNIEnv* env, jobject thiz, jlong jni_handle, jfloatArray query_embeddings,
    jint top_k, jfloat min_similarity_score);

JNIEXPORT jobject JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeGetNearestRecordsWithFilter(
    JNIEnv* env, jobject thiz, jlong jni_handle, jfloatArray query_embeddings,
    jint top_k, jfloat min_similarity_score, jobjectArray filter_keys,
    jobjectArray filter_values);

JNIEXPORT jobject JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeGetRecords(
    JNIEnv* env, jobject thiz, jlong jni_handle, jobjectArray ids);

JNIEXPORT void JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeDeleteByMetadata(
    JNIEnv* env, jobject thiz, jlong jni_handle, jobjectArray filter_keys,
    jobjectArray filter_values);

JNIEXPORT jobject JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeGetAllRecordIds(
    JNIEnv* env, jobject thiz, jlong jni_handle);

JNIEXPORT void JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeDeleteRecords(
    JNIEnv* env, jobject thiz, jlong jni_handle, jobjectArray ids);

JNIEXPORT void JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeSqlQuery(
    JNIEnv* env, jobject thiz, jlong jni_handle, jstring query);

JNIEXPORT void JNICALL
Java_com_google_mediapipe_tasks_retrieval_components_SqliteVectorStore_nativeClose(
    JNIEnv* env, jobject thiz, jlong jni_handle);

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_TASKS_JAVA_COM_GOOGLE_MEDIAPIPE_TASKS_RETRIEVAL_JNI_SQLITE_VECTOR_STORE_JNI_H_
