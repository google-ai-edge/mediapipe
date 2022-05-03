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

#include "mediapipe/java/com/google/mediapipe/framework/jni/graph.h"

#include <pthread.h>

#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/framework/tool/executor_util.h"
#include "mediapipe/framework/tool/name_util.h"
#include "mediapipe/gpu/graph_support.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/class_registry.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/jni_util.h"
#include "mediapipe/java/com/google/mediapipe/framework/jni/packet_context_jni.h"
#ifdef __ANDROID__
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif  // __ANDROID__
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/egl_surface_holder.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {
namespace android {

namespace internal {
// PacketWithContext is the native counterpart of the Java Packet.
class PacketWithContext {
 public:
  PacketWithContext(Graph* context, const Packet& packet)
      : context_(context), packet_(packet) {}

  ~PacketWithContext() {}

  Graph* GetContext() { return context_; }

  Packet& packet() { return packet_; }

 private:
  Graph* context_;
  Packet packet_;
};

// A callback handler that wraps the java callback, and submits it for
// execution through Graph.
class CallbackHandler {
 public:
  CallbackHandler(Graph* context, jobject callback)
      : context_(context), java_callback_(callback) {}

  ~CallbackHandler() {
    // The jobject global reference is managed by the Graph directly.
    // So no-op here.
    if (java_callback_) {
      LOG(ERROR) << "Java callback global reference is not released.";
    }
  }

  void PacketCallback(const Packet& packet) {
    context_->CallbackToJava(mediapipe::java::GetJNIEnv(), java_callback_,
                             packet);
  }

  void PacketWithHeaderCallback(const Packet& packet, const Packet& header) {
    context_->CallbackToJava(mediapipe::java::GetJNIEnv(), java_callback_,
                             packet, header);
  }

  void PacketListCallback(const std::vector<Packet>& packets) {
    context_->CallbackToJava(mediapipe::java::GetJNIEnv(), java_callback_,
                             packets);
  }

  std::function<void(const Packet&)> CreateCallback() {
    return std::bind(&CallbackHandler::PacketCallback, this,
                     std::placeholders::_1);
  }

  std::function<void(const std::vector<Packet>&)> CreatePacketListCallback() {
    return std::bind(&CallbackHandler::PacketListCallback, this,
                     std::placeholders::_1);
  }

  std::function<void(const Packet&, const Packet&)> CreateCallbackWithHeader() {
    return std::bind(&CallbackHandler::PacketWithHeaderCallback, this,
                     std::placeholders::_1, std::placeholders::_2);
  }

  // Releases the global reference to the java callback object.
  // This is called by the Graph, since releasing of a jni object
  // requires JNIEnv object that we can not keep a copy of.
  void ReleaseCallback(JNIEnv* env) {
    env->DeleteGlobalRef(java_callback_);
    java_callback_ = nullptr;
  }

 private:
  Graph* context_;
  // java callback object
  jobject java_callback_;
};
}  // namespace internal

Graph::Graph()
    : executor_stack_size_increased_(false), global_java_packet_cls_(nullptr) {}

Graph::~Graph() {
  if (running_graph_) {
    running_graph_->Cancel();
    running_graph_->WaitUntilDone().IgnoreError();
  }
  // Cleans up the jni objects.
  JNIEnv* env = mediapipe::java::GetJNIEnv();
  if (env == nullptr) {
    LOG(ERROR) << "Can't attach to java thread, no jni clean up performed.";
    return;
  }
  for (const auto& handler : callback_handlers_) {
    handler->ReleaseCallback(env);
  }
  if (global_java_packet_cls_) {
    env->DeleteGlobalRef(global_java_packet_cls_);
    global_java_packet_cls_ = nullptr;
  }
}

int64_t Graph::WrapPacketIntoContext(const Packet& packet) {
  absl::MutexLock lock(&all_packets_mutex_);
  auto packet_context = new internal::PacketWithContext(this, packet);
  // Since the value of the all_packets_ map is a unique_ptr, resets it with the
  // new allocated object.
  all_packets_[packet_context].reset(packet_context);
  VLOG(2) << "Graph packet reference buffer size: " << all_packets_.size();
  return reinterpret_cast<int64_t>(packet_context);
}

// static
Packet Graph::GetPacketFromHandle(int64_t packet_handle) {
  internal::PacketWithContext* packet_with_context =
      reinterpret_cast<internal::PacketWithContext*>(packet_handle);
  return packet_with_context->packet();
}

// static
Graph* Graph::GetContextFromHandle(int64_t packet_handle) {
  internal::PacketWithContext* packet_with_context =
      reinterpret_cast<internal::PacketWithContext*>(packet_handle);
  return packet_with_context->GetContext();
}

// static
bool Graph::RemovePacket(int64_t packet_handle) {
  internal::PacketWithContext* packet_with_context =
      reinterpret_cast<internal::PacketWithContext*>(packet_handle);
  Graph* context = packet_with_context->GetContext();
  absl::MutexLock lock(&(context->all_packets_mutex_));
  return context->all_packets_.erase(packet_with_context) != 0;
}

void Graph::EnsureMinimumExecutorStackSizeForJava() {}

absl::Status Graph::AddCallbackHandler(std::string output_stream_name,
                                       jobject java_callback) {
  if (!graph_config()) {
    return absl::InternalError("Graph is not loaded!");
  }
  std::unique_ptr<internal::CallbackHandler> handler(
      new internal::CallbackHandler(this, java_callback));
  std::string side_packet_name;
  tool::AddCallbackCalculator(output_stream_name, graph_config(),
                              &side_packet_name,
                              /* use_std_function = */ true);
  EnsureMinimumExecutorStackSizeForJava();
  side_packets_callbacks_.emplace(
      side_packet_name, MakePacket<std::function<void(const Packet&)>>(
                            handler->CreateCallback()));
  callback_handlers_.emplace_back(std::move(handler));
  return absl::OkStatus();
}

absl::Status Graph::AddMultiStreamCallbackHandler(
    std::vector<std::string> output_stream_names, jobject java_callback,
    bool observe_timestamp_bounds) {
  if (!graph_config()) {
    return absl::InternalError("Graph is not loaded!");
  }
  auto handler =
      absl::make_unique<internal::CallbackHandler>(this, java_callback);
  tool::AddMultiStreamCallback(
      output_stream_names, handler->CreatePacketListCallback(), graph_config(),
      &side_packets_, observe_timestamp_bounds);
  EnsureMinimumExecutorStackSizeForJava();
  callback_handlers_.emplace_back(std::move(handler));
  return absl::OkStatus();
}

int64_t Graph::AddSurfaceOutput(const std::string& output_stream_name) {
  if (!graph_config()) {
    LOG(ERROR) << "Graph is not loaded!";
    return 0;
  }

#if MEDIAPIPE_DISABLE_GPU
  LOG(FATAL) << "GPU support has been disabled in this build!";
#else
  CalculatorGraphConfig::Node* sink_node = graph_config()->add_node();
  sink_node->set_name(mediapipe::tool::GetUnusedNodeName(
      *graph_config(), absl::StrCat("egl_surface_sink_", output_stream_name)));
  sink_node->set_calculator("GlSurfaceSinkCalculator");
  sink_node->add_input_stream(output_stream_name);
  sink_node->add_input_side_packet(
      absl::StrCat(kGpuSharedTagName, ":", kGpuSharedSidePacketName));

  const std::string input_side_packet_name =
      mediapipe::tool::GetUnusedSidePacketName(
          *graph_config(), absl::StrCat(output_stream_name, "_surface"));
  sink_node->add_input_side_packet(
      absl::StrCat("SURFACE:", input_side_packet_name));

  auto it_inserted = output_surface_side_packets_.emplace(
      input_side_packet_name,
      AdoptAsUniquePtr(new mediapipe::EglSurfaceHolder()));

  return WrapPacketIntoContext(it_inserted.first->second);
#endif  // MEDIAPIPE_DISABLE_GPU
}

absl::Status Graph::LoadBinaryGraph(std::string path_to_graph) {
  std::string graph_config_string;
  absl::Status status =
      mediapipe::file::GetContents(path_to_graph, &graph_config_string);
  if (!status.ok()) {
    return status;
  }
  return LoadBinaryGraph(graph_config_string.c_str(),
                         graph_config_string.length());
}

absl::Status Graph::LoadBinaryGraph(const char* data, int size) {
  CalculatorGraphConfig graph_config;
  if (!graph_config.ParseFromArray(data, size)) {
    return absl::InvalidArgumentError("Failed to parse the graph");
  }
  graph_configs_.push_back(graph_config);
  return absl::OkStatus();
}

absl::Status Graph::LoadBinaryGraphTemplate(const char* data, int size) {
  CalculatorGraphTemplate graph_template;
  if (!graph_template.ParseFromArray(data, size)) {
    return absl::InvalidArgumentError("Failed to parse the graph");
  }
  graph_templates_.push_back(graph_template);
  return absl::OkStatus();
}

absl::Status Graph::SetGraphType(std::string graph_type) {
  graph_type_ = graph_type;
  return absl::OkStatus();
}

absl::Status Graph::SetGraphOptions(const char* data, int size) {
  if (!graph_options_.ParseFromArray(data, size)) {
    return absl::InvalidArgumentError("Failed to parse the graph");
  }
  return absl::OkStatus();
}

CalculatorGraphConfig Graph::GetCalculatorGraphConfig() {
  CalculatorGraph temp_graph;
  absl::Status status = InitializeGraph(&temp_graph);
  if (!status.ok()) {
    LOG(ERROR) << "GetCalculatorGraphConfig failed:\n" << status.message();
  }
  return temp_graph.Config();
}

void Graph::CallbackToJava(JNIEnv* env, jobject java_callback_obj,
                           const Packet& packet) {
  jclass callback_cls = env->GetObjectClass(java_callback_obj);

  auto& class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string packet_class_name = class_registry.GetClassName(
      mediapipe::android::ClassRegistry::kPacketClassName);
  std::string process_method_name = class_registry.GetMethodName(
      mediapipe::android::ClassRegistry::kPacketCallbackClassName, "process");

  jmethodID processMethod =
      env->GetMethodID(callback_cls, process_method_name.c_str(),
                       absl::StrFormat("(L%s;)V", packet_class_name).c_str());

  int64_t packet_handle = WrapPacketIntoContext(packet);
  // Creates a Java Packet.
  VLOG(2) << "Creating java packet preparing for callback to java.";
  jobject java_packet =
      CreateJavaPacket(env, global_java_packet_cls_, packet_handle);
  VLOG(2) << "Calling java callback.";
  env->CallVoidMethod(java_callback_obj, processMethod, java_packet);
  // release the packet after callback.
  RemovePacket(packet_handle);
  env->DeleteLocalRef(callback_cls);
  env->DeleteLocalRef(java_packet);
  VLOG(2) << "Returned from java callback.";
}

void Graph::CallbackToJava(JNIEnv* env, jobject java_callback_obj,
                           const Packet& packet, const Packet& header_packet) {
  jclass callback_cls = env->GetObjectClass(java_callback_obj);

  auto& class_registry = mediapipe::android::ClassRegistry::GetInstance();
  std::string packet_class_name = class_registry.GetClassName(
      mediapipe::android::ClassRegistry::kPacketClassName);
  std::string process_method_name = class_registry.GetMethodName(
      mediapipe::android::ClassRegistry::kPacketWithHeaderCallbackClassName,
      "process");

  jmethodID processMethod = env->GetMethodID(
      callback_cls, process_method_name.c_str(),
      absl::StrFormat("(L%s;L%s;)V", packet_class_name, packet_class_name)
          .c_str());

  int64_t packet_handle = WrapPacketIntoContext(packet);
  int64_t header_packet_handle = WrapPacketIntoContext(header_packet);
  // Creates a Java Packet.
  jobject java_packet =
      CreateJavaPacket(env, global_java_packet_cls_, packet_handle);
  jobject java_header_packet =
      CreateJavaPacket(env, global_java_packet_cls_, header_packet_handle);
  env->CallVoidMethod(java_callback_obj, processMethod, java_packet,
                      java_header_packet);
  // release the packet after callback.
  RemovePacket(packet_handle);
  RemovePacket(header_packet_handle);
  env->DeleteLocalRef(callback_cls);
  env->DeleteLocalRef(java_packet);
  env->DeleteLocalRef(java_header_packet);
}

void Graph::CallbackToJava(JNIEnv* env, jobject java_callback_obj,
                           const std::vector<Packet>& packets) {
  jclass callback_cls = env->GetObjectClass(java_callback_obj);

  auto& class_registry = mediapipe::android::ClassRegistry::GetInstance();
  const std::string process_method_name = class_registry.GetMethodName(
      mediapipe::android::ClassRegistry::kPacketListCallbackClassName,
      "process");
  jmethodID processMethod = env->GetMethodID(
      callback_cls, process_method_name.c_str(), "(Ljava/util/List;)V");

  // TODO: move to register natives.
  jclass list_cls = env->FindClass("java/util/ArrayList");
  jobject java_list =
      env->NewObject(list_cls, env->GetMethodID(list_cls, "<init>", "()V"));
  jmethodID add_method =
      env->GetMethodID(list_cls, "add", "(Ljava/lang/Object;)Z");
  std::vector<int64_t> packet_handles;
  for (const Packet& packet : packets) {
    int64_t packet_handle = WrapPacketIntoContext(packet);
    packet_handles.push_back(packet_handle);
    jobject java_packet =
        CreateJavaPacket(env, global_java_packet_cls_, packet_handle);
    env->CallBooleanMethod(java_list, add_method, java_packet);
    env->DeleteLocalRef(java_packet);
  }

  VLOG(2) << "Calling java callback.";
  env->CallVoidMethod(java_callback_obj, processMethod, java_list);
  // release the packet after callback.
  for (int64_t packet_handle : packet_handles) {
    RemovePacket(packet_handle);
  }
  env->DeleteLocalRef(callback_cls);
  env->DeleteLocalRef(list_cls);
  env->DeleteLocalRef(java_list);
  VLOG(2) << "Returned from java callback.";
}

void Graph::SetPacketJavaClass(JNIEnv* env) {
  if (global_java_packet_cls_ == nullptr) {
    auto& class_registry = ClassRegistry::GetInstance();
    std::string packet_class_name = class_registry.GetClassName(
        mediapipe::android::ClassRegistry::kPacketClassName);
    jclass packet_cls = env->FindClass(packet_class_name.c_str());
    global_java_packet_cls_ =
        reinterpret_cast<jclass>(env->NewGlobalRef(packet_cls));
  }
}

absl::Status Graph::RunGraphUntilClose(JNIEnv* env) {
  // Get a global reference to the packet class, so it can be used in other
  // native thread for call back.
  SetPacketJavaClass(env);
  // Running as a synchronized mode, the same Java thread is available through
  // out the run.
  CalculatorGraph calculator_graph;
  absl::Status status = InitializeGraph(&calculator_graph);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
    running_graph_.reset(nullptr);
    return status;
  }
  // TODO: gpu & services set up!
  status = calculator_graph.Run(CreateCombinedSidePackets());
  LOG(INFO) << "Graph run finished.";

  return status;
}

absl::Status Graph::StartRunningGraph(JNIEnv* env) {
  if (running_graph_) {
    return absl::InternalError("Graph is already running.");
  }
  // Get a global reference to the packet class, so it can be used in other
  // native thread for call back.
  SetPacketJavaClass(env);
  // Running as a synchronized mode, the same Java thread is available
  // throughout the run.
  running_graph_.reset(new CalculatorGraph());
  // Set the mode for adding packets to graph input streams.
  running_graph_->SetGraphInputStreamAddMode(graph_input_stream_add_mode_);
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "input packet streams:";
    for (auto& name : graph_config()->input_stream()) {
      LOG(INFO) << name;
    }
  }
  absl::Status status;
#if !MEDIAPIPE_DISABLE_GPU
  if (gpu_resources_) {
    status = running_graph_->SetGpuResources(gpu_resources_);
    if (!status.ok()) {
      LOG(ERROR) << status.message();
      running_graph_.reset(nullptr);
      return status;
    }
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  for (const auto& service_packet : service_packets_) {
    status = running_graph_->SetServicePacket(*service_packet.first,
                                              service_packet.second);
    if (!status.ok()) {
      LOG(ERROR) << status.message();
      running_graph_.reset(nullptr);
      return status;
    }
  }

  status = InitializeGraph(running_graph_.get());
  if (!status.ok()) {
    LOG(ERROR) << status.message();
    running_graph_.reset(nullptr);
    return status;
  }
  LOG(INFO) << "Start running the graph, waiting for inputs.";
  status =
      running_graph_->StartRun(CreateCombinedSidePackets(), stream_headers_);
  if (!status.ok()) {
    LOG(ERROR) << status;
    running_graph_.reset(nullptr);
    return status;
  }
  return absl::OkStatus();
}

absl::Status Graph::SetTimestampAndMovePacketToInputStream(
    const std::string& stream_name, int64_t packet_handle, int64_t timestamp) {
  internal::PacketWithContext* packet_with_context =
      reinterpret_cast<internal::PacketWithContext*>(packet_handle);
  Packet& packet = packet_with_context->packet();

  // Set the timestamp of the packet in-place by calling the rvalue-reference
  // version of At here.
  packet = std::move(packet).At(Timestamp::CreateNoErrorChecking(timestamp));

  // Then std::move it into the input stream.
  return AddPacketToInputStream(stream_name, std::move(packet));
}

absl::Status Graph::AddPacketToInputStream(const std::string& stream_name,
                                           const Packet& packet) {
  if (!running_graph_) {
    return absl::FailedPreconditionError("Graph must be running.");
  }

  return running_graph_->AddPacketToInputStream(stream_name, packet);
}

absl::Status Graph::AddPacketToInputStream(const std::string& stream_name,
                                           Packet&& packet) {
  if (!running_graph_) {
    return absl::FailedPreconditionError("Graph must be running.");
  }

  return running_graph_->AddPacketToInputStream(stream_name, std::move(packet));
}

absl::Status Graph::CloseInputStream(std::string stream_name) {
  if (!running_graph_) {
    return absl::FailedPreconditionError("Graph must be running.");
  }
  LOG(INFO) << "Close input stream: " << stream_name;
  return running_graph_->CloseInputStream(stream_name);
}

absl::Status Graph::CloseAllInputStreams() {
  LOG(INFO) << "Close all input streams.";
  if (!running_graph_) {
    return absl::FailedPreconditionError("Graph must be running.");
  }
  return running_graph_->CloseAllInputStreams();
}

absl::Status Graph::CloseAllPacketSources() {
  LOG(INFO) << "Close all input streams.";
  if (!running_graph_) {
    return absl::FailedPreconditionError("Graph must be running.");
  }
  return running_graph_->CloseAllPacketSources();
}

absl::Status Graph::WaitUntilDone(JNIEnv* env) {
  if (!running_graph_) {
    return absl::FailedPreconditionError("Graph must be running.");
  }
  absl::Status status = running_graph_->WaitUntilDone();
  running_graph_.reset(nullptr);
  return status;
}

absl::Status Graph::WaitUntilIdle(JNIEnv* env) {
  if (!running_graph_) {
    return absl::FailedPreconditionError("Graph must be running.");
  }
  return running_graph_->WaitUntilIdle();
}

void Graph::SetInputSidePacket(const std::string& stream_name,
                               const Packet& packet) {
  side_packets_[stream_name] = packet;
}

void Graph::SetStreamHeader(const std::string& stream_name,
                            const Packet& packet) {
  stream_headers_[stream_name] = packet;
  LOG(INFO) << stream_name << " stream header being set.";
}

void Graph::SetGraphInputStreamAddMode(
    CalculatorGraph::GraphInputStreamAddMode mode) {
  graph_input_stream_add_mode_ = mode;
}

#if !MEDIAPIPE_DISABLE_GPU
mediapipe::GpuResources* Graph::GetGpuResources() const {
  return gpu_resources_.get();
}
#endif  // !MEDIAPIPE_DISABLE_GPU

absl::Status Graph::SetParentGlContext(int64 java_gl_context) {
#if MEDIAPIPE_DISABLE_GPU
  LOG(FATAL) << "GPU support has been disabled in this build!";
#else
  if (gpu_resources_) {
    return absl::AlreadyExistsError(
        "trying to set the parent GL context, but the gpu shared "
        "data has already been set up.");
  }
  ASSIGN_OR_RETURN(gpu_resources_,
                   mediapipe::GpuResources::Create(
                       reinterpret_cast<EGLContext>(java_gl_context)));
#endif  // MEDIAPIPE_DISABLE_GPU
  return absl::OkStatus();
}

void Graph::SetServicePacket(const GraphServiceBase& service, Packet packet) {
  service_packets_[&service] = std::move(packet);
}

void Graph::CancelGraph() {
  if (running_graph_) {
    running_graph_->Cancel();
  }
}

std::map<std::string, Packet> Graph::CreateCombinedSidePackets() {
  std::map<std::string, Packet> combined_side_packets = side_packets_callbacks_;
  combined_side_packets.insert(side_packets_.begin(), side_packets_.end());
  combined_side_packets.insert(output_surface_side_packets_.begin(),
                               output_surface_side_packets_.end());
  return combined_side_packets;
}

ProfilingContext* Graph::GetProfilingContext() {
  if (running_graph_) {
    return running_graph_->profiler();
  }
  return nullptr;
}

CalculatorGraphConfig* Graph::graph_config() {
  // Return the last specified graph config with the required graph_type.
  for (auto it = graph_configs_.rbegin(); it != graph_configs_.rend(); ++it) {
    if (it->type() == graph_type()) {
      return &*it;
    }
  }
  for (auto it = graph_templates_.rbegin(); it != graph_templates_.rend();
       ++it) {
    if (it->mutable_config()->type() == graph_type()) {
      return it->mutable_config();
    }
  }
  return nullptr;
}

std::string Graph::graph_type() {
  // If a graph-type is specified, that type is used.  Otherwise the
  // graph-type of the last specified graph config is used.
  if (graph_type_ != "<none>") {
    return graph_type_;
  }
  if (!graph_configs_.empty()) {
    return graph_configs_.back().type();
  }
  if (!graph_templates_.empty()) {
    return graph_templates_.back().config().type();
  }
  return "";
}

absl::Status Graph::InitializeGraph(CalculatorGraph* graph) {
  if (graph_configs_.size() == 1 && graph_templates_.empty()) {
    return graph->Initialize(*graph_config());
  } else {
    return graph->Initialize(graph_configs_, graph_templates_, {}, graph_type(),
                             &graph_options_);
  }
}

}  // namespace android
}  // namespace mediapipe
