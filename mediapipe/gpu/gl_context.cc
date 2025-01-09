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

#include "mediapipe/gpu/gl_context.h"

#include <sys/types.h>

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/dynamic_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port.h"  // IWYU pragma: keep
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/gpu/gl_context_internal.h"
#include "mediapipe/gpu/gpu_buffer_format.h"

#ifndef __EMSCRIPTEN__
#include "absl/debugging/leak_check.h"
#include "mediapipe/gpu/gl_thread_collector.h"
#endif

#ifndef GL_MAJOR_VERSION
#define GL_MAJOR_VERSION 0x821B
#endif

#ifndef GL_MINOR_VERSION
#define GL_MINOR_VERSION 0x821C
#endif

namespace mediapipe {

namespace internal_gl_context {

bool IsOpenGlVersionSameOrAbove(const OpenGlVersion& version,
                                const OpenGlVersion& expected_version) {
  return (version.major == expected_version.major &&
          version.minor >= expected_version.minor) ||
         version.major > expected_version.major;
}

}  // namespace internal_gl_context

static void SetThreadName(const char* name) {
#if defined(__GLIBC_PREREQ)
#define LINUX_STYLE_SETNAME_NP __GLIBC_PREREQ(2, 12)
#elif defined(__BIONIC__)
#define LINUX_STYLE_SETNAME_NP 1
#endif  // __GLIBC_PREREQ
#if LINUX_STYLE_SETNAME_NP
  char thread_name[16];  // Linux requires names (with nul) fit in 16 chars
  strncpy(thread_name, name, sizeof(thread_name));
  thread_name[sizeof(thread_name) - 1] = '\0';
  int res = pthread_setname_np(pthread_self(), thread_name);
  if (res != 0) {
    ABSL_LOG_FIRST_N(INFO, 1)
        << "Can't set pthread names: name: \"" << name << "\"; error: " << res;
  }
#elif __APPLE__
  pthread_setname_np(name);
#endif
  ABSL_ANNOTATE_THREAD_NAME(name);
}

GlContext::DedicatedThread::DedicatedThread() {
  ABSL_CHECK_EQ(pthread_create(&gl_thread_id_, nullptr, ThreadBody, this), 0);
}

GlContext::DedicatedThread::~DedicatedThread() {
  if (IsCurrentThread()) {
    ABSL_CHECK(self_destruct_);
    ABSL_CHECK_EQ(pthread_detach(gl_thread_id_), 0);
  } else {
    // Give an invalid job to signal termination.
    PutJob({});
    ABSL_CHECK_EQ(pthread_join(gl_thread_id_, nullptr), 0);
  }
}

void GlContext::DedicatedThread::SelfDestruct() {
  self_destruct_ = true;
  // Give an invalid job to signal termination.
  PutJob({});
}

GlContext::DedicatedThread::Job GlContext::DedicatedThread::GetJob() {
  absl::MutexLock lock(&mutex_);
  while (jobs_.empty()) {
    has_jobs_cv_.Wait(&mutex_);
  }
  Job job = std::move(jobs_.front());
  jobs_.pop_front();
  return job;
}

void GlContext::DedicatedThread::PutJob(Job job) {
  absl::MutexLock lock(&mutex_);
  jobs_.push_back(std::move(job));
  has_jobs_cv_.SignalAll();
}

void* GlContext::DedicatedThread::ThreadBody(void* instance) {
  DedicatedThread* thread = static_cast<DedicatedThread*>(instance);
  thread->ThreadBody();
  return nullptr;
}

#ifdef __APPLE__
#define AUTORELEASEPOOL @autoreleasepool
#else
#define AUTORELEASEPOOL
#endif  // __APPLE__

void GlContext::DedicatedThread::ThreadBody() {
  SetThreadName("mediapipe_gl_runner");

#ifndef __EMSCRIPTEN__
  GlThreadCollector::ThreadStarting();
#endif
  // The dedicated GL thread is not meant to be used on Apple platforms, but
  // in case it is, the use of an autorelease pool here will reap each task's
  // temporary allocations.
  while (true) AUTORELEASEPOOL {
      Job job = GetJob();
      // Lack of a job means termination. Or vice versa.
      if (!job) {
        break;
      }
      job();
    }
  if (self_destruct_) {
    delete this;
  }
#ifndef __EMSCRIPTEN__
  GlThreadCollector::ThreadEnding();
#endif
}

absl::Status GlContext::DedicatedThread::Run(GlStatusFunction gl_func) {
  // Neither ENDO_SCOPE nor ENDO_TASK seem to work here.
  if (IsCurrentThread()) {
    return gl_func();
  }
  bool done = false;  // Guarded by mutex_ after initialization.
  absl::Status status;
  PutJob([this, gl_func, &done, &status]() {
    status = gl_func();
    absl::MutexLock lock(&mutex_);
    done = true;
    gl_job_done_cv_.SignalAll();
  });

  absl::MutexLock lock(&mutex_);
  while (!done) {
    gl_job_done_cv_.Wait(&mutex_);
  }
  return status;
}

void GlContext::DedicatedThread::RunWithoutWaiting(GlVoidFunction gl_func) {
  // Note: this is invoked by GlContextExecutor. To avoid starvation of
  // non-calculator tasks in the presence of GL source calculators, calculator
  // tasks must always be scheduled as new tasks, or another solution needs to
  // be set up to avoid starvation. See b/78522434.
  ABSL_CHECK(gl_func);
  PutJob(std::move(gl_func));
}

bool GlContext::DedicatedThread::IsCurrentThread() {
  return pthread_equal(gl_thread_id_, pthread_self());
}

bool GlContext::ParseGlVersion(absl::string_view version_string, GLint* major,
                               GLint* minor) {
  size_t pos = version_string.find('.');
  if (pos == absl::string_view::npos || pos < 1) {
    return false;
  }
  // GL_VERSION is supposed to start with the version number; see, e.g.,
  // https://www.khronos.org/registry/OpenGL-Refpages/es3/html/glGetString.xhtml
  // https://www.khronos.org/opengl/wiki/OpenGL_Context#OpenGL_version_number
  // However, in rare cases one will encounter non-conforming configurations
  // that have some prefix before the number. To deal with that, we walk
  // backwards from the dot.
  size_t start = pos - 1;
  while (start > 0 && isdigit(version_string[start - 1])) --start;
  if (!absl::SimpleAtoi(version_string.substr(start, (pos - start)), major)) {
    return false;
  }
  auto rest = version_string.substr(pos + 1);
  pos = rest.find(' ');
  size_t pos2 = rest.find('.');
  if (pos == absl::string_view::npos ||
      (pos2 != absl::string_view::npos && pos2 < pos)) {
    pos = pos2;
  }
  if (!absl::SimpleAtoi(rest.substr(0, pos), minor)) {
    return false;
  }
  return true;
}

GlVersion GlContext::GetGlVersion() const {
#ifdef GL_ES_VERSION_2_0  // This actually means "is GLES available".
  return gl_major_version() < 3 ? GlVersion::kGLES2 : GlVersion::kGLES3;
#else  // This is the "desktop GL" case.
  return GlVersion::kGL;
#endif
}

bool GlContext::HasGlExtension(absl::string_view extension) const {
  return gl_extensions_.find(extension) != gl_extensions_.end();
}

// Function for GL3.0+ to query for and store all of our available GL extensions
// in an easily-accessible set.  The glGetString call is actually *not* required
// to work with GL_EXTENSIONS for newer GL versions, so we must maintain both
// variations of this function.
absl::Status GlContext::GetGlExtensions() {
  // RET_CHECK logs by default, but here we just want to check the precondition;
  // we'll fall back to the alternative implementation for older versions.
  RET_CHECK(gl_major_version_ >= 3).SetNoLogging();
  gl_extensions_.clear();
  // glGetStringi only introduced in GL 3.0+; so we exit out this function if
  // we don't have that function defined, regardless of version number reported.
  // The function itself is also fully stubbed out if we're linking against an
  // API version without a glGetStringi declaration. Although Emscripten
  // sometimes provides this function, its default library implementation
  // appears to only provide glGetString, so we skip this for Emscripten
  // platforms to avoid possible undefined symbol or runtime errors.
#if (GL_VERSION_3_0 || GL_ES_VERSION_3_0) && !defined(__EMSCRIPTEN__)
  if (!SymbolAvailable(&glGetStringi)) {
    ABSL_LOG(ERROR)
        << "GL major version > 3.0 indicated, but glGetStringi not "
        << "defined. Falling back to deprecated GL extensions querying "
        << "method.";
    return absl::InternalError("glGetStringi not defined, but queried");
  }
  int num_extensions = 0;
  glGetIntegerv(GL_NUM_EXTENSIONS, &num_extensions);
  if (glGetError() != 0) {
    return absl::InternalError("Error querying for number of extensions");
  }

  for (int i = 0; i < num_extensions; ++i) {
    const GLubyte* res = glGetStringi(GL_EXTENSIONS, i);
    if (glGetError() != 0 || res == nullptr) {
      return absl::InternalError("Error querying for an extension by index");
    }
    const char* signed_res = reinterpret_cast<const char*>(res);
    gl_extensions_.insert(signed_res);
  }

  return absl::OkStatus();
#else
  return absl::InternalError("GL version mismatch in GlGetExtensions");
#endif  // (GL_VERSION_3_0 || GL_ES_VERSION_3_0) && !defined(__EMSCRIPTEN__)
}

// Same as GetGlExtensions() above, but for pre-GL3.0, where glGetStringi did
// not exist.
absl::Status GlContext::GetGlExtensionsCompat() {
  gl_extensions_.clear();

  const GLubyte* res = glGetString(GL_EXTENSIONS);
  if (glGetError() != 0 || res == nullptr) {
    ABSL_LOG(ERROR) << "Error querying for GL extensions";
    return absl::InternalError("Error querying for GL extensions");
  }
  const char* signed_res = reinterpret_cast<const char*>(res);
  gl_extensions_ = absl::StrSplit(signed_res, ' ');

  return absl::OkStatus();
}

absl::Status GlContext::FinishInitialization(bool create_thread) {
  if (create_thread) {
    thread_ = absl::make_unique<GlContext::DedicatedThread>();
    MP_RETURN_IF_ERROR(thread_->Run([this] { return EnterContext(nullptr); }));
  }

  return Run([this]() -> absl::Status {
    // Clear any GL errors at this point: as this is a fresh context
    // there shouldn't be any, but if we adopted an existing context (e.g. in
    // some Emscripten cases), there might be some existing tripped error.
    ForceClearExistingGlErrors();

    absl::string_view version_string;
    const GLubyte* version_string_ptr = glGetString(GL_VERSION);
    if (version_string_ptr != nullptr) {
      version_string = reinterpret_cast<const char*>(version_string_ptr);
    } else {
      // This may happen when using SwiftShader, but the numeric versions are
      // available and will be used instead.
      ABSL_LOG(WARNING) << "failed to get GL_VERSION string";
    }

    // We will decide later whether we want to use the version numbers we query
    // for, or instead derive that information from the context creation result,
    // which we cache here.
    GLint gl_major_version_from_context_creation = gl_major_version_;

    // Let's try getting the numeric version if possible.
    glGetIntegerv(GL_MAJOR_VERSION, &gl_major_version_);
    GLenum err = glGetError();
    if (err == GL_NO_ERROR) {
      glGetIntegerv(GL_MINOR_VERSION, &gl_minor_version_);
    } else {
      // GL_MAJOR_VERSION is not supported on GL versions below 3. We have to
      // parse the version string.
      if (!ParseGlVersion(version_string, &gl_major_version_,
                          &gl_minor_version_)) {
        ABSL_LOG(WARNING) << "invalid GL_VERSION format: '" << version_string
                          << "'; assuming 2.0";
        gl_major_version_ = 2;
        gl_minor_version_ = 0;
      }
    }

    // If our platform-specific CreateContext already set a major GL version,
    // then we use that.  Otherwise, we use the queried-for result. We do this
    // as a workaround for a Swiftshader on Android bug where the ES2 context
    // can report major version 3 instead of 2 when queried. Therefore we trust
    // the result from context creation more than from query. See b/152519932
    // for more details.
    if (gl_major_version_from_context_creation > 0 &&
        gl_major_version_ != gl_major_version_from_context_creation) {
      ABSL_LOG(WARNING) << "Requested a context with major GL version "
                        << gl_major_version_from_context_creation
                        << " but context reports major version "
                        << gl_major_version_ << ". Setting to "
                        << gl_major_version_from_context_creation << ".0";
      gl_major_version_ = gl_major_version_from_context_creation;
      gl_minor_version_ = 0;
    }

    ABSL_LOG(INFO) << "GL version: " << gl_major_version_ << "."
                   << gl_minor_version_ << " (" << version_string
                   << "), renderer: " << glGetString(GL_RENDERER);

    {
      auto status = GetGlExtensions();
      if (!status.ok()) {
        status = GetGlExtensionsCompat();
      }
      MP_RETURN_IF_ERROR(status);
    }

#if GL_ES_VERSION_2_0  // This actually means "is GLES available".
    // No linear float filtering by default, check extensions.
    can_linear_filter_float_textures_ =
        HasGlExtension("OES_texture_float_linear") ||
        HasGlExtension("GL_OES_texture_float_linear");
#else
    // Desktop GL should always allow linear filtering.
    can_linear_filter_float_textures_ = true;
#endif  // GL_ES_VERSION_2_0

    return absl::OkStatus();
  });
}

GlContext::GlContext() = default;

GlContext::~GlContext() {
  destructing_ = true;
  // Note: on Apple platforms, this object contains Objective-C objects.
  // The destructor will release them, but ARC must be on.
#ifdef __OBJC__
#if !__has_feature(objc_arc)
#error This file must be built with ARC.
#endif
#endif  // __OBJC__

  auto clear_attachments = [this] {
    attachments_.clear();
    if (profiling_helper_) {
      profiling_helper_->LogAllTimestamps();
    }
  };

  if (thread_) {
    auto status = thread_->Run([this, clear_attachments] {
      clear_attachments();
      return ExitContext(nullptr);
    });
    ABSL_LOG_IF(ERROR, !status.ok())
        << "Failed to deactivate context on thread: " << status;
    if (thread_->IsCurrentThread()) {
      thread_.release()->SelfDestruct();
    }
  } else if (IsCurrent()) {
    clear_attachments();
  } else if (HasContext()) {
    ContextBinding saved_context;
    auto status = SwitchContextAndRun([&clear_attachments] {
      clear_attachments();
      return absl::OkStatus();
    });
    ABSL_LOG_IF(ERROR, !status.ok()) << status;
  }
  DestroyContext();
}

void GlContext::SetProfilingContext(
    std::shared_ptr<mediapipe::ProfilingContext> profiling_context) {
  // Create the GlProfilingHelper if it is uninitialized.
  if (!profiling_helper_ && profiling_context) {
    profiling_helper_ = profiling_context->CreateGlProfilingHelper();
  }
}

absl::Status GlContext::SwitchContextAndRun(GlStatusFunction gl_func) {
  ContextBinding saved_context;
  MP_RETURN_IF_ERROR(EnterContext(&saved_context)) << " (entering GL context)";
  auto status = gl_func();
  LogUncheckedGlErrors(CheckForGlErrors());
  MP_RETURN_IF_ERROR(ExitContext(&saved_context)) << " (exiting GL context)";
  return status;
}

absl::Status GlContext::Run(GlStatusFunction gl_func, int node_id,
                            Timestamp input_timestamp) {
  absl::Status status;
  if (profiling_helper_) {
    gl_func = [=] {
      profiling_helper_->MarkTimestamp(node_id, input_timestamp,
                                       /*is_finish=*/false);
      auto status = gl_func();
      profiling_helper_->MarkTimestamp(node_id, input_timestamp,
                                       /*is_finish=*/true);
      return status;
    };
  }
  if (thread_) {
    bool had_gl_errors = false;
    status = thread_->Run([this, gl_func, &had_gl_errors] {
      auto status = gl_func();
      had_gl_errors = CheckForGlErrors();
      return status;
    });
    LogUncheckedGlErrors(had_gl_errors);
  } else {
    status = SwitchContextAndRun(gl_func);
  }
  return status;
}

void GlContext::RunWithoutWaiting(GlVoidFunction gl_func) {
  if (thread_) {
    // Add ref to keep the context alive while the task is executing.
    auto context = shared_from_this();
    thread_->RunWithoutWaiting([this, context, gl_func] {
      gl_func();
      LogUncheckedGlErrors(CheckForGlErrors());
    });
  } else {
    // TODO: queue up task instead.
    auto status = SwitchContextAndRun([gl_func] {
      gl_func();
      return absl::OkStatus();
    });
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Error in RunWithoutWaiting: " << status;
    }
  }
}

std::weak_ptr<GlContext>& GlContext::CurrentContext() {
  // Workaround for b/67878799.
#ifndef __EMSCRIPTEN__
  absl::LeakCheckDisabler disable_leak_check;
#endif
  ABSL_CONST_INIT thread_local std::weak_ptr<GlContext> current_context;
  return current_context;
}

absl::Status GlContext::SwitchContext(ContextBinding* saved_context,
                                      const ContextBinding& new_context)
    ABSL_NO_THREAD_SAFETY_ANALYSIS {
  std::shared_ptr<GlContext> old_context_obj = CurrentContext().lock();
  std::shared_ptr<GlContext> new_context_obj =
      new_context.context_object.lock();
  if (saved_context) {
    saved_context->context_object = old_context_obj;
    GetCurrentContextBinding(saved_context);
  }
  // Check that the context object is consistent with the native context.
  if (old_context_obj && saved_context) {
    ABSL_DCHECK(old_context_obj->context_ == saved_context->context);
  }
  if (new_context_obj) {
    ABSL_DCHECK(new_context_obj->context_ == new_context.context);
  }

  if (new_context_obj && (old_context_obj == new_context_obj)) {
    return absl::OkStatus();
  }

  if (old_context_obj) {
    // 1. Even if we cannot restore the new context, we want to get out of the
    // old one (we may be deliberately trying to exit it).
    // 2. We need to unset the old context before we unlock the old mutex,
    // Therefore, we first unset the old one before setting the new one.
    MP_RETURN_IF_ERROR(SetCurrentContextBinding({}));
    old_context_obj->context_use_mutex_.Unlock();
    CurrentContext().reset();
  }

  if (new_context_obj) {
    new_context_obj->context_use_mutex_.Lock();
    auto status = SetCurrentContextBinding(new_context);
    if (status.ok()) {
      CurrentContext() = new_context_obj;
    } else {
      new_context_obj->context_use_mutex_.Unlock();
    }
    return status;
  } else {
    return SetCurrentContextBinding(new_context);
  }
}

GlContext::ContextBinding GlContext::ThisContextBinding() {
  GlContext::ContextBinding result = ThisContextBindingPlatform();
  if (!destructing_) {
    result.context_object = shared_from_this();
  }
  return result;
}

absl::Status GlContext::EnterContext(ContextBinding* saved_context) {
  ABSL_DCHECK(HasContext());
  return SwitchContext(saved_context, ThisContextBinding());
}

absl::Status GlContext::ExitContext(const ContextBinding* saved_context) {
  ContextBinding no_context;
  if (!saved_context) {
    saved_context = &no_context;
  }
  return SwitchContext(nullptr, *saved_context);
}

std::shared_ptr<GlContext> GlContext::GetCurrent() {
  return CurrentContext().lock();
}

void GlContext::GlFinishCalled() {
  absl::MutexLock lock(&mutex_);
  ++gl_finish_count_;
  wait_for_gl_finish_cv_.SignalAll();
}

class GlFinishSyncPoint : public GlSyncPoint {
 public:
  explicit GlFinishSyncPoint(const std::shared_ptr<GlContext>& gl_context)
      : GlSyncPoint(gl_context),
        gl_finish_count_(gl_context_->gl_finish_count()) {}

  void Wait() override {
    gl_context_->WaitForGlFinishCountPast(gl_finish_count_);
  }

  bool IsReady() override {
    return gl_context_->gl_finish_count() > gl_finish_count_;
  }

 private:
  // Number of glFinish calls done before the creation of this token.
  int64_t gl_finish_count_ = -1;
};

// Just handles a GLsync. No context management.
class GlSyncWrapper {
 public:
  GlSyncWrapper() : sync_(nullptr) {}
  explicit GlSyncWrapper(GLsync sync) : sync_(sync) {}

  void Create() {
    Clear();
    sync_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    // Defer the flush for WebGL until the glClientWaitSync call as it's a
    // costly IPC call in Chrome's WebGL implementation.
#ifndef __EMSCRIPTEN__
    glFlush();
#endif
  }

  ~GlSyncWrapper() { Clear(); }

  GlSyncWrapper(const GlSyncWrapper&) = delete;
  GlSyncWrapper(GlSyncWrapper&& other) : sync_(nullptr) {
    *this = std::move(other);
  }
  GlSyncWrapper& operator=(const GlSyncWrapper&) = delete;
  GlSyncWrapper& operator=(GlSyncWrapper&& other) {
    using std::swap;
    swap(sync_, other.sync_);
    return *this;
  }
  GlSyncWrapper& operator=(std::nullptr_t) {
    Clear();
    return *this;
  }

  operator bool() const { return sync_ != nullptr; }
  bool operator==(std::nullptr_t) const { return sync_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return sync_ != nullptr; }

  void Wait() {
    if (!sync_) return;
    GLuint flags = 0;
    uint64_t timeout = std::numeric_limits<uint64_t>::max();
#ifdef __EMSCRIPTEN__
    // Setting GL_SYNC_FLUSH_COMMANDS_BIT ensures flush happens before we wait
    // on the fence. This is necessary since we defer the flush on WebGL.
    flags = GL_SYNC_FLUSH_COMMANDS_BIT;
    // WebGL only supports small implementation dependent timeout values. In
    // particular, Chrome only supports a timeout of 0.
    timeout = 0;
#endif
    GLenum result = glClientWaitSync(sync_, flags, timeout);
    if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED) {
      // TODO: we could clear at this point so later calls are faster,
      // but we need to do so in a thread-safe way.
      // Clear();
    }
    // TODO: do something if the wait fails?
  }

  // This method exists only for investigation purposes to distinguish stack
  // traces: external vs. internal context.
  // TODO: remove after glWaitSync crashes are resolved.
  void WaitOnGpuExternalContext() { glWaitSync(sync_, 0, GL_TIMEOUT_IGNORED); }

  void WaitOnGpu() {
    if (!sync_) return;
    // WebGL2 specifies a waitSync call, but since cross-context
    // synchronization is not supported, it's actually a no-op. Firefox prints
    // a warning when it's called, so let's just skip the call. See
    // b/184637485 for details.
#ifndef __EMSCRIPTEN__

    if (!GlContext::IsAnyContextCurrent()) {
      // glWaitSync must be called on with some context current. Doing the
      // opposite doesn't necessarily result in a crash or GL error. Hence,
      // just logging an error and skipping the call.
      ABSL_LOG_FIRST_N(ERROR, 1)
          << "An attempt to wait for a sync without any context current.";
      return;
    }

    auto context = GlContext::GetCurrent();
    if (context == nullptr) {
      // This can happen when WaitOnGpu is invoked on an external context,
      // created by other than GlContext::Create means.
      WaitOnGpuExternalContext();
      return;
    }

    // GlContext::ShouldUseFenceSync guards creation of sync objects, so this
    // CHECK should never fail if clients use MediaPipe APIs in an intended way.
    // TODO: remove after glWaitSync crashes are resolved.
    ABSL_CHECK(context->ShouldUseFenceSync()) << absl::StrFormat(
        "An attempt to wait for a sync when it should not be used. (OpenGL "
        "Version "
        "%d.%d)",
        context->gl_major_version(), context->gl_minor_version());

    glWaitSync(sync_, 0, GL_TIMEOUT_IGNORED);
#endif
  }

  bool IsReady() {
    if (!sync_) return true;
    GLuint flags = 0;
#ifdef __EMSCRIPTEN__
    // Setting GL_SYNC_FLUSH_COMMANDS_BIT ensures flush happens before we wait
    // on the fence. This is necessary since we defer the flush on WebGL.
    flags = GL_SYNC_FLUSH_COMMANDS_BIT;
#endif
    GLenum result = glClientWaitSync(sync_, flags, 0);
    if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED) {
      // TODO: we could clear at this point so later calls are faster,
      // but we need to do so in a thread-safe way.
      // Clear();
      return true;
    }
    return false;
  }

 private:
  void Clear() {
    if (sync_) {
      glDeleteSync(sync_);
      sync_ = nullptr;
    }
  }

  GLsync sync_;
};

class GlFenceSyncPoint : public GlSyncPoint {
 public:
  explicit GlFenceSyncPoint(const std::shared_ptr<GlContext>& gl_context)
      : GlSyncPoint(gl_context) {
    gl_context_->Run([this] { sync_.Create(); });
  }

  ~GlFenceSyncPoint() {
    if (sync_) {
      gl_context_->RunWithoutWaiting(
          [sync = new GlSyncWrapper(std::move(sync_))] { delete sync; });
    }
  }

  GlFenceSyncPoint(const GlFenceSyncPoint&) = delete;
  GlFenceSyncPoint& operator=(const GlFenceSyncPoint&) = delete;

  void Wait() override {
    if (!sync_) return;
    if (GlContext::IsAnyContextCurrent()) {
      sync_.Wait();
      return;
    }
    // In case a current GL context is not available, we fall back using the
    // captured gl_context_.
    gl_context_->Run([this] { sync_.Wait(); });
  }

  void WaitOnGpu() override {
    if (!sync_) return;
    // TODO: do not wait if we are already on the same context?
    sync_.WaitOnGpu();
  }

  bool IsReady() override {
    if (!sync_) return true;
    bool ready = false;
    // TODO: we should not block on the original context if possible.
    gl_context_->Run([this, &ready] { ready = sync_.IsReady(); });
    return ready;
  }

 private:
  GlSyncWrapper sync_;
};

class GlExternalFenceSyncPoint : public GlSyncPoint {
 public:
  // The provided GlContext is used as a fallback when a context is needed (e.g.
  // for deletion), but it's not the context the sync was created on, so we pass
  // nullptr to GlSyncPoint.
  explicit GlExternalFenceSyncPoint(
      const std::shared_ptr<GlContext>& graph_service_gl_context)
      : GlSyncPoint(nullptr),
        graph_service_gl_context_(graph_service_gl_context) {
    sync_.Create();
  }

  ~GlExternalFenceSyncPoint() {
    if (sync_) {
      graph_service_gl_context_->RunWithoutWaiting(
          [sync = new GlSyncWrapper(std::move(sync_))] { delete sync; });
    }
  }

  GlExternalFenceSyncPoint(const GlExternalFenceSyncPoint&) = delete;
  GlExternalFenceSyncPoint& operator=(const GlExternalFenceSyncPoint&) = delete;

  void Wait() override {
    // TODO: can we assume this is always called with a GLContext being current?
    sync_.Wait();
  }

  void WaitOnGpu() override { sync_.WaitOnGpu(); }

  bool IsReady() override {
    // TODO: can we assume this is always called with a GLContext being current?
    return sync_.IsReady();
  }

 private:
  GlSyncWrapper sync_;
  std::shared_ptr<GlContext> graph_service_gl_context_;
};

void GlMultiSyncPoint::Add(std::shared_ptr<GlSyncPoint> new_sync) {
  if (new_sync->GetContext() != nullptr) {
    for (auto& sync : syncs_) {
      if (sync->GetContext() == new_sync->GetContext()) {
        sync = std::move(new_sync);
        return;
      }
    }
  }
  syncs_.emplace_back(std::move(new_sync));
}

void GlMultiSyncPoint::Wait() {
  for (auto& sync : syncs_) {
    sync->Wait();
  }
  // At this point all the syncs have been reached, so clear them out.
  syncs_.clear();
}

void GlMultiSyncPoint::WaitOnGpu() {
  for (auto& sync : syncs_) {
    sync->WaitOnGpu();
  }
  // TODO: when do we clear out these syncs?
}

bool GlMultiSyncPoint::IsReady() {
  syncs_.erase(
      std::remove_if(syncs_.begin(), syncs_.end(),
                     std::bind(&GlSyncPoint::IsReady, std::placeholders::_1)),
      syncs_.end());
  return syncs_.empty();
}

// Set this to 1 to disable syncing. This can be used to verify that a test
// correctly detects sync issues.
#define MEDIAPIPE_DISABLE_GL_SYNC_FOR_DEBUG 0

#if MEDIAPIPE_DISABLE_GL_SYNC_FOR_DEBUG
class GlNopSyncPoint : public GlSyncPoint {
 public:
  explicit GlNopSyncPoint(const std::shared_ptr<GlContext>& gl_context)
      : GlSyncPoint(gl_context) {}

  void Wait() override {}

  bool IsReady() override { return true; }
};
#endif

bool GlContext::ShouldUseFenceSync() const {
  using internal_gl_context::OpenGlVersion;
#if defined(__EMSCRIPTEN__)
  // In Emscripten the glWaitSync function is non-null depending on linkopts,
  // but only works in a WebGL2 context.
  constexpr OpenGlVersion kMinVersionSyncAvaiable = {.major = 3, .minor = 0};
#elif defined(MEDIAPIPE_MOBILE)
  // OpenGL ES, glWaitSync is available since 3.0
  constexpr OpenGlVersion kMinVersionSyncAvaiable = {.major = 3, .minor = 0};
#else
  // TODO: specify major/minor version per remaining platforms.
  // By default, ignoring major/minor version requirement for backward
  // compatibility.
  constexpr OpenGlVersion kMinVersionSyncAvaiable = {.major = 0, .minor = 0};
#endif

  return SymbolAvailable(&glWaitSync) &&
         internal_gl_context::IsOpenGlVersionSameOrAbove(
             {.major = gl_major_version(), .minor = gl_minor_version()},
             kMinVersionSyncAvaiable);
}

std::shared_ptr<GlSyncPoint> GlContext::CreateSyncToken() {
  std::shared_ptr<GlSyncPoint> token;
#if MEDIAPIPE_DISABLE_GL_SYNC_FOR_DEBUG
  token.reset(new GlNopSyncPoint(shared_from_this()));
#else
  if (ShouldUseFenceSync()) {
    token.reset(new GlFenceSyncPoint(shared_from_this()));
  } else {
    token.reset(new GlFinishSyncPoint(shared_from_this()));
  }
#endif
  return token;
}

PlatformGlContext GlContext::GetCurrentNativeContext() {
  ContextBinding ctx;
  GetCurrentContextBinding(&ctx);
  return ctx.context;
}

bool GlContext::IsAnyContextCurrent() {
  return GetCurrentNativeContext() != kPlatformGlContextNone;
}

std::shared_ptr<GlSyncPoint>
GlContext::CreateSyncTokenForCurrentExternalContext(
    const std::shared_ptr<GlContext>& delegate_graph_context) {
  ABSL_CHECK(delegate_graph_context);
  if (!IsAnyContextCurrent()) return nullptr;
  if (delegate_graph_context->ShouldUseFenceSync()) {
    return std::shared_ptr<GlSyncPoint>(
        new GlExternalFenceSyncPoint(delegate_graph_context));
  } else {
    glFinish();
    return nullptr;
  }
}

std::shared_ptr<GlSyncPoint> GlContext::TestOnly_CreateSpecificSyncToken(
    SyncTokenTypeForTest type) {
  std::shared_ptr<GlSyncPoint> token;
  switch (type) {
    case SyncTokenTypeForTest::kGlFinish:
      token.reset(new GlFinishSyncPoint(shared_from_this()));
      return token;
  }
  return nullptr;
}

// Atomically set var to the greater of its current value or target.
template <typename T>
static void assign_larger_value(std::atomic<T>* var, T target) {
  T current = var->load();
  while (current < target && !var->compare_exchange_weak(current, target)) {
  }
}

// Note: this can get called from an arbitrary thread which is dealing with a
// GlFinishSyncPoint originating from this context.
void GlContext::WaitForGlFinishCountPast(int64_t count_to_pass) {
  if (gl_finish_count_ > count_to_pass) return;

  // If we've been asked to do a glFinish, note the count we need to reach and
  // signal the context our thread may currently be blocked on.
  {
    absl::MutexLock lock(&mutex_);
    assign_larger_value(&gl_finish_count_target_, count_to_pass + 1);
    wait_for_gl_finish_cv_.SignalAll();
    if (context_waiting_on_) {
      context_waiting_on_->wait_for_gl_finish_cv_.SignalAll();
    }
  }

  auto finish_task = [this, count_to_pass]() {
    // When a GlFinishSyncToken is created it takes the current finish count
    // from the GlContext, and we must wait for gl_finish_count_ to pass it.
    // Therefore, we need to do at most one more glFinish call. This DCHECK
    // is used for documentation and sanity-checking purposes.
    ABSL_DCHECK(gl_finish_count_ >= count_to_pass);
    if (gl_finish_count_ == count_to_pass) {
      glFinish();
      GlFinishCalled();
    }
  };

  if (IsCurrent()) {
    // If we are already on the current context, we cannot call
    // RunWithoutWaiting, since that task will not run until this function
    // returns. Instead, call it directly.
    finish_task();
    return;
  }

  std::shared_ptr<GlContext> other = GetCurrent();
  if (other) {
    // If another context is current, make a note that it is blocked on us, so
    // it can signal the right condition variable if it is asked to do a
    // glFinish.
    absl::MutexLock other_lock(&other->mutex_);
    ABSL_DCHECK(!other->context_waiting_on_);
    other->context_waiting_on_ = this;
  }
  // We do not schedule this action using Run because we don't necessarily
  // want to wait for it to complete. If another job calls GlFinishCalled
  // sooner, we are done.
  RunWithoutWaiting(std::move(finish_task));
  {
    absl::MutexLock lock(&mutex_);
    while (gl_finish_count_ <= count_to_pass) {
      if (other && other->gl_finish_count_ < other->gl_finish_count_target_) {
        // If another context's dedicated thread is current, it is blocked
        // waiting for this context to issue a glFinish call. But this context
        // may also block waiting for the other context to do the same: this can
        // happen when two contexts are handling each other's GlFinishSyncPoints
        // (e.g. a producer and a consumer). To avoid a deadlock a context that
        // is waiting on another context must still service Wait calls it may
        // receive from its own GlFinishSyncPoints.
        //
        // We unlock this context's mutex to avoid holding both at the same
        // time.
        mutex_.Unlock();
        {
          glFinish();
          other->GlFinishCalled();
        }
        mutex_.Lock();
        // Because we temporarily unlocked mutex_, we cannot wait on the
        // condition variable wait away; we need to go back to re-checking the
        // condition. Otherwise we might miss a signal.
        continue;
      }
      wait_for_gl_finish_cv_.Wait(&mutex_);
    }
  }

  if (other) {
    // The other context is no longer waiting on us.
    absl::MutexLock other_lock(&other->mutex_);
    other->context_waiting_on_ = nullptr;
  }
}

void GlContext::WaitSyncToken(const std::shared_ptr<GlSyncPoint>& token) {
  ABSL_CHECK(token);
  token->Wait();
}

bool GlContext::SyncTokenIsReady(const std::shared_ptr<GlSyncPoint>& token) {
  ABSL_CHECK(token);
  return token->IsReady();
}

void GlContext::ForceClearExistingGlErrors() {
  LogUncheckedGlErrors(CheckForGlErrors(/*force=*/true));
}

bool GlContext::CheckForGlErrors() { return CheckForGlErrors(false); }

bool GlContext::CheckForGlErrors(bool force) {
#if UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING
  if (!force) {
    ABSL_LOG_FIRST_N(WARNING, 1) << "OpenGL error checking is disabled";
    return false;
  }
#endif

  if (!HasContext()) return false;
  GLenum error;
  bool had_error = false;
  while ((error = glGetError()) != GL_NO_ERROR) {
    had_error = true;
    switch (error) {
      case GL_INVALID_ENUM:
        ABSL_LOG(INFO) << "Found unchecked GL error: GL_INVALID_ENUM";
        break;
      case GL_INVALID_VALUE:
        ABSL_LOG(INFO) << "Found unchecked GL error: GL_INVALID_VALUE";
        break;
      case GL_INVALID_OPERATION:
        ABSL_LOG(INFO) << "Found unchecked GL error: GL_INVALID_OPERATION";
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        ABSL_LOG(INFO)
            << "Found unchecked GL error: GL_INVALID_FRAMEBUFFER_OPERATION";
        break;
      case GL_OUT_OF_MEMORY:
        ABSL_LOG(INFO) << "Found unchecked GL error: GL_OUT_OF_MEMORY";
        break;
      default:
        ABSL_LOG(INFO) << "Found unchecked GL error: UNKNOWN ERROR";
        break;
    }
  }
  return had_error;
}

void GlContext::LogUncheckedGlErrors(bool had_gl_errors) {
  if (had_gl_errors) {
    // TODO: ideally we would print a backtrace here, or at least
    // the name of the current calculator, to make it easier to find the
    // culprit. In practice, getting a backtrace from Android without crashing
    // is nearly impossible, so screw it. Just change this to ABSL_LOG(FATAL)
    // when you want to debug.
    ABSL_LOG(WARNING) << "Ignoring unchecked GL error.";
  }
}

const GlTextureInfo& GlTextureInfoForGpuBufferFormat(GpuBufferFormat format,
                                                     int plane) {
  std::shared_ptr<GlContext> ctx = GlContext::GetCurrent();
  ABSL_CHECK(ctx != nullptr);
  return GlTextureInfoForGpuBufferFormat(format, plane, ctx->GetGlVersion());
}

void GlContext::SetStandardTextureParams(GLenum target, GLint internal_format) {
  // Default to using linear filter everywhere. For float32 textures, fall back
  // to GL_NEAREST if linear filtering unsupported.
  GLint filter;
  switch (internal_format) {
    case GL_R32F:
    case GL_RG32F:
    case GL_RGBA32F:
      // 32F (unlike 16f) textures do not always support texture filtering
      // (According to OpenGL ES specification [TEXTURE IMAGE SPECIFICATION])
      filter = can_linear_filter_float_textures_ ? GL_LINEAR : GL_NEAREST;
      break;
    default:
      filter = GL_LINEAR;
  }
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, filter);
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, filter);
  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

const GlContext::Attachment<GLuint> kUtilityFramebuffer(
    [](GlContext&) -> GlContext::Attachment<GLuint>::Ptr {
      GLuint framebuffer;
      glGenFramebuffers(1, &framebuffer);
      if (!framebuffer) return nullptr;
      return {new GLuint(framebuffer), [](void* ptr) {
                GLuint* fb = static_cast<GLuint*>(ptr);
                glDeleteFramebuffers(1, fb);
                delete fb;
              }};
    });

}  // namespace mediapipe
