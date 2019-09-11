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
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/gpu/gl_context_internal.h"

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
    LOG_FIRST_N(INFO, 1) << "Can't set pthread names: name: \"" << name
                         << "\"; error: " << res;
  }
#elif __APPLE__
  pthread_setname_np(name);
#endif
  ANNOTATE_THREAD_NAME(name);
}

GlContext::DedicatedThread::DedicatedThread() {
  CHECK_EQ(pthread_create(&gl_thread_id_, nullptr, ThreadBody, this), 0);
}

GlContext::DedicatedThread::~DedicatedThread() {
  if (IsCurrentThread()) {
    CHECK(self_destruct_);
    CHECK_EQ(pthread_detach(gl_thread_id_), 0);
  } else {
    // Give an invalid job to signal termination.
    PutJob({});
    CHECK_EQ(pthread_join(gl_thread_id_, nullptr), 0);
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

::mediapipe::Status GlContext::DedicatedThread::Run(GlStatusFunction gl_func) {
  // Neither ENDO_SCOPE nor ENDO_TASK seem to work here.
  if (IsCurrentThread()) {
    return gl_func();
  }
  bool done = false;  // Guarded by mutex_ after initialization.
  ::mediapipe::Status status;
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
  CHECK(gl_func);
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

::mediapipe::Status GlContext::FinishInitialization(bool create_thread) {
  if (create_thread) {
    thread_ = absl::make_unique<GlContext::DedicatedThread>();
    MP_RETURN_IF_ERROR(thread_->Run([this] { return EnterContext(nullptr); }));
  }

  return Run([this]() -> ::mediapipe::Status {
    absl::string_view version_string(
        reinterpret_cast<const char*>(glGetString(GL_VERSION)));

    // Let's try getting the numeric version if possible.
    glGetIntegerv(GL_MAJOR_VERSION, &gl_major_version_);
    GLenum err = glGetError();
    if (err == GL_NO_ERROR) {
      glGetIntegerv(GL_MINOR_VERSION, &gl_minor_version_);
    } else {
      // GL_MAJOR_VERSION is not supported on GL versions below 3. We have to
      // parse the version std::string.
      if (!ParseGlVersion(version_string, &gl_major_version_,
                          &gl_minor_version_)) {
        LOG(WARNING) << "invalid GL_VERSION format: '" << version_string
                     << "'; assuming 2.0";
        gl_major_version_ = 2;
        gl_minor_version_ = 0;
      }
    }

    LOG(INFO) << "GL version: " << gl_major_version_ << "." << gl_minor_version_
              << " (" << glGetString(GL_VERSION) << ")";

    return ::mediapipe::OkStatus();
  });
}

GlContext::GlContext() {}

GlContext::~GlContext() {
  // Note: on Apple platforms, this object contains Objective-C objects.
  // The destructor will release them, but ARC must be on.
#ifdef __OBJC__
#if !__has_feature(objc_arc)
#error This file must be built with ARC.
#endif
#endif  // __OBJC__
  if (thread_) {
    auto status = thread_->Run([this] {
      if (profiling_helper_) {
        profiling_helper_->LogAllTimestamps();
      }
      return ExitContext(nullptr);
    });
    if (!status.ok()) {
      LOG(ERROR) << "Failed to deactivate context on thread: " << status;
    }
    if (thread_->IsCurrentThread()) {
      thread_.release()->SelfDestruct();
    }
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

::mediapipe::Status GlContext::Run(GlStatusFunction gl_func, int node_id,
                                   Timestamp input_timestamp) {
  ::mediapipe::Status status;
  if (thread_) {
    bool had_gl_errors = false;
    status = thread_->Run(
        [this, gl_func, node_id, &input_timestamp, &had_gl_errors] {
          if (profiling_helper_) {
            profiling_helper_->MarkTimestamp(node_id, input_timestamp,
                                             /*is_finish=*/false);
          }
          auto status = gl_func();
          if (profiling_helper_) {
            profiling_helper_->MarkTimestamp(node_id, input_timestamp,
                                             /*is_finish=*/true);
          }
          had_gl_errors = CheckForGlErrors();
          return status;
        });
    LogUncheckedGlErrors(had_gl_errors);
  } else {
    ContextBinding saved_context;
    MP_RETURN_IF_ERROR(EnterContext(&saved_context));
    if (profiling_helper_) {
      profiling_helper_->MarkTimestamp(node_id, input_timestamp,
                                       /*is_finish=*/false);
    }
    status = gl_func();
    if (profiling_helper_) {
      profiling_helper_->MarkTimestamp(node_id, input_timestamp,
                                       /*is_finish=*/true);
    }
    LogUncheckedGlErrors(CheckForGlErrors());
    MP_RETURN_IF_ERROR(ExitContext(&saved_context));
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
    ContextBinding saved_context;
    auto status = EnterContext(&saved_context);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to enter context: " << status;
      return;
    }
    gl_func();
    LogUncheckedGlErrors(CheckForGlErrors());
    status = ExitContext(&saved_context);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to exit context: " << status;
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

::mediapipe::Status GlContext::SwitchContext(ContextBinding* saved_context,
                                             const ContextBinding& new_context)
    NO_THREAD_SAFETY_ANALYSIS {
  std::shared_ptr<GlContext> old_context_obj = CurrentContext().lock();
  std::shared_ptr<GlContext> new_context_obj =
      new_context.context_object.lock();
  if (saved_context) {
    saved_context->context_object = old_context_obj;
    GetCurrentContextBinding(saved_context);
  }
  // Check that the context object is consistent with the native context.
  if (old_context_obj && saved_context) {
    DCHECK(old_context_obj->context_ == saved_context->context);
  }
  if (new_context_obj) {
    DCHECK(new_context_obj->context_ == new_context.context);
  }

  if (new_context_obj && (old_context_obj == new_context_obj)) {
    return ::mediapipe::OkStatus();
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

::mediapipe::Status GlContext::EnterContext(ContextBinding* saved_context) {
  DCHECK(HasContext());
  return SwitchContext(saved_context, ThisContextBinding());
}

::mediapipe::Status GlContext::ExitContext(
    const ContextBinding* saved_context) {
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

class GlFenceSyncPoint : public GlSyncPoint {
 public:
  explicit GlFenceSyncPoint(const std::shared_ptr<GlContext>& gl_context)
      : GlSyncPoint(gl_context) {
    gl_context_->Run([this] {
      sync_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
      glFlush();
    });
  }

  ~GlFenceSyncPoint() {
    if (sync_) {
      GLsync sync = sync_;
      gl_context_->RunWithoutWaiting([sync] { glDeleteSync(sync); });
    }
  }

  GlFenceSyncPoint(const GlFenceSyncPoint&) = delete;
  GlFenceSyncPoint& operator=(const GlFenceSyncPoint&) = delete;

  void Wait() override {
    if (!sync_) return;
    gl_context_->Run([this] {
      GLenum result =
          glClientWaitSync(sync_, 0, std::numeric_limits<uint64_t>::max());
      if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED) {
        glDeleteSync(sync_);
        sync_ = nullptr;
      }
      // TODO: do something if the wait fails?
    });
  }

  void WaitOnGpu() override {
    if (!sync_) return;
    // TODO: do not wait if we are already on the same context?
    glWaitSync(sync_, 0, GL_TIMEOUT_IGNORED);
  }

  bool IsReady() override {
    if (!sync_) return true;
    bool ready = false;
    // TODO: we should not block on the original context if possible.
    gl_context_->Run([this, &ready] {
      GLenum result = glClientWaitSync(sync_, 0, 0);
      if (result == GL_ALREADY_SIGNALED || result == GL_CONDITION_SATISFIED) {
        glDeleteSync(sync_);
        sync_ = nullptr;
        ready = true;
      }
    });
    return ready;
  }

 private:
  GLsync sync_;
};

void GlMultiSyncPoint::Add(std::shared_ptr<GlSyncPoint> new_sync) {
  for (auto& sync : syncs_) {
    if (&sync->GetContext() == &new_sync->GetContext()) {
      sync = std::move(new_sync);
      return;
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

std::shared_ptr<GlSyncPoint> GlContext::CreateSyncToken() {
  std::shared_ptr<GlSyncPoint> token;
#if MEDIAPIPE_DISABLE_GL_SYNC_FOR_DEBUG
  token.reset(new GlNopSyncPoint(shared_from_this()));
#else
  if (SymbolAvailable(&glWaitSync)) {
    token.reset(new GlFenceSyncPoint(shared_from_this()));
  } else {
    token.reset(new GlFinishSyncPoint(shared_from_this()));
  }
#endif
  return token;
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

void GlContext::WaitForGlFinishCountPast(int64_t count_to_pass) {
  if (gl_finish_count_ > count_to_pass) return;
  auto finish_task = [this, count_to_pass]() {
    // When a GlFinishSyncToken is created it takes the current finish count
    // from the GlContext, and we must wait for gl_finish_count_ to pass it.
    // Therefore, we need to do at most one more glFinish call. This DCHECK
    // is used for documentation and sanity-checking purposes.
    DCHECK(gl_finish_count_ >= count_to_pass);
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
  // We do not schedule this action using Run because we don't necessarily
  // want to wait for it to complete. If another job calls GlFinishCalled
  // sooner, we are done.
  RunWithoutWaiting(std::move(finish_task));
  absl::MutexLock lock(&mutex_);
  while (gl_finish_count_ <= count_to_pass) {
    wait_for_gl_finish_cv_.Wait(&mutex_);
  }
}

void GlContext::WaitSyncToken(const std::shared_ptr<GlSyncPoint>& token) {
  CHECK(token);
  token->Wait();
}

bool GlContext::SyncTokenIsReady(const std::shared_ptr<GlSyncPoint>& token) {
  CHECK(token);
  return token->IsReady();
}

bool GlContext::CheckForGlErrors() {
#if UNSAFE_EMSCRIPTEN_SKIP_GL_ERROR_HANDLING
  LOG_FIRST_N(WARNING, 1) << "MediaPipe OpenGL error checking is disabled";
  return false;
#endif

  if (!HasContext()) return false;
  GLenum error;
  bool had_error = false;
  while ((error = glGetError()) != GL_NO_ERROR) {
    had_error = true;
    switch (error) {
      case GL_INVALID_ENUM:
        LOG(INFO) << "Found unchecked GL error: GL_INVALID_ENUM";
        break;
      case GL_INVALID_VALUE:
        LOG(INFO) << "Found unchecked GL error: GL_INVALID_VALUE";
        break;
      case GL_INVALID_OPERATION:
        LOG(INFO) << "Found unchecked GL error: GL_INVALID_OPERATION";
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        LOG(INFO)
            << "Found unchecked GL error: GL_INVALID_FRAMEBUFFER_OPERATION";
        break;
      case GL_OUT_OF_MEMORY:
        LOG(INFO) << "Found unchecked GL error: GL_OUT_OF_MEMORY";
        break;
      default:
        LOG(INFO) << "Found unchecked GL error: UNKNOWN ERROR";
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
    // is nearly impossible, so screw it. Just change this to LOG(FATAL) when
    // you want to debug.
    LOG(WARNING) << "Ignoring unchecked GL error.";
  }
}

}  // namespace mediapipe
