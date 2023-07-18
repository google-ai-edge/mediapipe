// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

// clang-format off
//#include "utils.hpp"
// clang-format on

typedef std::function<void(size_t id, const std::exception_ptr& ptr)>
    QueueCallbackFunction;

/// @brief Wrapper class for InferenceEngine::InferRequest. Handles asynchronous callbacks and calculates execution
/// time.
class InferReqWrap final {
public:
    using Ptr = std::shared_ptr<InferReqWrap>;

    ~InferReqWrap() = default;

    explicit InferReqWrap(ov::CompiledModel &model, size_t id, QueueCallbackFunction callbackQueue)
            : _request(model.create_infer_request()),
              _id(id),
              _callbackQueue(callbackQueue) {
      _request.set_callback([&](const std::exception_ptr &ptr) {
          _callbackQueue(_id, ptr);
      });
    }

    void start_async() {
      _request.start_async();
    }

    void wait() {
      _request.wait();
    }

    void infer() {
      _request.infer();
      _callbackQueue(_id, nullptr);
    }

    void set_shape(const std::string& name, const ov::Shape& dims) {
      // TODO check return status
      _request.get_tensor(name).set_shape(dims);
    }

    ov::Tensor get_tensor(const std::string& name) {
      return _request.get_tensor(name);
    }

    ov::Tensor get_output_tensor(size_t i) {
      return _request.get_output_tensor(i);
    }

    void set_tensor(const std::string& name, const ov::Tensor& data) {
      _request.set_tensor(name, data);
    }

    void set_input_tensor(size_t i, const ov::Tensor& data) {
      _request.set_input_tensor(i, data);
    }

//    // in case of using GPU memory we need to allocate CL buffer for
//    // output blobs. By encapsulating cl buffer inside InferReqWrap
//    // we will control the number of output buffers and access to it.
//    std::map<std::string, ov::gpu::BufferType>& get_output_cl_buffer() {
//        return outputClBuffer;
//    }

private:
    ov::InferRequest _request;
    size_t _id;
    QueueCallbackFunction _callbackQueue;
//    std::map<std::string, ::gpu::BufferType> outputClBuffer;
};

class InferRequestsQueue final {
public:
    InferRequestsQueue(ov::CompiledModel& model, size_t nireq)
    {
      for (size_t id = 0; id < nireq; id++) {
        requests.push_back(std::make_shared<InferReqWrap>(model,
                                                          id,
                                                          std::bind(&InferRequestsQueue::put_idle_request,
                                                                    this,
                                                                    std::placeholders::_1,
                                                                    std::placeholders::_2)));
        _idleIds.push(id);
      }
    }

    ~InferRequestsQueue() {
      // Inference Request guarantee that it will wait for all asynchronous internal tasks in destructor
      // So it should be released before any context that the request can use inside internal asynchronous tasks
      // For example all members of InferRequestsQueue would be destroyed before `requests` vector
      // So requests can try to use this members from `putIdleRequest()` that would be called from request callback
      // To avoid this we should move this vector declaration after all members declaration or just clear it manually
      // in destructor
      requests.clear();
    }

    void put_idle_request(size_t id,
                          const std::exception_ptr& ptr = nullptr) {
      std::unique_lock<std::mutex> lock(_mutex);
      if (ptr) {
        inferenceException = ptr;
      } else {
        _idleIds.push(id);
      }
      _cv.notify_one();
    }

    InferReqWrap::Ptr get_idle_request() {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [this] {
          if (inferenceException) {
            try {
              std::rethrow_exception(inferenceException);
            } catch (const std::exception &ex) {
              throw ex;
            }
          }
          return _idleIds.size() > 0;
      });
      auto request = requests.at(_idleIds.front());
      _idleIds.pop();
      return request;
    }

    void wait_all() {
      std::unique_lock<std::mutex> lock(_mutex);
      _cv.wait(lock, [this] {
          if (inferenceException) {
            try {
              std::rethrow_exception(inferenceException);
            } catch (const std::exception& ex) {
              throw ex;
            }
          }
          return _idleIds.size() == requests.size();
      });
    }

    std::vector<InferReqWrap::Ptr> requests;

private:
    std::queue<size_t> _idleIds;
    std::mutex _mutex;
    std::condition_variable _cv;
    std::exception_ptr inferenceException = nullptr;
};
