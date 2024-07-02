#ifndef MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_
#define MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/packet.h"

namespace mediapipe {

class GraphServiceManager {
 public:
  using ServiceMap = std::map<std::string, Packet>;

  template <typename T>
  absl::Status SetServiceObject(const GraphService<T>& service,
                                std::shared_ptr<T> object) {
    return SetServicePacket(service,
                            MakePacket<std::shared_ptr<T>>(std::move(object)));
  }

  absl::Status SetServicePacket(const GraphServiceBase& service, Packet p);

  template <typename T>
  std::shared_ptr<T> GetServiceObject(const GraphService<T>& service) const {
    Packet p = GetServicePacket(service);
    if (p.IsEmpty()) return nullptr;
    return p.Get<std::shared_ptr<T>>();
  }
  const ServiceMap& ServicePackets() { return service_packets_; }

 private:
  Packet GetServicePacket(const GraphServiceBase& service) const;
  // Mutex protection since the GraphServiceManager instance can be shared among
  // multiple (nested) MP graphs.
  mutable absl::Mutex service_packets_mutex_;
  ServiceMap service_packets_ ABSL_GUARDED_BY(service_packets_mutex_);
  friend class CalculatorGraph;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_
