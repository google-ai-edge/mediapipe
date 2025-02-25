#ifndef MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_
#define MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "mediapipe/framework/graph_service.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

class GraphServiceManager {
 public:
  using ServiceMap = std::map<std::string, Packet>;

  GraphServiceManager() = default;

  template <typename T>
  absl::Status SetServiceObject(const GraphService<T>& service,
                                std::shared_ptr<T> object) {
    RET_CHECK(object != nullptr) << "SetServiceObject called for "
                                 << service.key << " with empty shared_ptr.";
    return SetServicePacket(service,
                            MakePacket<std::shared_ptr<T>>(std::move(object)));
  }

  absl::Status SetServicePacket(const GraphServiceBase& service, Packet p);

  void SetServicePackets(const ServiceMap& service_packets) {
    service_packets_ = service_packets;
  }

  template <typename T>
  std::shared_ptr<T> GetServiceObject(const GraphService<T>& service) const {
    Packet p = GetServicePacket(service);
    if (p.IsEmpty()) return nullptr;
    auto ptr = p.Get<std::shared_ptr<T>>();
    ABSL_CHECK(ptr != nullptr)
        << "GraphService " << service.key << " is not set (empty shared_ptr).";

    return ptr;
  }
  const ServiceMap& ServicePackets() const { return service_packets_; }

 private:
  Packet GetServicePacket(const GraphServiceBase& service) const;

  ServiceMap service_packets_;
  friend class CalculatorGraph;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_GRAPH_SERVICE_MANAGER_H_
