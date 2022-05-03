#include "mediapipe/framework/graph_service_manager.h"

#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {
const GraphService<int> kIntService("mediapipe::IntService");
}  // namespace

TEST(GraphServiceManager, SetGetServiceObject) {
  GraphServiceManager service_manager;

  EXPECT_EQ(service_manager.GetServiceObject(kIntService), nullptr);

  MP_EXPECT_OK(service_manager.SetServiceObject(kIntService,
                                                std::make_shared<int>(100)));
  ASSERT_NE(service_manager.GetServiceObject(kIntService), nullptr);
  EXPECT_EQ(*service_manager.GetServiceObject(kIntService), 100);
}

TEST(GraphServiceManager, SetServicePacket) {
  GraphServiceManager service_manager;

  MP_EXPECT_OK(service_manager.SetServicePacket(
      kIntService,
      mediapipe::MakePacket<std::shared_ptr<int>>(std::make_shared<int>(100))));
  ASSERT_NE(service_manager.GetServiceObject(kIntService), nullptr);
  EXPECT_EQ(*service_manager.GetServiceObject(kIntService), 100);
}

TEST(GraphServiceManager, ServicePackets) {
  GraphServiceManager service_manager;

  EXPECT_TRUE(service_manager.ServicePackets().empty());

  MP_EXPECT_OK(service_manager.SetServiceObject(kIntService,
                                                std::make_shared<int>(100)));

  EXPECT_EQ(service_manager.ServicePackets().size(), 1);
  ASSERT_NE(service_manager.ServicePackets().find(kIntService.key),
            service_manager.ServicePackets().end());
  EXPECT_EQ(*service_manager.ServicePackets()
                 .at(kIntService.key)
                 .Get<std::shared_ptr<int>>(),
            100);
}

}  // namespace mediapipe
