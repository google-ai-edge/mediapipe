#include "mediapipe/util/log_fatal_to_breakpad.h"

#import <Foundation/Foundation.h>

#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#import "googlemac/iPhone/Shared/GoogleIOSBreakpad/Classes/GoogleBreakpadController.h"

namespace mediapipe {
namespace {
NSString* MakeNSString(absl::string_view str) {
  return [[NSString alloc] initWithBytes:str.data()
                                  length:str.length()
                                encoding:NSUTF8StringEncoding];
}
}  // namespace

static NSString* const kFatalLogMessageKey = @"fatal_log_message";

class BreakpadFatalLogSink : public absl::LogSink {
 public:
  BreakpadFatalLogSink()
      : breakpad_controller_([GoogleBreakpadController sharedInstance]) {}
  void Send(const absl::LogEntry& entry) override {
    if (entry.log_severity() != absl::LogSeverity::kFatal) return;
    __block NSString* message = MakeNSString(entry.text_message_with_prefix());
    [breakpad_controller_ withBreakpadRef:^(BreakpadRef breakpad) {
      // NOTE: This block runs on Breakpad's background queue.
      if (!breakpad) return;
      BreakpadAddUploadParameter(breakpad, kFatalLogMessageKey, message);
    }];
  }

 private:
  GoogleBreakpadController* breakpad_controller_;
};

absl::LogSink* GetBreakpadFatalLogSink() {
  static BreakpadFatalLogSink sink;
  return &sink;
}

// This log sink is automatically enabled when including this library.
static const auto kRegisterLogSink = [] {
  absl::AddLogSink(GetBreakpadFatalLogSink());
  return true;
}();

}  // namespace mediapipe
