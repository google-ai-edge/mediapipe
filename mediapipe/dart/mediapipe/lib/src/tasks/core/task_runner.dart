import '../../../generated/mediapipe/framework/calculator.pb.dart';

// TODO: Wrap C++ TaskRunner with this, similarly to this Python wrapper:
// https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/python/framework_bindings.cc?q=python%20framework_bindings.cc
class TaskRunner {
  TaskRunner(this.graphConfig);

  final CalculatorGraphConfig graphConfig;

  // TODO: Actually decode this line for correct parameter type:
  // https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/tasks/python/text/text_classifier.py;l=181
  Map<String, Packet> process(Map<String, Object> data) => {};
}

// TODO: Wrap C++ Packet with this, similarly to this Python wrapper:
// https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/python/pybind/packet.h
class Packet {}
