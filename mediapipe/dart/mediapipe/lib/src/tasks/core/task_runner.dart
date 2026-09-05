import 'dart:ffi' as ffi;
// TODO: This will require a web-specific solution.
import 'dart:io';
import 'package:path/path.dart' as path;

import '../../../generated/mediapipe/framework/calculator.pb.dart';

// TODO: Figure out ffi type for Maps
typedef ProcessCC = Map<String, Packet> Function(Map<String, Object> data);
typedef Process = Map<String, Packet> Function(Map<String, Object> data);

// TODO: Wrap C++ TaskRunner with this, similarly to this Python wrapper:
// https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/python/framework_bindings.cc?q=python%20framework_bindings.cc
class TaskRunner {
  TaskRunner(this.graphConfig) {
    var libraryPath =
        path.join(Directory.current.absolute.path, 'cc', 'main.dylib');
    mediaPipe = ffi.DynamicLibrary.open(libraryPath);
  }

  final CalculatorGraphConfig graphConfig;

  late ffi.DynamicLibrary mediaPipe;

  // TODO: Actually decode this line for correct parameter type:
  // https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/tasks/python/text/text_classifier.py;l=181
  Map<String, Packet> process(Map<String, Object> data) {
    throw UnimplementedError();
    // final Process ccProcess =
    //     mediaPipe.lookup<ffi.NativeFunction<ProcessCC>>('process').asFunction();
    // return ccProcess(data);
  }
}

// TODO: Wrap C++ Packet with this, similarly to this Python wrapper:
// https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/python/pybind/packet.h
class Packet {}
