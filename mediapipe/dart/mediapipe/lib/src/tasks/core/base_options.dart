import 'dart:io' as io;
import 'dart:typed_data';
import 'package:path/path.dart' as path;
import 'package:protobuf/protobuf.dart' as $pb;

import '../../../generated/mediapipe/calculators/calculators.dart';
import '../../../generated/mediapipe/tasks/tasks.dart' as tasks_pb;

/// Class to extend in task-specific *Options classes. Funnels the three
/// [BaseOptions] attributes into their own object.
abstract class TaskOptions {
  TaskOptions({this.modelAssetBuffer, this.modelAssetPath, this.delegate})
      : baseOptions = BaseOptions(
          delegate: delegate,
          modelAssetBuffer: modelAssetBuffer,
          modelAssetPath: modelAssetPath,
        );

  final Uint8List? modelAssetBuffer;
  final String? modelAssetPath;
  final Delegate? delegate;

  final BaseOptions baseOptions;

  $pb.GeneratedMessage toProto();

  /// In proto2 syntax, extensions are unique IDs, suitable for keys in a hash
  /// map, which power the Extensions pattern for protos to house arbitrary
  /// extended data.
  ///
  /// In proto3, this pattern is replaced with the [Any] protobuf, as the
  /// convention for setting the unique identifiers surpassed the maximum upper
  /// bound of 29 bits as allocated in the protobuf spec.
  $pb.Extension get ext;
}

final class BaseOptions {
  BaseOptions({this.modelAssetBuffer, this.modelAssetPath, this.delegate});

  /// The model asset file contents as bytes;
  Uint8List? modelAssetBuffer;

  /// Path to the model asset file.
  String? modelAssetPath;

  /// Acceleration strategy to use. GPU support is currently limited to
  /// Ubuntu platform.
  Delegate? delegate;

  /// See also: https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/tasks/python/core/base_options.py;l=63-89;rcl=548857458
  tasks_pb.BaseOptions toProto() {
    String? absModelPath =
        modelAssetPath != null ? path.absolute(modelAssetPath!) : null;

    if (!io.Platform.isLinux && delegate == Delegate.gpu) {
      throw Exception(
        'GPU Delegate is not yet supported for ${io.Platform.operatingSystem}',
      );
    }
    tasks_pb.Acceleration? acceleration;
    if (delegate == Delegate.cpu) {
      acceleration = tasks_pb.Acceleration.create()
        ..tflite = InferenceCalculatorOptions_Delegate_TfLite.create();
    }

    final modelAsset = tasks_pb.ExternalFile.create();
    if (absModelPath != null) {
      modelAsset.fileName = absModelPath;
    }
    if (modelAssetBuffer != null) {
      modelAsset.fileContent = modelAssetBuffer!;
    }

    final options = tasks_pb.BaseOptions.create()..modelAsset = modelAsset;
    if (acceleration != null) {
      options.acceleration = acceleration;
    }
    return options;
  }
}

/// Hardware location to perform the given task.
enum Delegate { cpu, gpu }
