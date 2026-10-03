import 'package:protobuf/protobuf.dart' as $pb;
import '../../../generated/mediapipe/tasks/tasks.dart' as tasks_pb;
import '../tasks.dart';

class TextClassifier {
  /// Primary constructor for [TextClassifier].
  TextClassifier(this.options)
      : _taskInfo = TaskInfo(
          taskGraph: taskGraphName,
          inputStreams: <String>['$textTag:$textInStreamName'],
          outputStreams: <String>[
            '$classificationsTag:$classificationsStreamName'
          ],
          taskOptions: options,
        ) {
    _taskRunner = TaskRunner(_taskInfo.generateGraphConfig());
  }

  /// Shortcut constructor which only accepts a local path to the model.
  factory TextClassifier.fromAssetPath(String assetPath) => TextClassifier(
        TextClassifierOptions(modelAssetPath: assetPath),
      );

  /// Configuration options for this [TextClassifier].
  final TextClassifierOptions options;

  /// Configuration object passed to the [TaskRunner].
  final TaskInfo _taskInfo;

  TaskRunner get taskRunner => _taskRunner!;
  TaskRunner? _taskRunner;

  static const classificationsStreamName = 'classifications_out';
  static const classificationsTag = 'CLASSIFICATIONS';
  static const textTag = 'TEXT';
  static const textInStreamName = 'text_in';
  static const taskGraphName =
      'mediapipe.tasks.text.text_classifier.TextClassifierGraph';

  // TODO: Don't return protobuf objects. Instead, convert to plain Dart objects.
  /// Performs classification on the input `text`.
  Future<tasks_pb.ClassificationResult> classify(String text) async {
    // TODO: Actually decode this line to correctly fill up this map parameter
    // https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/tasks/python/text/text_classifier.py;l=181
    final outputPackets = taskRunner.process({textInStreamName: Object()});

    // TODO: Obviously this is not real
    return tasks_pb.ClassificationResult.create();
  }
}

class TextClassifierOptions extends TaskOptions {
  TextClassifierOptions({
    this.displayNamesLocale,
    this.maxResults,
    this.scoreThreshold,
    this.categoryAllowlist,
    this.categoryDenylist,
    super.modelAssetBuffer,
    super.modelAssetPath,
    super.delegate,
  });

  /// The locale to use for display names specified through the TFLite Model
  /// Metadata.
  String? displayNamesLocale;

  /// The maximum number of top-scored classification results to return.
  int? maxResults;

  /// Overrides the ones provided in the model metadata. Results below this
  /// value are rejected.
  double? scoreThreshold;

  /// Allowlist of category names. If non-empty, classification results whose
  /// category name is not in this set will be discarded. Duplicate or unknown
  /// category names are ignored. Mutually exclusive with `categoryDenylist`.
  List<String>? categoryAllowlist;

  /// Denylist of category names. If non-empty, classification results whose
  /// category name is in this set will be discarded. Duplicate or unknown
  /// category names are ignored. Mutually exclusive with `categoryAllowList`.
  List<String>? categoryDenylist;

  @override
  $pb.Extension get ext => tasks_pb.TextClassifierGraphOptions.ext;

  @override
  tasks_pb.TextClassifierGraphOptions toProto() {
    final classifierOptions = tasks_pb.ClassifierOptions.create();
    if (displayNamesLocale != null) {
      classifierOptions.displayNamesLocale = displayNamesLocale!;
    }
    if (maxResults != null) {
      classifierOptions.maxResults = maxResults!;
    }
    if (scoreThreshold != null) {
      classifierOptions.scoreThreshold = scoreThreshold!;
    }
    if (categoryAllowlist != null) {
      classifierOptions.categoryAllowlist.addAll(categoryAllowlist!);
    }
    if (categoryDenylist != null) {
      classifierOptions.categoryDenylist.addAll(categoryDenylist!);
    }

    return tasks_pb.TextClassifierGraphOptions.create()
      ..baseOptions = baseOptions.toProto()
      ..classifierOptions = classifierOptions;
  }
}
