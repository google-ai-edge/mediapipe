import 'package:mediapipe/generated/mediapipe/calculators/calculators.dart';

import '../../../generated/mediapipe/framework/framework.dart';
import '../tasks.dart';

class TaskInfo<T extends TaskOptions> {
  TaskInfo({
    required this.taskGraph,
    required this.inputStreams,
    required this.outputStreams,
    required this.taskOptions,
  });
  String taskGraph;
  List<String> inputStreams;
  List<String> outputStreams;
  T taskOptions;

  CalculatorGraphConfig generateGraphConfig({
    bool enableFlowLimiting = false,
  }) {
    assert(inputStreams.isNotEmpty, 'TaskInfo.inputStreams must be non-empty');
    assert(
      outputStreams.isNotEmpty,
      'TaskInfo.outputStreams must be non-empty',
    );

    FlowLimiterCalculatorOptions.ext;
    final taskSubgraphOptions = CalculatorOptions();
    taskSubgraphOptions.addExtension(taskOptions.ext, taskOptions.toProto());

    if (!enableFlowLimiting) {
      return CalculatorGraphConfig.create()
        ..node.add(
          CalculatorGraphConfig_Node.create()
            ..calculator = taskGraph
            ..inputStream.addAll(inputStreams)
            ..outputStream.addAll(outputStreams)
            ..options = taskSubgraphOptions,
        );
    }

    // When a FlowLimiterCalculator is inserted to lower the overall graph
    // latency, the task doesn't guarantee that each input must have the
    // corresponding output.
    final taskSubgraphInputs =
        inputStreams.map<String>(_addStreamNamePrefix).toList();
    String finishedStream = 'FINISHED: ${_stripTagIndex(outputStreams.first)}';

    final flowLimiterOptions = CalculatorOptions.create();
    flowLimiterOptions.setExtension(
      FlowLimiterCalculatorOptions.ext,
      FlowLimiterCalculatorOptions.create()
        ..maxInFlight = 1
        ..maxInQueue = 1,
    );

    final flowLimiter = CalculatorGraphConfig_Node.create()
      ..calculator = 'FlowLimiterCalculator'
      ..inputStreamInfo.add(
        InputStreamInfo.create()
          ..tagIndex = 'FINISHED'
          ..backEdge = true,
      )
      ..inputStream.addAll(inputStreams.map<String>(_stripTagIndex).toList())
      ..inputStream.add(finishedStream)
      ..outputStream.addAll(
        taskSubgraphInputs.map<String>(_stripTagIndex).toList(),
      )
      ..options = flowLimiterOptions;

    final config = CalculatorGraphConfig.create()
      ..node.add(
        CalculatorGraphConfig_Node.create()
          ..calculator = taskGraph
          ..inputStream.addAll(taskSubgraphInputs)
          ..outputStream.addAll(outputStreams)
          ..options = taskSubgraphOptions,
      )
      ..node.add(flowLimiter)
      ..inputStream.addAll(inputStreams)
      ..outputStream.addAll(outputStreams);
    return config;
  }
}

String _stripTagIndex(String tagIndexName) => tagIndexName.split(':').last;
String _addStreamNamePrefix(String tagIndexName) {
  final split = tagIndexName.split(':');
  split.last = 'trottled_${split.last}';
  return split.join(':');
}
