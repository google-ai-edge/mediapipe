import 'package:args/args.dart';
import 'package:dart_builder/dart_builder.dart' as db;

final argParser = ArgParser()
  ..addOption(
    'protoc',
    help: 'Override the path to protoc',
  )
  ..addOption(
    'protobuf',
    help: 'Supply a path to google/protobuf if it is not '
        'next to google/mediapipe',
  )
  ..addOption('output', abbr: 'o');

Future<void> main(List<String> arguments) async {
  final results = argParser.parse(arguments);
  final options = db.DartProtoBuilderOptions(
    protocPath: results['protoc'],
    protobufPath: results['protobuf'],
    outputPath: results['output'],
  );

  final protoBuilder = db.DartProtoBuilder(options);
  await protoBuilder.run();
}
