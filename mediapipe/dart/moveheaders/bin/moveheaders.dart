import 'dart:io' as io;
import 'package:moveheaders/moveheaders.dart' as moveheaders;
import 'package:args/args.dart';
import 'package:path/path.dart' as path;

final parser = ArgParser()
  ..addFlag(
    'overwrite',
    abbr: 'o',
    defaultsTo: true,
    help: 'If true, will overwrite existing header files '
        'at destination locations.',
  )
  ..addOption(
    'destination',
    abbr: 'd',
    help: 'The location of google/flutter-mediapipe. Defaults to being '
        'adjacent to google/mediapipe.',
  );

void main(List<String> arguments) {
  final results = parser.parse(arguments);
  final config = moveheaders.Options(
    allowOverwrite: results['overwrite'],
  );

  final io.Directory mediaPipeDirectory = moveheaders.findMediaPipeRoot();

  final passedDestination =
      results['destination'] != null && results['destination'] != '';
  final flutterMediaPipeDirectory = io.Directory(
    passedDestination
        ? results['destination']
        : path.joinAll([
            mediaPipeDirectory.parent.absolute.path,
            'flutter-mediapipe',
          ]),
  );

  if (!flutterMediaPipeDirectory.existsSync()) {
    io.stderr.writeln(
      'Could not find ${flutterMediaPipeDirectory.absolute.path}. '
      'Folder does not exist.',
    );
    io.exit(1);
  }

  moveheaders.moveHeaders(
    mediaPipeDirectory,
    flutterMediaPipeDirectory,
    config,
  );
}
