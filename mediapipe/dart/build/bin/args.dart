import 'package:args/command_runner.dart';
import 'package:build/sync_headers.dart';

final runner = CommandRunner(
  'build',
  'Performs build operations for google/flutter-mediapipe that '
      'depend on contents in this repository',
)..addCommand(SyncHeadersCommand());
