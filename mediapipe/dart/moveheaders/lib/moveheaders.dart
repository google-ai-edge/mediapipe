import 'dart:convert';
import 'dart:io' as io;
import 'package:io/ansi.dart';
import 'package:path/path.dart' as path;
import 'package:process/process.dart';

/// google/mediapipe paths
final containers = 'mediapipe/tasks/c/components/containers';
final processors = 'mediapipe/tasks/c/components/processors';
final originCore = 'mediapipe/tasks/c/core';
final tc = 'mediapipe/tasks/c/text/text_classifier';

/// google/flutter-mediapipe paths
final core = 'packages/mediapipe-core/third_party/mediapipe/';
final text = 'packages/mediapipe-task-text/third_party/mediapipe/';

/// First string is its location in this repository;
/// Second string is its location in `google/flutter-mediapipe`,
/// Third string is the file name
List<(String, String, String)> headerPaths = [
  (containers, core, 'category.h'),
  (containers, core, 'classification_result.h'),
  (originCore, core, 'base_options.h'),
  (processors, core, 'classifier_options.h'),
  (tc, text, 'text_classifier.h'),
];

class Options {
  const Options({
    required this.allowOverwrite,
  });

  final bool allowOverwrite;
}

Future<void> moveHeaders(
  io.Directory mediaPipe,
  io.Directory flutterMediaPipe,
  Options config,
) async {
  final mgr = LocalProcessManager();
  for (final tup in headerPaths) {
    final headerFile = io.File(path.joinAll(
      [mediaPipe.absolute.path, tup.$1, tup.$3],
    ));
    if (!headerFile.existsSync()) {
      io.stderr.writeln(
        'Expected to find ${headerFile.path}, but '
        'file does not exist.',
      );
      io.exit(1);
    }
    final destinationPath = path.joinAll(
      [flutterMediaPipe.absolute.path, tup.$2, tup.$3],
    );
    if (io.File(destinationPath).existsSync() && !config.allowOverwrite) {
      io.stdout.writeln(
        'Warning: Not overwriting existing file at $destinationPath. '
        'Skipping ${tup.$3}.',
      );
      continue;
    }

    io.stderr.writeln(wrapWith('Moving ${tup.$3}', [green]));
    final process = await mgr.start([
      'cp',
      headerFile.path,
      destinationPath,
    ]);
    int processExitCode = await process.exitCode;
    if (processExitCode != 0) {
      final processStdErr = utf8.decoder.convert(await process.stderr.drain());
      io.stderr.write(wrapWith(processStdErr, [red]));
      io.exit(processExitCode);
    }
  }
}

io.Directory findMediaPipeRoot() {
  io.Directory dir = io.Directory(path.current);
  while (true) {
    if (isMediaPipeRoot(dir)) {
      return dir;
    }
    dir = dir.parent;
    if (dir.parent.path == dir.path) {
      io.stderr.writeln(
        wrapWith(
          'Failed to find google/mediapipe root directory. '
          'Did you execute this command from within the repository?',
          [red],
        ),
      );
      io.exit(1);
    }
  }
}

bool isMediaPipeRoot(io.Directory dir) {
  final bazelrcExists = io.File(
    path.joinAll(
      [dir.absolute.path, '.bazelrc'],
    ),
  ).existsSync();
  final mediapipeExists = io.Directory(
    path.joinAll(
      [dir.absolute.path, 'mediapipe'],
    ),
  ).existsSync();
  final dotGitExists = io.Directory(
    path.joinAll(
      [dir.absolute.path, '.git'],
    ),
  ).existsSync();
  return bazelrcExists && mediapipeExists && dotGitExists;
}
