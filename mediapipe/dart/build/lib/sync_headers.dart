import 'dart:convert';
import 'dart:io' as io;
import 'package:args/command_runner.dart';
import 'package:build/repo_finder.dart';
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

class SyncHeadersCommand extends Command with RepoFinderMixin {
  @override
  String description = 'Syncs header files to google/flutter-mediapipe';
  @override
  String name = 'sync';

  SyncHeadersCommand() {
    argParser.addFlag(
      'overwrite',
      abbr: 'o',
      defaultsTo: true,
      help: 'If true, will overwrite existing header files '
          'at destination locations.',
    );
    addDestinationOption(argParser);
  }

  @override
  Future<void> run() async {
    final io.Directory mediaPipeDirectory = findMediaPipeRoot();
    final io.Directory flutterMediaPipeDirectory = findFlutterMediaPipeRoot(
      mediaPipeDirectory,
      argResults!['destination'],
    );

    final config = Options(
      allowOverwrite: argResults!['overwrite'],
    );

    return moveHeaders(
      mediaPipeDirectory,
      flutterMediaPipeDirectory,
      config,
    );
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
        final processStdErr =
            utf8.decoder.convert(await process.stderr.drain());
        io.stderr.write(wrapWith(processStdErr, [red]));
        io.exit(processExitCode);
      }
    }
  }
}

class Options {
  const Options({
    required this.allowOverwrite,
  });

  final bool allowOverwrite;
}
