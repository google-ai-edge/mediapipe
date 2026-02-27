import 'dart:io' as io;
import 'package:io/ansi.dart' as ansi;

class Command {
  Command(
    this.arguments, {
    this.stdOutHandler,
    this.stdErrHandler,
    this.workingDirectory,
    bool? debug,
  })  : assert(arguments.isNotEmpty),
        _debug = debug ?? false;

  final bool _debug;
  final String? workingDirectory;
  final List<String> arguments;
  void Function(String)? stdOutHandler;
  void Function(String)? stdErrHandler;

  io.ProcessResult? _result;

  /// Runs a command and scans its stdout contents for a given needle.
  /// If the needle is found, the command is considered to have completed
  /// successfully.
  factory Command.needleInOutput(
    List<String> arguments, {
    required String needle,
    List<String> ifMissing = const <String>[],
    bool shouldExitIfMissing = true,
  }) {
    return Command(
      arguments,
      stdOutHandler: (String output) {
        bool foundNeedle = false;
        bool contains(String line) => line.contains(needle);
        if (output.split('\n').any(contains)) {
          foundNeedle = true;
        }
        if (!foundNeedle) {
          for (String line in ifMissing) {
            io.stderr.writeln(ansi.wrapWith(line, [ansi.red]));
          }
          if (shouldExitIfMissing) {
            io.exit(1);
          }
        }
      },
    );
  }

  factory Command.which(
    String commandName, {
    String? documentationUrl,
    bool shouldExitIfMissing = true,
  }) {
    return Command(
      ['which', commandName],
      stdOutHandler: (String output) {
        if (output.isEmpty) {
          io.stderr.writeAll([
            ansi.wrapWith(
              '$commandName not installed or not on path\n',
              [ansi.red],
            ),
            if (documentationUrl != null)
              ansi.wrapWith(
                'Visit $documentationUrl for installation instructions\n',
                [ansi.red],
              ),
          ]);
          if (shouldExitIfMissing) {
            io.exit(1);
          }
        }
      },
    );
  }

  factory Command.run(
    List<String> command, {
    bool? debug,
    String? workingDirectory,
  }) =>
      Command(
        command,
        stdOutHandler: (String output) =>
            output.trim().isNotEmpty ? io.stdout.writeln(output.trim()) : null,
        stdErrHandler: (String output) =>
            output.trim().isNotEmpty ? io.stderr.writeln(output.trim()) : null,
        debug: debug,
        workingDirectory: workingDirectory,
      );

  Future<void> run() async {
    if (_debug) {
      io.stdout.writeln('>>> ${arguments.join(' ')}');
    }
    _result = await io.Process.run(
      arguments.first,
      arguments.sublist(1),
      workingDirectory: workingDirectory,
    );
    stdOutHandler?.call(_result!.stdout);
    stdErrHandler?.call(_result!.stderr);
  }

  int? get exitCode => _result?.exitCode;
}
