# Copyright 2017 The Tulsi Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging routines used by Tulsi scripts."""

import logging
import logging.handlers
import os
import sys


def validity_check():
  """Returns a warning message from logger initialization, if applicable."""
  return None


class Logger(object):
  """Tulsi specific logging."""

  def __init__(self):
    logging_dir = os.path.expanduser('~/Library/Application Support/Tulsi')
    if not os.path.exists(logging_dir):
      os.mkdir(logging_dir)

    logfile = os.path.join(logging_dir, 'build_log.txt')

    # Currently only creates a single logger called 'tulsi_logging'. If
    # additional loggers are needed, consider adding a name attribute to the
    # Logger.
    self._logger = logging.getLogger('tulsi_logging')
    self._logger.setLevel(logging.INFO)

    try:
      file_handler = logging.handlers.RotatingFileHandler(logfile,
                                                          backupCount=20)
      file_handler.setLevel(logging.INFO)
      # Create a new log file for each build.
      file_handler.doRollover()
      self._logger.addHandler(file_handler)
    except (IOError, OSError) as err:
      filename = 'none'
      if hasattr(err, 'filename'):
        filename = err.filename
      sys.stderr.write('Failed to set up logging to file: %s (%s).\n' %
                       (os.strerror(err.errno), filename))
      sys.stderr.flush()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    self._logger.addHandler(console)

  def log_bazel_message(self, message):
    self._logger.info(message)

  def log_action(self, action_name, action_id, seconds, start=None, end=None):
    """Logs the start, duration, and end of an action."""
    del action_id  # Unused by this logger.
    if start:
      self._logger.info('<**> %s start: %f', action_name, start)

    # Log to file and print to stdout for display in the Xcode log.
    self._logger.info('<*> %s completed in %0.3f ms',
                      action_name, seconds * 1000)

    if end:
      self._logger.info('<**> %s end: %f', action_name, end)
