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

"""Parses a stream of JSON build event protocol messages from a file."""

import json


class _FileLineReader(object):
  """Reads lines from a streaming file.

  This will repeatedly check the file for an entire line to read. It will
  buffer partial lines until they are completed.
  This is meant for files that are being modified by an long-living external
  program.
  """

  def __init__(self, file_obj):
    """Creates a new FileLineReader object.

    Args:
      file_obj: The file object to watch.

    Returns:
      A FileLineReader instance.
    """
    self._file_obj = file_obj
    self._buffer = []

  def check_for_changes(self):
    """Checks the file for any changes, returning the line read if any."""
    line = self._file_obj.readline()
    self._buffer.append(line)

    # Only parse complete lines.
    if not line.endswith('\n'):
      return None
    full_line = ''.join(self._buffer)
    del self._buffer[:]
    return full_line


class BazelBuildEvent(object):
  """Represents a Bazel Build Event.

  Public Properties:
    event_dict: the source dictionary for this event.
    stdout: stdout string, if any.
    stderr: stderr string, if any.
    files: list of file URIs.
  """

  def __init__(self, event_dict):
    """Creates a new BazelBuildEvent object.

    Args:
      event_dict: Dictionary representing a build event

    Returns:
      A BazelBuildEvent instance.
    """
    self.event_dict = event_dict
    self.stdout = None
    self.stderr = None
    self.files = []
    if 'progress' in event_dict:
      self._update_fields_for_progress(event_dict['progress'])
    if 'namedSetOfFiles' in event_dict:
      self._update_fields_for_named_set_of_files(event_dict['namedSetOfFiles'])

  def _update_fields_for_progress(self, progress_dict):
    self.stdout = progress_dict.get('stdout')
    self.stderr = progress_dict.get('stderr')

  def _update_fields_for_named_set_of_files(self, named_set):
    files = named_set.get('files', [])
    for file_obj in files:
      uri = file_obj.get('uri', '')
      if uri.startswith('file://'):
        self.files.append(uri[7:])


class BazelBuildEventsWatcher(object):
  """Watches a build events JSON file."""

  def __init__(self, json_file, warning_handler=None):
    """Creates a new BazelBuildEventsWatcher object.

    Args:
      json_file: The JSON file object to watch.
      warning_handler: Handler function for warnings accepting a single string.

    Returns:
      A BazelBuildEventsWatcher instance.
    """
    self.file_reader = _FileLineReader(json_file)
    self.warning_handler = warning_handler
    self._read_any_events = False

  def has_read_events(self):
    return self._read_any_events

  def check_for_new_events(self):
    """Checks the file for new BazelBuildEvents.

    Returns:
      A list of all new BazelBuildEvents.
    """
    new_events = []
    while True:
      line = self.file_reader.check_for_changes()
      if not line:
        break
      try:
        build_event_dict = json.loads(line)
      except (UnicodeDecodeError, ValueError) as e:
        handler = self.warning_handler
        if handler:
          handler('Could not decode BEP event "%s"\n' % line)
          handler('Received error of %s, "%s"\n' % (type(e), e))
        break
      self._read_any_events = True
      build_event = BazelBuildEvent(build_event_dict)
      new_events.append(build_event)
    return new_events
