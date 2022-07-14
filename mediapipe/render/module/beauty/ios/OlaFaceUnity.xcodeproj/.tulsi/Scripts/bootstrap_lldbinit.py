#!/usr/bin/python3
# Copyright 2018 The Tulsi Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bootstraps the presence and setup of ~/.lldbinit-tulsiproj."""

import io
import os
import shutil
import sys


TULSI_LLDBINIT_FILE = os.path.expanduser('~/.lldbinit-tulsiproj')

CHANGE_NEEDED = 0
NO_CHANGE = 1
NOT_FOUND = 2


class BootstrapLLDBInit(object):
  """Bootstrap Xcode's preferred lldbinit for Bazel debugging."""

  def _ExtractLLDBInitContent(self, lldbinit_path, source_string,
                              add_source_string):
    """Extracts non-Tulsi content in a given lldbinit if needed.

    Args:
      lldbinit_path: Absolute path to the lldbinit we are writing to.
      source_string: String that we wish to write or remove from the lldbinit.
      add_source_string: Boolean indicating whether we intend to write or remove
        the source string.

    Returns:
      (int, [string]): A tuple featuring the status code along with the list
                       of strings representing content to write to lldbinit
                       that does not account for the Tulsi-generated strings.
                       Status code will be 0 if Tulsi-generated strings are
                       not all there. Status code will be 1 if we intend to
                       write Tulsi strings and all strings were accounted for.
                       Alternatively, if we intend to remove the Tulsi strings,
                       the status code will be 1 if none of the strings were
                       found. Status code will be 2 if the lldbinit file could
                       not be found.
    """
    if not os.path.isfile(lldbinit_path):
      return (NOT_FOUND, [])
    content = []
    with open(lldbinit_path) as f:
      ignoring = False

      # Split on the newline. This works as long as the last string isn't
      # suffixed with \n.
      source_lines = source_string.split('\n')

      source_idx = 0

      # If the last line was suffixed with \n, last elements would be length
      # minus 2, accounting for the extra \n.
      source_last = len(source_lines) - 1

      for line in f:

        # For each line found matching source_string, increment the iterator
        # and do not append that line to the list.
        if source_idx <= source_last and source_lines[source_idx] in line:

          # If we intend to write the source string and all lines were found,
          # return an error code with empty content.
          if add_source_string and source_idx == source_last:
            return (NO_CHANGE, [])

          # Increment for each matching line found.
          source_idx += 1
          ignoring = True

        if ignoring:

          # If the last line was found...
          if source_lines[source_last] in line:
            # Stop ignoring lines and continue appending to content.
            ignoring = False
          continue

        # If the line could not be found within source_string, append to the
        # content array.
        content.append(line)

    # If we intend to remove the source string and none of the lines to remove
    # were found, return an error code with empty content.
    if not add_source_string and source_idx == 0:
      return (NO_CHANGE, [])

    return (CHANGE_NEEDED, content)

  def _LinkTulsiLLDBInit(self, add_source_string):
    """Adds or removes a reference to ~/.lldbinit-tulsiproj to the primary lldbinit file.

    Xcode 8+ executes the contents of the first available lldbinit on startup.
    To help work around this, an external reference to ~/.lldbinit-tulsiproj is
    added to that lldbinit. This causes Xcode's lldb-rpc-server to load the
    possibly modified contents between Debug runs of any given app. Note that
    this only happens after a Debug session terminates; the cache is only fully
    invalidated after Xcode is relaunched.

    Args:
      add_source_string: Boolean indicating whether we intend to write or remove
        the source string.
    """

    # ~/.lldbinit-Xcode is the only lldbinit file that Xcode will read if it is
    # present, therefore it has priority.
    lldbinit_path = os.path.expanduser('~/.lldbinit-Xcode')
    if not os.path.isfile(lldbinit_path):
      # If ~/.lldbinit-Xcode does not exist, write the reference to
      # ~/.lldbinit-tulsiproj to ~/.lldbinit, the second lldbinit file that
      # Xcode will attempt to read if ~/.lldbinit-Xcode isn't present.
      lldbinit_path = os.path.expanduser('~/.lldbinit')

    # String that we plan to inject or remove from this lldbinit.
    source_string = ('# <TULSI> LLDB bridge [:\n'
                     '# This was autogenerated by Tulsi in order to modify '
                     'LLDB source-maps at build time.\n'
                     'command source %s\n' % TULSI_LLDBINIT_FILE +
                     '# ]: <TULSI> LLDB bridge')

    # Retrieve the contents of lldbinit if applicable along with a return code.
    return_code, content = self._ExtractLLDBInitContent(lldbinit_path,
                                                        source_string,
                                                        add_source_string)

    out = io.StringIO()

    if add_source_string:
      if return_code == CHANGE_NEEDED:
        # Print the existing contents of this ~/.lldbinit without any malformed
        # tulsi lldbinit block, and add the correct tulsi lldbinit block to the
        # end of it.
        for line in content:
          out.write(line)
      elif return_code == NO_CHANGE:
        # If we should ignore the contents of this lldbinit, and it has the
        # association with ~/.lldbinit-tulsiproj that we want, do not modify it.
        return

      # Add a newline after the source_string for protection from other elements
      # within the lldbinit file.
      out.write(source_string + '\n')
    else:
      if return_code != CHANGE_NEEDED:
        # The source string was not found in the lldbinit so do not modify it.
        return

      # Print the existing contents of this ~/.lldbinit without the tulsi
      # lldbinit block.
      for line in content:
        out.write(line)

      out.seek(0, os.SEEK_END)
      if out.tell() == 0:
        # The file did not contain any content other than the source string so
        # remove the file altogether.
        os.remove(lldbinit_path)
        return

    with open(lldbinit_path, 'w') as outfile:
      out.seek(0)
      # Negative length to make copyfileobj write the whole file at once.
      shutil.copyfileobj(out, outfile, -1)

  def __init__(self, do_inject_link=True):
    self._LinkTulsiLLDBInit(do_inject_link)


if __name__ == '__main__':
  BootstrapLLDBInit()
  sys.exit(0)
