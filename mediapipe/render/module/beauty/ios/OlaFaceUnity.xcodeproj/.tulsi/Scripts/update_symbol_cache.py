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

"""Update the Tulsi dSYM symbol cache."""

import sqlite3
from symbol_cache_schema import SQLITE_SYMBOL_CACHE_PATH
from symbol_cache_schema import SymbolCacheSchema


class UpdateSymbolCache(object):
  """Provides a common interface to update a UUID referencing a dSYM."""

  def UpdateUUID(self, uuid, dsym_path, arch):
    """Updates a given UUID entry in the database.

    Args:
      uuid: A UUID representing a binary slice in the dSYM bundle.
      dsym_path: An absolute path to the dSYM bundle.
      arch: The binary slice's architecture.

    Returns:
      None: If no error occurred in inserting the new set of values.
      String: If a sqlite3.error was raised upon attempting to store new
              values into the dSYM cache.
    """
    con = self.cache_schema.connection
    cur = con.cursor()
    # Relies on the UNIQUE constraint between dsym_path + architecture to
    # update the UUID if dsym_path and arch match an existing pair, or
    # create a new row if this combination of dsym_path and arch is unique.
    try:
      cur.execute('INSERT OR REPLACE INTO symbol_cache '
                  '(uuid, dsym_path, architecture) '
                  'VALUES("%s", "%s", "%s");' % (uuid, dsym_path, arch))
      con.commit()
    except sqlite3.Error as e:
      return e.message
    return None

  def __init__(self, db_path=SQLITE_SYMBOL_CACHE_PATH):
    self.cache_schema = SymbolCacheSchema(db_path)
