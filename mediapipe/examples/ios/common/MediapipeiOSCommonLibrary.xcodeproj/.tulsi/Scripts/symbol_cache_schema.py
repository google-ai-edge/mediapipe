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

"""Manages our dSYM SQLite database schema."""

import errno
import os
import sqlite3


SQLITE_SYMBOL_CACHE_PATH = os.path.expanduser('~/Library/Application Support/'
                                              'Tulsi/Scripts/symbol_cache.db')


class SymbolCacheSchema(object):
  """Defines and updates the SQLite database used for DBGShellCommands."""

  current_version = 1

  def UpdateSchemaV1(self, connection):
    """Updates the database to the v1 schema.

    Args:
      connection: Connection to the database that needs to be updated.

    Returns:
      True if the database reported that it was updated to v1.
      False if not.
    """
    # Create the table (schema version 1).
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE symbol_cache('
                   'uuid TEXT PRIMARY KEY, '
                   'dsym_path TEXT, '
                   'architecture TEXT'
                   ');')
    # NOTE: symbol_cache (uuid) already has an index, as the PRIMARY KEY.
    # Create a unique index to keep dSYM paths and architectures unique.
    cursor.execute('CREATE UNIQUE INDEX idx_dsym_arch '
                   'ON '
                   'symbol_cache('
                   'dsym_path, '
                   'architecture'
                   ');')
    cursor.execute('PRAGMA user_version = 1;')

    # Verify the updated user_version, as confirmation of the update.
    cursor.execute('PRAGMA user_version;')
    return cursor.fetchone()[0] == 1

  def VerifySchema(self, connection):
    """Updates the database to the latest schema.

    Args:
      connection: Connection to the database that needs to be updated.

    Returns:
      True if the database reported that it was updated to the latest schema.
      False if not.
    """
    cursor = connection.cursor()
    cursor.execute('PRAGMA user_version;')  # Default is 0
    db_version = cursor.fetchone()[0]

    # Update to the latest schema in the given database, if necessary.
    if db_version < self.current_version:
      # Future schema updates will build on this.
      if self.UpdateSchemaV1(connection):
        db_version = 1

    # Return if the database has been updated to the latest schema.
    return db_version == self.current_version

  def InitDB(self, db_path):
    """Initializes a new connection to a SQLite database.

    Args:
      db_path: String representing a reference to the SQLite database.

    Returns:
      A sqlite3.connection object representing an active connection to
      the database referenced by db_path.
    """
    # If this is not an in-memory SQLite database...
    if ':memory:' not in db_path:
      # Create all subdirs before we create a new db or connect to existing.
      if not os.path.isfile(db_path):
        try:
          os.makedirs(os.path.dirname(db_path))
        except OSError as e:
          if e.errno != errno.EEXIST:
            raise

    connection = sqlite3.connect(db_path)

    # Update to the latest schema and return the db connection.
    if self.VerifySchema(connection):
      return connection
    else:
      return None

  def __init__(self, db_path=SQLITE_SYMBOL_CACHE_PATH):
    self.connection = self.InitDB(db_path)

  def __del__(self):
    if self.connection:
      self.connection.close()
