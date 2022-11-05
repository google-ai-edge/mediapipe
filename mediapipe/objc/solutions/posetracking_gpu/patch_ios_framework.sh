#!/bin/bash

# Copyright (c) 2022 Baracoda. All rights reserved
#
# Copying this file via any medium without the prior written consent of Baracoda is strictly
# prohibited
#
# Proprietary and confidential

set -eu
set -o pipefail

for outputPath in $2
do
  fileName=$(basename "$outputPath" .framework)
  outputPath=$(dirname $outputPath)

  for inputPath in $1
  do
    if [[ $inputPath == *"$fileName"*.zip ]]; then
      fullPath=$(dirname "$inputPath")

      frameworkPath="$fullPath"/"$fileName".framework

      rm -rf "$fullPath"/"$fileName".framework
      unzip "$inputPath" -d "$fullPath"

      if [ -d "$frameworkPath".dSYM ]; then
        cp -R "$frameworkPath".dSYM "$outputPath"/"$fileName".framework.dSYM
      fi

      if [ -f "$frameworkPath"/module.modulemap ]; then
        mkdir "$frameworkPath"/Modules
        mv "$frameworkPath"/module.modulemap "$frameworkPath"/Modules
      fi

      cp -R "$frameworkPath" "$outputPath"/"$fileName".framework
    fi
  done;
done;