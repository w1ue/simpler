#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
build_dir="$repo_dir/build/wse-persistent"
python_bin="$repo_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Create the project-local environment: python3 -m venv .venv" >&2
    exit 1
fi
"$python_bin" -c 'import numpy'
cmake -S "$repo_dir/src/wse/platform/sim/persistent" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$python_bin"
cmake --build "$build_dir" --parallel 2
ctest --test-dir "$build_dir" --output-on-failure
echo "Results: $build_dir/validation/{single,multi}/results.json"
