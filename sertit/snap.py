# Copyright 2025, SERTIT-ICube - France, https://sertit.unistra.fr/
# This file is part of sertit-utils project
#     https://github.com/sertit/sertit-utils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SNAP tools
"""

import logging
import os
import subprocess

import psutil
from packaging.version import Version

from sertit import misc, perf, strings
from sertit.logs import SU_NAME

MAX_CORES = perf.MAX_CORES
SU_MAX_CORE = perf.SU_MAX_CORE


JAVA_OPTS_XMX = "JAVA_OPTS_XMX"
"""
Maximum memory to use, default is 95% of the total virtual memory.
You can update it with the environment variable :code:`JAVA_OPTS_XMX`.
"""

SU_SNAP_TILE_SIZE = "SERTIT_UTILS_SNAP_TILE_SIZE"
"""
SNAP tile size, 512 by default.
You can update it with the environment variable :code:`SERTIT_UTILS_SNAP_TILE_SIZE`.
"""

SU_SNAP_LOG_LEVEL = "SERTIT_UTILS_SNAP_LOG_LEVEL"
"""
SNAP log level, WARNING by default.
You can update it with the environment variable :code:`SERTIT_UTILS_SNAP_LOG_LEVEL`.
"""

LOGGER = logging.getLogger(SU_NAME)


def bytes2snap(nof_bytes: int) -> str:
    """
    Convert nof bytes into snap-compatible Java options.

    Example:
        >>> bytes2snap(32000)
        '31K'

    Args:
        nof_bytes (int): Byte nb

    Returns:
        str: Human-readable in bits

    """
    symbols = ("K", "M", "G", "T", "P", "E", "Z", "Y")
    prefix = {}
    for idx, sym in enumerate(symbols):
        prefix[sym] = 1 << (idx + 1) * 10
    for sym in reversed(symbols):
        if nof_bytes >= prefix[sym]:
            value = int(float(nof_bytes) / prefix[sym])
            return f"{value}{sym}"
    return f"{nof_bytes}B"


def get_gpt_cli(
    graph_path: str, other_args: list, display_snap_opt: bool = False
) -> list:
    """
    Get GPT command line with system OK optimizations.
    To see options, type this command line with --diag (but it won't run the graph)

    Example:
        >>> get_gpt_cli("graph_path", other_args=[], display_snap_opt=True)
        SNAP Release version 8.0
        SNAP home: C:/Program Files/snap/bin/..
        SNAP debug: null
        SNAP log level: WARNING
        Java home: c:/program files/snap/jre/jre
        Java version: 1.8.0_242
        Processors: 16
        Max memory: 53.3 GB
        Cache size: 30.0 GB
        Tile parallelism: 14
        Tile size: 2048 x 2048 pixels
        To configure your gpt memory usage:
        Edit snap/bin/gpt.vmoptions
        To configure your gpt cache size and parallelism:
        Edit .snap/etc/snap.properties or gpt -c ${cachesize-in-GB}G -q ${parallelism}

        ['gpt', '"graph_path"', '-q', 14, '-J-Xms2G -J-Xmx60G', '-J-Dsnap.log.level=WARNING',
        '-J-Dsnap.jai.defaultTileSize=2048', '-J-Dsnap.dataio.reader.tileWidth=2048',
        '-J-Dsnap.dataio.reader.tileHeigh=2048', '-J-Dsnap.jai.prefetchTiles=true', '-c 30G']

    Args:
        graph_path (str): Graph path
        other_args (list): Other args as a list such as :code:`['-Pfile="in_file.zip", '-Pout="out_file.dim"']`
        display_snap_opt (bool): Display SNAP options via --diag

    Returns:
        list: GPT command line as a list
    """
    # Overload with env variables
    max_cores = perf.get_max_cores()
    tile_size = int(os.getenv(SU_SNAP_TILE_SIZE, 512))
    snap_log_level = os.getenv(SU_SNAP_LOG_LEVEL, "WARNING")
    max_mem = int(os.environ.get(JAVA_OPTS_XMX, 0.95 * psutil.virtual_memory().total))

    gpt_cli = [
        "gpt",
        strings.to_cmd_string(graph_path),
        "-q",
        max_cores,  # Maximum parallelism
        f"-J-Xms2G -J-Xmx{bytes2snap(max_mem)}",  # Initially/max allocated memory
        f"-J-Dsnap.log.level={snap_log_level}",
        f"-J-Dsnap.jai.defaultTileSize={tile_size}",
        f"-J-Dsnap.dataio.reader.tileWidth={tile_size}",
        f"-J-Dsnap.dataio.reader.tileHeight={tile_size}",
        "-J-Dsnap.dataio.bigtiff.compression.type=LZW",
        "-J-Dsnap.dataio.bigtiff.tiling.width=512",
        "-J-Dsnap.dataio.bigtiff.tiling.height=512",
        "-J-Dsnap.jai.prefetchTiles=true",
        f"-c {bytes2snap(int(0.5 * max_mem))}",  # Tile cache set to 50% of max memory (up to 75%)
        # '-x',
        *other_args,
    ]  # Clears the internal tile cache after writing a complete row to the target file

    # LOGs
    LOGGER.debug(gpt_cli)
    if display_snap_opt:
        misc.run_cli(gpt_cli + ["--diag"])

    return gpt_cli


def get_snap_version() -> Version:
    """Get SNAP version with a call to GPT --diag"""
    snap_version = None
    try:
        output = subprocess.run(["gpt", "--diag"], capture_output=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("'gpt' not found in your PATH") from exc

    stdout = output.stdout.decode("utf-8")

    if stdout is not None:
        version_str = stdout.split("\n")
        try:
            version_str = [v for v in version_str if "version" in v][0]
        except IndexError as ex:
            LOGGER.debug(ex)
        else:
            snap_version = version_str.split(" ")[-1]

            snap_version = Version(snap_version)

    return snap_version
