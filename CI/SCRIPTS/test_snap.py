# -*- coding: utf-8 -*-
# Copyright 2021, SERTIT-ICube - France, https://sertit.unistra.fr/
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
""" Script testing SNAP functions """

import logging
import shutil

import pytest

from sertit import snap

LOGGER = logging.getLogger("Test_logger")


def test_b2snap():
    """Testing SNAP functions"""
    assert snap.bytes2snap(32000) == "31K"


@pytest.mark.skipif(
    shutil.which("gpt") is None, reason="Only works if SNAP GPT's exe can be found."
)
def test_snap():
    # Do not test everything here, depends on the computer...
    cli = snap.get_gpt_cli("graph_path", other_args=[], display_snap_opt=True)
    must_appear = [
        "gpt",
        '"graph_path"',
        "-q",
        "-J-Dsnap.log.level=WARNING",
        "-J-Dsnap.jai.defaultTileSize=2048",
        "-J-Dsnap.dataio.reader.tileWidth=2048",
        "-J-Dsnap.dataio.reader.tileHeigh=2048",
        "-J-Dsnap.jai.prefetchTiles=true",
    ]

    for substr in must_appear:
        assert substr in cli
