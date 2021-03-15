""" Script testing the snap functions """

import logging
from sertit import snap

LOGGER = logging.getLogger("Test_logger")


def test_snap():
    """ Testing SNAP functions """
    assert snap.bytes2snap(32000) == '31K'

    # Do not test everything here, depends on the computer...
    cli = snap.get_gpt_cli("graph_path", other_args=[], display_snap_opt=True)
    must_appear = ['gpt', '"graph_path"', '-q', '-J-Dsnap.log.level=WARNING',
                   '-J-Dsnap.jai.defaultTileSize=2048', '-J-Dsnap.dataio.reader.tileWidth=2048',
                   '-J-Dsnap.dataio.reader.tileHeigh=2048', '-J-Dsnap.jai.prefetchTiles=true']

    for substr in must_appear:
        assert substr in cli
