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
"""Script testing XML functions"""

import io
import os
import tempfile

import pandas as pd

from ci.script_utils import files_path, xml_path
from sertit import ci, xml

ci.reduce_verbosity()


def _compare_xml_str(str_1: str, str_2: str) -> None:
    """
    Check XML converted to strings

    Args:
        str_1 (str): XML as string 1
        str_2 (str): XML as string 2
    """
    assert str_1.replace(" ", "").upper() == str_2.replace(" ", "").upper()


def _assert_str(str_1: str, str_2: str) -> None:
    """
    Check strings

    Args:
        str_1 (str): String 1
        str_2 (str): String 2
    """
    assert str_1 == str_2, f"str_1: {str_1} != str_2: {str_2}"


def test_xml():
    """Test XML functions"""

    # Check from pandas
    xml_str = """<?xml version='1.0' encoding='utf-8'?>
        <data>
         <row>
           <shape>square</shape>
           <degrees>360</degrees>
           <sides>4.0</sides>
         </row>
         <row>
           <shape>circle</shape>
           <degrees>360</degrees>
           <sides/>
         </row>
         <row>
           <shape>triangle</shape>
           <degrees>180</degrees>
           <sides>3.0</sides>
         </row>
        </data>
        """

    pd_xml = pd.read_xml(io.StringIO(xml_str))
    df_xml = xml.df_to_xml(pd_xml)
    _compare_xml_str(xml.to_string(df_xml), xml_str)

    # Check from dataset (using pandas is OK)
    pd_dict = {
        "Name": [
            "Owen Harris",
        ],
        "Age": [22],
        "Sex": ["male"],
    }
    pd_df = pd.DataFrame(pd_dict)

    attrs = list(pd_dict.keys())
    cv_xml = xml.convert_to_xml(pd_df, attrs)
    for attr in attrs:
        xml_attr = cv_xml.findtext(f".//{attr}")
        dict_attr = str(pd_dict[attr][0])
        assert xml_attr == dict_attr, f"{xml_attr=} =! {dict_attr=}"

    # Operations on XML
    xml.remove(cv_xml, "Age")
    assert cv_xml.findtext(".//Age") is None

    xml.add(cv_xml, "Age", 25)
    _assert_str(cv_xml.findtext(".//Age"), "25")

    xml.update_attrib(cv_xml, "Name", "prefix", "Mr.")
    _assert_str(cv_xml.find(".//Name").attrib["prefix"], "Mr.")

    xml.update_txt(cv_xml, "Age", 22)
    _assert_str(cv_xml.findtext(".//Age"), "22")

    xml.update_txt_fct(cv_xml, "Age", lambda x: round(int(x) / 10) * 10)
    _assert_str(cv_xml.findtext(".//Age"), "20")

    # Write
    true_xml = str(xml_path() / "true.xml")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_xml = os.path.join(tmp_dir, "tmp.xml")
        xml.write(cv_xml, tmp_xml)
        ci.assert_xml_equal(xml.read(true_xml), xml.read(tmp_xml))

    # Read archive
    # Based on `files.read_archived_xml`, so it is considered to work.
    # Just test the case with complete path to the archive
    l8_archived = files_path() / "LM05_L1TP_200030_20121230_20200820_02_T2_CI.zip"
    xml_archived = f"{l8_archived}!LM05_L1TP_200030_20121230_20200820_02_T2_CI/LM05_L1TP_200030_20121230_20200820_02_T2_MTL.xml"

    ci.assert_xml_equal(
        xml.read_archive(l8_archived, r".*_MTL\.xml"), xml.read_archive(xml_archived)
    )


def test_dict_xml():
    xml_dict = {
        "ACCODE": "LaSRC",
        "AREA_OR_POINT": "Area",
        "arop_ave_xshift(meters)": "0",
        "arop_ave_yshift(meters)": "0",
        "arop_ncp": "0",
        "arop_rmse(meters)": "0",
        "arop_s2_refimg": "NONE",
        "cloud_coverage": "46",
        "DATASTRIP_ID": "S2B_OPER_MSI_L1C_DS_2BPS_20221127T223847_S20221127T214906_N04.00",
        "HLS_PROCESSING_TIME": "2022-11-29T07:10:51Z",
        "HORIZONTAL_CS_CODE": "EPSG:32760",
        "HORIZONTAL_CS_NAME": "WGS84 / UTM zone 60S",
        "L1C_IMAGE_QUALITY": "NONE",
        "L1_PROCESSING_TIME": "2022-11-27T23:01:25.915427Z",
        "MEAN_SUN_AZIMUTH_ANGLE": "50.8114320532222",
        "MEAN_SUN_ZENITH_ANGLE": "37.2612285279726",
        "MEAN_VIEW_AZIMUTH_ANGLE": "104.448494899629",
        "MEAN_VIEW_ZENITH_ANGLE": "3.7539313224889",
        "MSI band 01 bandpass adjustment slope and offset": "0.995900,-0.000200",
        "NBAR_SOLAR_ZENITH": "38.4289484166757",
        "NCOLS": "3660",
        "NROWS": "3660",
        "OVR_RESAMPLING_ALG": "NEAREST",
        "PROCESSING_BASELINE": "04.00",
        "PRODUCT_URI": "S2B_MSIL1C_20221127T214909_N0400_R143_T60FXK_20221127T223847.SAFE",
        "SENSING_TIME": "2022-11-27T21:49:48.678784Z",
        "SPACECRAFT_NAME": "Sentinel-2B",
        "spatial_coverage": "53",
        "SPATIAL_RESOLUTION": "30",
        "TILE_ID": "S2B_OPER_MSI_L1C_TL_2BPS_20221127T223847_A029913_T60FXK_N04.00",
        "ULX": "600000",
        "ULY": "-5499960",
    }

    root = xml.dict_to_xml(xml_dict)

    for key, val in xml_dict.items():
        key_xml = key.replace(" ", "_").replace("(", "_").replace(")", "")
        _assert_str(root.findtext(f".//{key_xml}"), val)
