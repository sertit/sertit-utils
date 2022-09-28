# -*- coding: utf-8 -*-
# Copyright 2022, SERTIT-ICube - France, https://sertit.unistra.fr/
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
""" Script testing XML functions """
import os
import tempfile

import pandas as pd

from CI.SCRIPTS.script_utils import files_path, xml_path
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

    pd_xml = pd.read_xml(xml_str)
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
