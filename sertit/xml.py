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
"""Tools concerning XML management, simplifying lxml.etree"""

import contextlib
import logging
from datetime import datetime
from typing import Any, Callable

from lxml.etree import (
    ElementTree,
    XMLSyntaxError,
    _Element,
    fromstring,
    parse,
    tostring,
)
from lxml.html.builder import E

from sertit import AnyPath, files, path
from sertit.logs import SU_NAME
from sertit.misc import ListEnum
from sertit.types import AnyPathStrType

UTF_8 = "UTF-8"

LOGGER = logging.getLogger(SU_NAME)


def read(xml_path: AnyPathStrType) -> _Element:
    """
    Read an XML file, even stored on the cloud

    Args:
        path (AnyPathStrType): Path to the XML file

    Returns:
        _Element: XML Root
    """
    xml_path = AnyPath(xml_path)
    try:
        if path.is_cloud_path(xml_path):
            try:
                # Try using read_text (faster)
                root = fromstring(xml_path.read_text())
            except ValueError:
                # Try using read_bytes
                # Slower but works with:
                # {ValueError}Unicode strings with encoding declaration are not supported.
                # Please use bytes input or XML fragments without declaration.
                root = fromstring(xml_path.read_bytes())
        else:
            # pylint: disable=I1101:
            # Module 'lxml.etree' has no 'parse' member, but source is unavailable.
            xml_tree = parse(str(xml_path))
            root = xml_tree.getroot()

    except XMLSyntaxError as exc:
        raise ValueError(f"Invalid metadata XML for {xml_path}!") from exc

    return root


def read_archive(
    path: AnyPathStrType, regex: str = None, file_list: list = None
) -> _Element:
    """
    Read an XML file from inside an archive (zip or tar)
    Convenient duplicate of :code:`files.read_archived_xml`

    Manages two cases:

    - complete path to an XML file stored inside an archive. In this case the filetree from inside the archive should be separated with a :code:`!`. Don't need to start with zip or tar
    - path to the archive plus a regex looking inside the archive. Duplicate behaviour to :py:func:`files.read_archived_xml`

    Args:
        path (AnyPathStrType): Path to the XML file, stored inside an archive or path to the archive itself
        regex (str): Optional. If specified, the path should be the archive path and the regex should be the key to find the XML file inside the archive.
        file_list (list): List of files contained in the archive. Optional, if not given it will be re-computed.

    Returns:
        _Element: XML Root
    """

    try:
        if not regex:
            path, basename = str(path).split("!")
            regex = basename
            if path.startswith("zip://") or path.startswith("tar://"):
                path = path[5:]

        return files.read_archived_xml(path, regex, file_list=file_list)

    except XMLSyntaxError as exc:
        raise ValueError(f"Invalid metadata XML for {path}!") from exc


def write(xml: _Element, path: str) -> None:
    """
    Write an Element to disk

    Args:
        xml (_Element): XML root
        path (str): Path where to write the XML file
    """
    ElementTree(xml).write(str(path), pretty_print=True)


def add(el: _Element, field: str, value: Any) -> None:
    """
    Add in place a field to a given element

    Args:
        el (_Element): Element to complete
        field (str): New field
        value: Value to set
    """
    el.append(E(field, str(value)))


def remove(xml: _Element, field: str) -> None:
    """
    Remove in place field from a :code:`lxml _Element`

    Args:
        xml (_Element): Root XML
        field (str): Field to remove
    """
    [el.getparent().remove(el) for el in xml.iterfind(f".//{field}")]


def update_attrib(xml: _Element, field: str, attribute: str, value: Any) -> None:
    """
    Update in place an attribute of a field with a given value

    Args:
        xml (_Element): Root XML
        field (str): Field to update
        attribute (str): Attribute to update
        value (Any): Value to set
    """
    [el.attrib.update({attribute: str(value)}) for el in xml.iterfind(f".//{field}")]


def update_txt(xml: _Element, field: str, value: Any) -> None:
    """
    Update in place a text of a field

    Args:
        xml (_Element): Root XML
        field (str): Field to update
        value (Any): Value to set
    """
    try:
        xml.find(f".//{field}").text = str(value)
    except AttributeError:
        LOGGER.warning(f"Not existing {field} in XML!")
        pass


def update_txt_fct(xml: _Element, field: str, fct: Callable) -> None:
    """
    Update in place a text of a field by applying a function to the value of the given field

    Args:
        xml (_Element): Root XML
        field (str): Field to update
        fct (Callable): Function to apply
    """
    try:
        elem = xml.find(f".//{field}")
        value = elem.text
        elem.text = str(fct(value))
    except AttributeError:
        LOGGER.warning(f"Not existing {field} in XML!")
        pass


def convert_to_xml(src_ds: Any, attributes: list) -> _Element:
    """
    Convert any dataset containing the given attributes to :code:`lxml _Element`
    (i.e. netcdf dataset)

    Args:
        src_ds (Any): Any dataset containing the given attribute list
        attributes(list): List of attributes to set in the wanted XML

    Returns:
        _Element: Wanted XML

    """
    # Create XML attributes
    global_attr = []
    for attr in attributes:
        if hasattr(src_ds, attr):
            # Get it formatted
            val = getattr(src_ds, attr)
            if isinstance(val, ListEnum):
                str_val = val.value
            elif isinstance(val, datetime):
                str_val = val.isoformat()
            else:
                with contextlib.suppress(AttributeError):
                    # gpd, pd...
                    val = val.iat[0]
                str_val = str(val)
            global_attr.append(E(attr, str_val))

    xml = E.data(*global_attr)
    xml_el = fromstring(
        tostring(xml, pretty_print=True, xml_declaration=True, encoding=UTF_8)
    )

    return xml_el


def dict_to_xml(dict_to_cv: dict, attributes: list = None) -> _Element:
    """
    Convert any dict containing the given attributes to a :code:`lxml _Element`.

    Replacements in keys:

    - :code:`" "` to :code:`"_"`
    - :code:`"("` to :code:`"_"`
    - :code:`")"` to :code:`""`

    Args:
        dict_to_cv (Any): Dict to convert into a XML
        attributes(list): List of attributes to set in the wanted XML

    Returns:
        _Element: Wanted XML

    """
    # Create XML attributes
    global_attr = []

    if attributes is None:
        attributes = dict_to_cv.keys()

    for attr in attributes:
        val = dict_to_cv.get(attr)
        if val is not None:
            # Get it formatted
            if isinstance(val, ListEnum):
                str_val = val.value
            elif isinstance(val, datetime):
                str_val = val.isoformat()
            else:
                with contextlib.suppress(AttributeError):
                    # gpd, pd...
                    val = val.iat[0]
                str_val = str(val)
            global_attr.append(
                E(attr.replace(" ", "_").replace("(", "_").replace(")", ""), str_val)
            )

    xml = E.data(*global_attr)
    xml_el = fromstring(
        tostring(xml, pretty_print=True, xml_declaration=True, encoding=UTF_8)
    )

    return xml_el


def df_to_xml(src_ds: Any) -> _Element:
    """
    Convert a :code:`pandas.DataFrame` or similar (which has a :code:`.to_xml()` function) to a :code:`lxml _Element`

    Args:
        src_ds:

    Returns:
        _Element: Wanted XML
    """
    return fromstring(bytes(src_ds.to_xml(index=False), UTF_8))


def to_string(xml: _Element) -> str:
    """
    Convert XML root to string

    Args:
        xml (_Element): Root XML

    Returns:
        str: XML as a string

    """
    return tostring(
        xml, pretty_print=True, xml_declaration=True, encoding=UTF_8
    ).decode(UTF_8)
