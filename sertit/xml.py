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
""" Tools concerning XML management, simplifying lxml.etree """
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union

from cloudpathlib import CloudPath
from lxml.etree import (
    ElementTree,
    XMLSyntaxError,
    _Element,
    fromstring,
    parse,
    tostring,
)
from lxml.html.builder import E

from sertit import files
from sertit.logs import SU_NAME
from sertit.misc import ListEnum

UTF_8 = "UTF-8"

LOGGER = logging.getLogger(SU_NAME)


def read(path: Union[str, Path, CloudPath]) -> _Element:
    """
    Read an XML file, even stored on the cloud

    Args:
        path (Union[str, Path, CloudPath]): Path to the XML file

    Returns:
        _Element: XML Root
    """
    try:
        if isinstance(path, CloudPath):
            try:
                # Try using read_text (faster)
                root = fromstring(path.read_text())
            except ValueError:
                # Try using read_bytes
                # Slower but works with:
                # {ValueError}Unicode strings with encoding declaration are not supported.
                # Please use bytes input or XML fragments without declaration.
                root = fromstring(path.read_bytes())
        else:
            # pylint: disable=I1101:
            # Module 'lxml.etree' has no 'parse' member, but source is unavailable.
            xml_tree = parse(str(path))
            root = xml_tree.getroot()

    except XMLSyntaxError:
        raise ValueError(f"Invalid metadata XML for {path}!")

    return root


def read_archive(path: Union[str, Path, CloudPath], regex: str = None) -> _Element:
    """
    Read an XML file from inside an archive (zip or tar)
    Convenient duplicate of :code:`files.read_archived_xml`

    Manages two cases:
    - complete path to an XML file stored inside an archive. In this case the filetree from inside the archive should be separated with a :code:`!`. Don't need to start with zip or tar
    - path to the archive plus a regex looking inside the archive. Duplicate behaviour to :code:`files.read_archived_xml`

    Args:
        path (Union[str, Path, CloudPath]): Path to the XML file, stored inside an archive or path to the archive itself
        regex (str): Optional. If specified, the path should be the archive path and the regex should be the key to find the XML file inside the archive.

    Returns:
        _Element: XML Root
    """

    try:
        if not regex:
            path, basename = str(path).split("!")
            regex = basename
            if path.startswith("zip://") or path.startswith("tar://"):
                path = path[5:]

        return files.read_archived_xml(path, regex)

    except XMLSyntaxError:
        raise ValueError(f"Invalid metadata XML for {path}!")


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
    Remove in place field from a lxml _Element

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
        value (Callable): Function to apply
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
    Convert any dataset containig the given atgtributes to an XML _Element
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
                str_attr = val.value
            elif isinstance(val, datetime):
                str_attr = val.isoformat()
            else:
                try:
                    # gpd, pd...
                    val = val.iat[0]
                except AttributeError:
                    pass
                str_attr = str(val)
            global_attr.append(E(attr, str_attr))

    xml = E.data(*global_attr)
    xml_el = fromstring(
        tostring(xml, pretty_print=True, xml_declaration=True, encoding=UTF_8)
    )

    return xml_el


def df_to_xml(src_ds: Any) -> _Element:
    """
    Convert a pd.DataFrame or similar (which has a .to_xml() function) to a lxml _Element

    Args:
        src_ds:

    Returns:
        _Element: Wanted XML
    """
    return fromstring(bytes(src_ds.to_xml(index=False), UTF_8))


def to_string(xml: _Element) -> str:
    """
    Convert XMl root to string

    Args:
        xml (_Element): Root XML

    Returns:
        str: XML as a string

    """
    return tostring(
        xml, pretty_print=True, xml_declaration=True, encoding=UTF_8
    ).decode(UTF_8)
