import os
import shutil

import pytest
from lxml import etree, html

from CI.SCRIPTS.script_utils import files_path, s3_env
from sertit import archives, ci, files, path, s3, vectors


def test_archive(tmp_path):
    """Test extracting functions"""
    # Archives
    zip_file = files_path().joinpath("test_zip.zip")
    zip2_file = files_path().joinpath("test_zip.zip")  # For overwrite
    zip_without_directory = files_path().joinpath("test_zip_without_directory.zip")
    tar_file = files_path().joinpath("test_tar.tar")
    tar_gz_file = files_path().joinpath("test_targz.tar.gz")

    # Core dir
    core_dir = files_path().joinpath("core")
    folder = core_dir
    arch = [
        zip_file,
        tar_file,
        tar_gz_file,
        folder,
        zip2_file,
        zip_without_directory,
    ]

    # Extract
    extracted_dirs = archives.extract_files(arch, tmp_path, overwrite=True)
    archives.extract_files([zip2_file], tmp_path, overwrite=False)  # Already existing

    # Test
    for ex_dir in extracted_dirs:
        ci.assert_dir_equal(core_dir, ex_dir)

    # Archive
    archive_base = os.path.join(tmp_path, "archive")
    for fmt in ["zip", "tar", "gztar"]:
        archive_fn = archives.archive(
            folder_path=core_dir, archive_path=archive_base, fmt=fmt
        )
        out = archives.extract_file(archive_fn, tmp_path)
        # an additional folder is created
        out_dir = path.listdir_abspath(out)[0]
        ci.assert_dir_equal(core_dir, out_dir)

        # Remove out directory in order to avoid any interferences
        files.remove(out)

    # Add to zip
    zip_out = zip2_file if path.is_cloud_path(zip2_file) else archive_base + ".zip"
    core_copy = files.copy(core_dir, os.path.join(tmp_path, "core2"))
    zip_out = archives.add_to_zip(zip_out, core_copy)

    # Extract
    unzip_out = os.path.join(tmp_path, "out")
    unzip_out = archives.extract_file(zip_out, unzip_out)

    # Test
    unzip_dirs = path.listdir_abspath(unzip_out)

    assert len(unzip_dirs) == 2
    ci.assert_dir_equal(unzip_dirs[0], unzip_dirs[1])


@s3_env
def test_archived_files(tmp_path):
    landsat_name = "LM05_L1TP_200030_20121230_20200820_02_T2_CI"
    ok_folder = files_path().joinpath(landsat_name)
    zip_file = files_path().joinpath(f"{landsat_name}.zip")
    tar_file = files_path().joinpath(f"{landsat_name}.tar")
    targz_file = files_path().joinpath(f"{landsat_name}.tar.gz")
    sz_file = files_path().joinpath(f"{landsat_name}.7z")

    # VECTORS
    vect_name = "map-overlay.kml"
    vec_ok_path = ok_folder.joinpath(vect_name)
    if shutil.which("ogr2ogr"):  # Only works if ogr2ogr can be found.
        vect_regex = f".*{vect_name}"
        vect_zip = vectors.read(zip_file, archive_regex=vect_regex)
        vect_tar = vectors.read(tar_file, archive_regex=r".*overlay\.kml")
        vect_ok = vectors.read(vec_ok_path)
        assert not vect_ok.empty
        ci.assert_geom_equal(vect_ok, vect_zip)
        ci.assert_geom_equal(vect_ok, vect_tar)

    # XML
    xml_name = "LM05_L1TP_200030_20121230_20200820_02_T2_MTL.xml"
    xml_ok_path = ok_folder.joinpath(xml_name)
    xml_ok_path = str(s3.download(xml_ok_path, tmp_path))

    xml_regex = f".*{xml_name}"
    xml_zip = archives.read_archived_xml(zip_file, xml_regex)
    xml_tar = archives.read_archived_xml(tar_file, r".*_MTL\.xml")
    xml_ok = etree.parse(xml_ok_path).getroot()
    ci.assert_xml_equal(xml_ok, xml_zip)
    ci.assert_xml_equal(xml_ok, xml_tar)

    # FILE + HTML
    html_zip_file = files_path().joinpath("productPreview.zip")
    html_tar_file = files_path().joinpath("productPreview.tar")
    html_name = "productPreview.html"
    html_ok_path = files_path().joinpath(html_name)
    html_ok_path = str(s3.download(html_ok_path, tmp_path))

    html_regex = f".*{html_name}"

    # FILE
    file_zip = archives.read_archived_file(html_zip_file, html_regex)
    file_tar = archives.read_archived_file(html_tar_file, html_regex)
    html_ok = html.parse(html_ok_path).getroot()
    ci.assert_html_equal(html_ok, html.fromstring(file_zip))
    ci.assert_html_equal(html_ok, html.fromstring(file_tar))

    file_list = archives.get_archived_file_list(html_zip_file)
    ci.assert_html_equal(
        html_ok,
        html.fromstring(
            archives.read_archived_file(html_zip_file, html_regex, file_list=file_list)
        ),
    )

    # HTML
    html_zip = archives.read_archived_html(html_zip_file, html_regex)
    html_tar = archives.read_archived_html(html_tar_file, html_regex)
    ci.assert_html_equal(html_ok, html_zip)
    ci.assert_html_equal(html_ok, html_tar)
    ci.assert_html_equal(
        html_ok,
        archives.read_archived_html(
            html_tar_file,
            html_regex,
            file_list=archives.get_archived_file_list(html_tar_file),
        ),
    )

    # ERRORS
    with pytest.raises(TypeError):
        archives.read_archived_file(targz_file, xml_regex)
    with pytest.raises(TypeError):
        archives.read_archived_file(sz_file, xml_regex)
    with pytest.raises(FileNotFoundError):
        archives.read_archived_file(zip_file, "cdzeferf")
