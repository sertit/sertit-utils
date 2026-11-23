# Copyright 2026, SERTIT-ICube - France, https://sertit.unistra.fr/
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
"""ArcGIS related utilities."""

import logging
import logging.handlers
import logging.config
import os
import pickle
import tempfile
from functools import wraps
from typing import Any

try:
    import click
except ModuleNotFoundError as ex:  # pragma: no cover
    raise ModuleNotFoundError(
        "Please install 'click' to use the 'arcpy' package."
    ) from ex

from sertit import AnyPath, utils

# Arcpy types from inside a schema
SHORT = "int32:4"
""" 'Short' type for ArcGis GDB """

LONG = "int32:10"
""" 'Long' type for ArcGis GDB """

FLOAT = "float"
""" 'Float' type for ArcGis GDB """

DOUBLE = "float"
""" 'Double' type for ArcGis GDB """

TEXT = "str:255"
""" "Text" type for ArcGis GDB """

DATE = "datetime"
""" 'Date' type for ArcGis GDB """

BASE_ENV = "arcgispro-eo"
SEP = "-"
AGP_EO_ENV = f"{BASE_ENV}{SEP}backend"
AGP_EO_ENV_TEST = f"{BASE_ENV}{SEP}backend{SEP}testing"
AGP_ENVS = [AGP_EO_ENV, AGP_EO_ENV_TEST]
""" ArcGis Pro environments """


# flake8: noqa
def init_conda_arcpy_env():
    """
    Initialize the conda environment with ArcGis Pro.

    Resolves several known issues (e.g. GDAL version parsing, MEM dataset access).
    """
    try:  # pragma: no cover
        from packaging.version import InvalidVersion, Version

        try:
            import fiona
            from fiona import Env as fiona_env

            with fiona_env():
                gdal_version = fiona.env.get_gdal_release_name()
                Version(gdal_version)
        except InvalidVersion:
            # workaround to https://community.esri.com/t5/arcgis-pro-questions/arcgispro-py39-gdal-version-3-7-0e-is-recognized/m-p/1364021
            import geopandas as gpd

            gpd.options.io_engine = "pyogrio"
    except (ModuleNotFoundError, ImportError):  # pragma: no cover
        pass

    # Solve this missing env variable (removed from env in ArcGis):
    # rasterio._err.CPLE_AppDefinedError:
    # Opening a MEM dataset with the MEM:::DATAPOINTER= syntax is no longer supported by default for security reasons.
    # If you want to allow it, define the GDAL_MEM_ENABLE_OPEN configuration option to YES, or build GDAL with the GDAL_MEM_ENABLE_OPEN compilation definition
    os.environ["GDAL_MEM_ENABLE_OPEN"] = "YES"


class ArcPyLogger:  # pragma: no cover
    def __init__(self, name=None, prefix_log_file="atools_"):
        """
        This class initializes a ready-to-use Python logger for ArcGis Pro tools.
        Make sure that arcpy has been imported before using this class.
        It uses logging under the hood.
        It writes outputs to a temporary file and to the ArcGis console.
        The temporary file is removed when the user closes ArcGis.

        If you need a logger in an outside module or function, use :code:`logging.getLogger(LOGGER_NAME)`
        to get your logger.

        Args:
            name (str): The name of the logger
            prefix_log_file (str): The log filename is random, but you can prefix a name.
                The default value is :code:`atools_`.

        Example:
            >>> import logging
            >>> from sertit.arcpy import ArcPyLogger
            >>> arcpy_logger = ArcPyLogger(name="MyArcgisTool")
            Outputs written to file: C:\\Users\\bcoriat\\AppData\\Local\\Temp\\ArcGISProTemp15788\\MyArcgisTool_1bv0c1cl
            >>> logger = logging.getLogger("MyArcgisTool")
            >>> logger.info("Hello World !")
            Hello World !

        Warning:
            Python must keep a reference to the instantiated object during the execution of your program.
            That's why you must init this class once at the top level of your project.

            This will not work because Python destroys the object class.

            >>> ArcPyLogger(name="MyArcgisTool")
            >>> logger = logging.getLogger("MyArcgisTool")
            >>> logger.info("Hello World !")
        """
        self.name = name
        self.logger = None
        self.handler = None
        if name:
            self.prefix = name + "_"
        else:
            self.prefix = prefix_log_file
        self._set_logger()

    def __del__(self):
        self.logger.removeHandler(self.handler)

    def _set_logger(self):
        import tempfile

        logger = logging.getLogger(self.name)
        f = tempfile.NamedTemporaryFile(prefix=self.prefix, delete=False)

        # Create handler
        max_file_size = 1024 * 1024 * 2  # 2MB log files
        self.handler = ArcPyLogHandler(
            f.name,
            maxBytes=max_file_size,
            backupCount=10,
            encoding="utf-8",
        )
        logger.addHandler(self.handler)

        # Set formatter to handler
        formatter = logging.Formatter("%(levelname)-8s %(message)s")
        self.handler.setFormatter(formatter)

        # Set logger
        logger.setLevel(logging.DEBUG)
        self.logger = logger
        self.logger.info("You can read logs in the file: " + f.name)


class ArcPyLogHandler(logging.handlers.RotatingFileHandler):  # pragma: no cover
    """
    Custom logging class that bounces messages to the arcpy tool window as well
    as reflecting back to the file.
    """

    def emit(self, record):
        """
        Write the log message
        """

        import arcpy

        try:
            msg = record.msg % record.args
        except:
            try:
                msg = record.msg.format(record.args)
            except:
                msg = record.msg

        if record.levelno >= logging.ERROR:
            arcpy.AddError(msg)
        elif record.levelno >= logging.WARNING:
            arcpy.AddWarning(msg)
        elif record.levelno >= logging.INFO:
            arcpy.AddMessage(msg)

        super(ArcPyLogHandler, self).emit(record)


def init_arcgis_logger(
    curr_logger: logging.Logger, log_lvl: int = logging.INFO
) -> None:
    """
    Initialize a logger for tools running in the backend environment with the :code:`run_in_conda_env` function.
    This logger outputs information in JSON format to stdout; the :code:`run_in_conda_env` function
    catches the JSON and prints it properly to the user in ArcGis.

    Args:
        curr_logger (logging.Logger): Logger to be initialized
        log_lvl (int): Logging level to be set

    Example:
        >>> logger = logging.getLogger("logger_test")
        >>> init_arcgis_logger(logger, logging.INFO)
        >>> logger.info("MESSAGE")
    """
    f = tempfile.NamedTemporaryFile(prefix="atools_", delete=False)
    max_file_size = 1024 * 1024 * 2  # 2MB log files

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "fmt": {
                    "format": "%(levelname)-8s %(message)s",
                }
            },
            "handlers": {
                "arcgis": {
                    "()": "sertit.arcpy.ArcPyLogHandler",
                    "filename": f.name,
                    "maxBytes": max_file_size,
                    "backupCount": 10,
                    "encoding": "utf-8",
                    "level": logging.getLevelName(log_lvl),
                    "formatter": "fmt",
                },
            },
            "loggers": {
                curr_logger.name: {
                    "handlers": ["arcgis"],
                    "propagate": False,
                    "level": logging.getLevelName(log_lvl),
                }
            },
        }
    )

    curr_logger.info("You can read logs in the file: " + f.name)


@utils.requires(("pythonjsonlogger", "arcpy"))
def init_json_logger(curr_logger: logging.Logger, log_lvl: int = logging.INFO) -> None:
    """
    Initialize a logger for tools running in the backend environment with the :code:`run_in_conda_env` function.
    This logger outputs information in JSON format to stdout; the :code:`run_in_conda_env` function
    catches the JSON and prints it properly to the user in ArcGis.

    Args:
        curr_logger (logging.Logger): Logger to be initialized
        log_lvl (int): Logging level to be set

    Example:
        >>> logger = logging.getLogger("logger_test")
        >>> init_json_logger(logger, logging.INFO)
        >>> logger.info("MESSAGE")
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "fmt": {
                    "()": "pythonjsonlogger.json.JsonFormatter",
                    "fmt": ["message", "asctime", "exc_info", "levelname"],
                }
            },
            "handlers": {
                "stream": {
                    "level": logging.getLevelName(log_lvl),
                    "class": "logging.StreamHandler",
                    "formatter": "fmt",
                },
            },
            "loggers": {
                curr_logger.name: {
                    "handlers": ["stream"],
                    "propagate": False,
                    "level": logging.getLevelName(log_lvl),
                }
            },
        }
    )


def gp_layer_to_path(layer) -> str:  # pragma: no cover
    """
    Convert a GP layer to its source path.

    A GP layer in ArcGis is a layer in the content panel. The user can choose it in a dropdown menu
    or drag and drop it from the Windows explorer.
    This function retrieves the source path of the layer in both cases.

    Args:
        layer (arcpy layer): Feature layer or raster layer

    Returns:
        str: Path to the feature or raster layer source

    Example:
        For a python toolbox, in the getParameterInfo() method use the GPLayer, GPFeatureLayer or GPRasterLayer datatype.

        For a vector layer use GPFeatureLayer:

        >>> import arcpy
        >>> aoi = arcpy.Parameter(
        ...    displayName="Aoi",
        ...    name="aoi",
        ...    datatype="GPFeatureLayer",
        ...    parameterType="Required",
        ...    direction="Input",
        ... )

        For a raster layer, use GPRasterLayer:

        >>> nir_path = arcpy.Parameter(
        ...    displayName="Nir infrared band",
        ...    name="nir_path",
        ...    datatype="GPRasterLayer",
        ...    parameterType="Optional",
        ...    direction="Input",
        ... )

        If your layer may be a feature or a raster layer, use GPLayer:

        >>> dem_path = arcpy.Parameter(
        ...    displayName="DEM path as isoline or raster",
        ...    name="dem_path",
        ...    datatype="GPLayer",
        ...    parameterType="Optional",
        ...    direction="Input",
        ... )

        Then in the execute() method, you can use this function to retrieve the real path to the layer.

        >>> aoi_path = gp_layer_to_path(parameters[0].value)
        >>> print(aoi_path)
        D:/data/project/aoi/aoi.shp

    """
    # Get path
    if hasattr(layer, "dataSource"):
        path = layer.dataSource
    else:
        path = str(layer)

    # In case we have a raster, convert the path to one compatible with rasterio
    if hasattr(layer, "isRasterLayer") and layer.isRasterLayer:
        path = from_gdb_raster_to_rio_path(path)

    return path


def from_gdb_raster_to_rio_path(path_in_gdb: str) -> str:
    """Convert a raster stored in a gdb into a path readable by rasterio"""
    path_obj = AnyPath(path_in_gdb)
    if ".gdb" in str(path_obj.parent):
        rio_path = rf"OpenFileGDB:{str(path_obj.parent)}:{path_obj.name}"
    else:
        rio_path = path_in_gdb

    return rio_path


def run_in_conda_env(
    executable: list[str],
    logger_name: str = "sertit_utils",
    conda_env_name: str | None = None,
    python_path: str = "",
    extend_env: dict = {},
    shell=True,
):
    """
    This function runs an executable through :code:`conda run` in a Python subprocess.
    If :code:`conda_env_name` is None, it chooses an environment among:
    - arcgispro-eo-backend
    - arcgispro-eo-backend-testing
    - self

    Where :code:`self` is the current environment. If :code:`self` is given, a subprocess is launched anyway with the same Python interpreter.

    This function is designed to solve ArcGis limits by running executables in another conda environment,
    which solves a lot of issues.

    The `extend_env` argument allows one to add environment variable in the child subprocess.
    For example, if parent_env = { "PYTHON_PATH": "/path/to/python_lib,/another_path/to/python_lib", "VAR": "True" }
    and extend_env = {"PYTHON_PATH": "/child_path/to/python_lib,/another_path/to/python_lib"}
    Then, child_env = {"PYTHON_PATH": "/child_path/to/python_lib,/another_path/to/python_lib", "VAR": "True"}

    The child_env took the whole `PYTHON_PATH` variable from `extend_env` and the variable `VAR` from the parent_env.

    Args:
        executable (list[str]): Executable name, with additional arguments to be passed to the executable on invocation.
        logger_name (str): The logger name to use.
        conda_env_name (str | None): Name of the conda environment where to run the executable.
            Set it to "self" to force the subprocess to run in the current environment.
        python_path (str): Value to set for the PYTHONPATH variable in the child subprocess.
        extend_env (dict): Dict to extend the environment in the sub-process. extend_env is merged with
            the parent process environment by overriding it if necessary.
        shell (bool): Run subprocess in a shell if True.

    Returns:
        int: Return value (exit code) of the executable

    """
    import json
    import logging
    import os
    import pathlib
    import subprocess
    import sys
    import platform
    from sertit.exception import ListCondaEnvError

    logger = logging.getLogger(logger_name)

    list_env_cmd = ["conda", "env", "list", "--json"]
    if platform.system() == "Windows":
        CREATE_NO_WINDOW = 0x08000000
        env_list = subprocess.run(
            list_env_cmd,
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
        )
    else:
        env_list = subprocess.run(
            list_env_cmd,
            capture_output=True,
        )
    if env_list.returncode > 0:
        raise ListCondaEnvError(env_list.stderr)

    env_list = json.loads(env_list.stdout)
    current_env = env_list["default_prefix"]
    current_env_name = pathlib.Path(current_env).name
    current_is_testing = f"{SEP}{AGP_EO_ENV_TEST.split(SEP)[-1]}" in current_env_name

    if not conda_env_name:
        available_env = []
        available_prefix = []
        env_path_list = env_list["envs"]

        if env_path_list is None:
            raise FileNotFoundError(f"No env found on your conda installation!")

        # Find available environments
        for env in env_path_list:
            name = pathlib.Path(env).name
            prefix = pathlib.Path(env).parent
            if name in AGP_ENVS:
                available_env.append(name)
                available_prefix.append(str(prefix))

        # Choose the most appropriate environment to run the command line
        if len(available_env) > 0:
            # AGP_EO_ENV_TEST has priority over AGP_EO_ENV if testing is present in "current_env_name"
            if current_is_testing:
                conda_env_name = (
                    AGP_EO_ENV_TEST if AGP_EO_ENV_TEST in available_env else AGP_EO_ENV
                )
            # AGP_EO_ENV has priority over AGP_EO_ENV_TEST if testing is absent in "current_env_name"
            else:
                conda_env_name = (
                    AGP_EO_ENV if AGP_EO_ENV in available_env else AGP_EO_ENV_TEST
                )
            conda_env_prefix = available_prefix[available_env.index(conda_env_name)]
        else:
            # Default to the current environment if no backend environment is found
            conda_env_name = current_env_name
            conda_env_prefix = str(pathlib.Path(current_env).parent)

        # Create conda path
        conda_path = str(pathlib.Path(conda_env_prefix) / conda_env_name)

    elif conda_env_name == "self":
        conda_env_name = current_env_name
        conda_env_prefix = str(pathlib.Path(current_env).parent)
        conda_path = str(pathlib.Path(conda_env_prefix) / conda_env_name)

    else:
        conda_path = conda_env_name

    if conda_path is None:
        raise FileNotFoundError(f"Impossible to find your conda env!")

    cmd_line = [
        "conda",
        "run",
        "--live-stream",
        "-n",
        conda_env_name,
    ] + executable

    # Copy and clean the environment
    env = os.environ
    clean_env = {}
    for key, value in env.items():
        value_as_list = value.split(";")
        value_as_list_filtered = [
            el for el in value_as_list if el.find(current_env_name) == -1
        ]
        if len(value_as_list_filtered) > 0:
            value_as_str = ";".join(value_as_list_filtered)
            clean_env[key] = value_as_str

    # Remove any ArcGis leak in the environment variables on strategic variables
    for key in [
        "GDAL_DATA",
        "GDAL_DRIVER_PATH",
        "GDAL_CONFIG_FILE",
        "PROJ_DATA",
        "PROJ_LIB",
    ]:
        clean_env.pop(key, None)

    # Clean the path (remove any ArcGis Pro leak in the PATH)
    # i.e. C:\Program Files\ArcGIS\Pro\bin\Python\Scripts or C:\Program Files\ArcGIS\Pro\bin
    clean_env["PATH"] = ";".join(
        env_path
        for env_path in clean_env["PATH"].split(";")
        if "ArcGIS" not in env_path
    )

    # Override PYTHONPATH
    clean_env["PYTHONPATH"] = python_path

    # Add env_extend allowing to overwrite
    clean_env = clean_env | extend_env
    clean_env["SERTIT_LOGGER_TYPE"] = "BACKEND_SUBPROCESS"

    # Log things in debug, just in case there are other ArcGis leaks in the subprocess
    logger.info(
        f"['{current_env_name}' will delegate the process run to '{conda_env_name}']"
    )
    logger.debug("Before subprocess: ")
    logger.debug("Command line: ")
    logger.debug(" ".join(cmd_line))
    logger.debug("")
    logger.debug("Env: ")
    logger.debug(clean_env)

    with subprocess.Popen(
        cmd_line,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=clean_env,
        close_fds=False,
    ) as process:
        for line in process.stdout:
            line = line.decode(
                encoding=sys.stdout.encoding,
                errors=("replace" if sys.version_info < (3, 5) else "backslashreplace"),
            ).rstrip()
            try:
                line_json = json.loads(line)
                message = line_json.get("message")
                exc_info = line_json.get("exc_info")
                if exc_info and line_json.get("levelname") == "ERROR":
                    logger.error(exc_info)
                elif line_json.get("levelname") == "ERROR":
                    logger.error(message)
                elif line_json.get("levelname") == "WARNING":
                    logger.warning(message)
                else:
                    logger.info(message)

            except Exception as e:
                logger.info(line)

        # Get return value
        retval = process.wait(timeout=None)

        # Kill process
        process.kill()

        return retval


@click.command(
    context_settings=dict(
        help_option_names=["-h", "--help"], max_content_width=300, show_default=True
    )
)
@click.option(
    "-i",
    "--input-file",
    help="Path to the pickle corresponding to the inputs of your function.",
    type=str,
    required=False,
)
@click.option(
    "-o",
    "--output-file",
    help="Path to the pickle corresponding to the outputs of your function.",
    type=str,
    required=False,
)
def create_conda_env_cli(input_file: str, output_file: str):
    """
    Create a CLI that runs the decorated function in a new conda environment.

    To be used in the backend CLI main.

    Example:
        >>> from sertit.arcpy import create_conda_env_cli
        >>> from my_tool.my_tool_core import my_tool_core
        >>> create_conda_env_cli(standalone_mode=False)(my_tool_core)
    """

    def decorator(function):
        @wraps(function)
        def wrapper(input_file, output_file):
            with open(input_file, "rb") as fp_in:
                inputs: dict = pickle.load(fp_in)

            output = function(**inputs)
            with open(output_file, "wb") as fp_out:
                pickle.dump(output, fp_out)

        return wrapper(input_file, output_file)

    return decorator


def run_in_conda_env_cli(
    cli_path: str,
    inputs: dict,
    logger_name: str,
    tools_path: str,
    conda_env_name: str = "",
    extend_env: dict = {},
    shell: bool = True,
) -> tuple[int, Any]:
    """
    Run the given command line with the given inputs in a new conda environment.

    Scope: run a complicated tool with heavy dependencies from inside a conda environment containing arcpy.

    To be used in the .pyt file.

    Args:
        cli_path (str): Path to CLI file.
        inputs (dict): Inputs of the function wrapped by the CLI as a dictionary
        logger_name (str): Logger name
        tools_path (str): Path to the tools
        conda_env_name (str): Name of the conda environment where to run the executable.
                        Set it to "self" to force the subprocess to run in the current environment.
        extend_env (dict): Dict to extend environment in the sub-process. extend_env is merged with
                    the parent process environment by overriding it if necessary.
        shell (bool): Run subprocess in a shell if True.
    Returns:
        tuple[int, Any]: Return value and outputs of the function

    Example:
        >>> def main_arcgis(parameters, messages, tools_path):
        ...     from sertit.arcpy import run_in_conda_env_cli
        ...     from my_tool import LOGGER_NAME, my_backend_cli
        ...
        ...     inputs = {
        ...         "aoi_path": "aoi.shp",
        ...         "input": "input.tif",
        ...         "resolution": 10,
        ...     }
        ...
        ...     retval, output_path = run_in_conda_env_cli(
        ...         cli_path=my_backend_cli.__file__,
        ...         inputs=inputs,
        ...         logger_name=LOGGER_NAME,
        ...         tools_path=tools_path,
        ...     )
        ...
        ...     if retval == 0:
        ...         logger.info("RasterFillHoles was a success.")
        ...     else:
        ...         logger.error("Subprocess RasterFillHoles failed.")
    """
    with tempfile.NamedTemporaryFile(delete=False) as fp_in:
        fp_in_name = fp_in.name
    with tempfile.NamedTemporaryFile(delete=False) as fp_out:
        fp_out_name = fp_out.name
    with open(fp_in_name, "wb") as fp_in:
        pickle.dump(inputs, fp_in)

    cmd_line = [
        "python",
        str(cli_path),
        "--input-file",
        fp_in.name,
        "--output-file",
        fp_out.name,
    ]
    retval = run_in_conda_env(
        cmd_line,
        conda_env_name=conda_env_name,
        logger_name=logger_name,
        python_path=tools_path,
        extend_env=extend_env,
        shell=shell,
    )
    if retval == 0:
        with open(fp_out_name, "rb") as fp_out:
            ret = pickle.load(fp_out)
    else:
        ret = None

    return retval, ret
