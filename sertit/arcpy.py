import logging
import logging.handlers

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


# flake8: noqa
def init_conda_arcpy_env():
    """
    Initialize conda environment with Arcgis Pro.

    Resolves several issues.
    """
    try:
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
    except ModuleNotFoundError:
        pass


class ArcPyLogger:
    def __init__(self, name=None, prefix_log_file="atools_"):
        """
        This class init a ready to use python logger for ArcGis pro tool. Be sure that arcpy has been imported
        before using this class. It uses logging under the hood.
        It writes outputs to a temporary file and to the ArcGis console.
        The temporary file is removed when the user closes ArcGis.

        If you need a logger in an outside module or function, use `logging.getLogger(LOGGER_NAME)`
        to get your logger.

        Args:
            name (str) : The name of the logger
            prefix_log_file (str) : The log filename is random, but you can prefix a name.
                The default value is "{{ name }}_".

        Example:

        >>> import sertit, arcpy
        >>> from sertit import arcpy
        >>> arcpy_logger = arcpy.ArcPyLogger(name="MyArcgisTool")
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
        self.logger.info("Outputs written to file: " + f.name)


class ArcPyLogHandler(logging.handlers.RotatingFileHandler):
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


def feature_layer_to_path(feature_layer: str) -> str:
    """
    Convert a feature layer to its source path.

    Args:
        feature_layer (str): Feature layer

    Returns:
        str: Path to the feature layer source

    """
    # Get path
    if hasattr(feature_layer, "dataSource"):
        path = feature_layer.dataSource
    else:
        path = str(feature_layer)

    return path
