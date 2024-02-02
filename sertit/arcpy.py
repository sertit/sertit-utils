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
    """
    This class init a ready to use python logger (thanks to logging) for ArcGis tool.
    It writes outputs to a temporary file and to the ArcGis console.
    The temporary file is removed when the user closes ArcGis.

    You just have to init this class once. Then, call your logger with `logging.getLogger(LOGGER_NAME)`
    where LOGGER_NAME is the name of your logger.
    """
    def __init__(self, name=None, prefix_log_file="atools_"):
        self.name = name
        self.logger = None
        self.handler = None
        self.prefix = prefix_log_file
        self._set_logger()

    def __del__(self):
        self.logger.removeHandler(self.handler)

    def _set_logger(self):

        import tempfile

        logger = logging.getLogger(self.name)
        f = tempfile.NamedTemporaryFile(prefix=self.prefix, delete=False)

        self.handler = ArcPyLogHandler(
            f.name, maxBytes=1024 * 1024 * 2, backupCount=10  # 2MB log files
        )
        logger.addHandler(self.handler)

        formatter = logging.Formatter("%(levelname)-8s %(message)s")
        self.handler.setFormatter(formatter)
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
