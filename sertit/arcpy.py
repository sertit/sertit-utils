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
