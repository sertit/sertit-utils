URLS=[
"sertit/index.html",
"sertit/strings.html",
"sertit/network.html",
"sertit/vectors.html",
"sertit/files.html",
"sertit/misc.html",
"sertit/logs.html",
"sertit/rasters.html",
"sertit/rasters_rio.html",
"sertit/display.html",
"sertit/ci.html",
"sertit/snap.html"
];
INDEX=[
{
"ref":"sertit",
"url":0,
"doc":" Source Code : https: github.com/sertit/sertit-utils [![pypi](https: img.shields.io/pypi/v/sertit.svg)](https: pypi.python.org/pypi/sertit) [![Conda](https: img.shields.io/conda/vn/conda-forge/sertit.svg)](https: anaconda.org/conda-forge/sertit) [![Tests](https: github.com/sertit/sertit-utils/actions/workflows/test.yml/badge.svg)](https: github.com/sertit/sertit-utils/actions/workflows/test.yml) [![pre-commit](https: img.shields.io/badge/pre commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https: github.com/pre-commit/pre-commit) [![black](https: img.shields.io/badge/code%20style-black-000000.svg)](https: github.com/python/black) [![Apache](https: img.shields.io/badge/License-Apache%202.0-blue.svg)](https: github.com/sertit/eoreader/blob/master/LICENSE) Library gathering functions for all SERTIT's projects. Find the API documentation [ here ](https: sertit.github.io/sertit-utils/).  Installing  Pip For installing this library to your environment, please type this:  pip install sertit[full]  [full] will allow you to use the whole library, but you will need to install also rioxarray and  geopandas (with GDAL installation issues on Windows, so please install them from wheels that you can find [here](https: www.lfd.uci.edu/~gohlke/pythonlibs/ rasterio . However, if you do not need everything, you can type instead: -  nothing , and you won't need  rasterio ,  rioxarray :  pip install sertit  extra-index-url  . -  [rasters] , and you won't need  rioxarray :  pip install sertit[rasters]  extra-index-url  . -  [rasters_rio] :  pip install sertit[rasters_rio]  extra-index-url  . -  [colorlog] :  pip install sertit[colorlog]  extra-index-url  . to have  colorlog installed  Conda You can install it via conda (but you will automatically have the full version):  conda config  env  set channel_priority strict  conda install -c conda-forge sertit  What is in it ?  Files File gathering file-related functions: - paths - Create archive - Add folder to zip file - file extraction - file name - copy/remove - find files - JSON/pickles - hash  Logs - Init simple logger - Create complex logger (file and stream + color) - Shutdown logger  Misc - Function on lists: convert a list to a dict, remove empty values . - Function on dicts: nested set, check mandatory keys, find by key - Run a command line - Get a function name - Test if in docker - Change current directory ( cd ) as a context manager  Strings - Conversion from string to bool, logging level, list, list of dates . - Convert the string to be usable in command line - Case conversion ( snake_case to/from  CamelCase )  Vectors - Load an AOI as WKT - Get UTM projection from lat/lon - Manage bounds and polygons - Get  geopandas.Geodataframe from polygon and CRS  Rasters and rasters_rio Basically, these functions are overloads of rasterio's functions: - Get extent and footprint of a raster - Read/write overload of rasterio functions - Masking and cropping with masked array - Collocation (superimpose) - Sieving - Vectorization and get nodata vector - Merge rasters (as GTiff and VRT) - Get the path of the BEAM-DIMAP image that can be read by rasterio - Manage bit arrays The main difference between the two is that  rasters outputs one  xarray variable when  rasters_rio outputs  numpy.ma.masked_arrays +  dict for the raster array and its metadata.  Network - Standard [Exponential Backoff](https: en.wikipedia.org/wiki/Exponential_backoff) algorithm  SNAP - Function converting bytes to SNAP understanding - Function creating a SNAP optimized commande line  Documentation An HTML documentation is provided to document the code. It can be found: - online ([here](https: sertit.github.io/sertit-utils/ , - on git, in  docs . To consult it, just open the  index.html file in a web browser (you need first to clone this project) To generate the HTML documentation, just type  pdoc sertit -o docs\\html -f  html -c sort_identifiers=False "
},
{
"ref":"sertit.strings",
"url":1,
"doc":"Tools concerning strings"
},
{
"ref":"sertit.strings.str_to_bool",
"url":1,
"doc":"Convert a string to a bool. Accepted values (compared in lower case): -  True   yes ,  true ,  t ,  1 -  False   no ,  false ,  f ,  0   >>> str_to_bool(\"yes\")  True  Works with \"yes\", \"true\", \"t\", \"y\", \"1\" (accepted with any letter case) True >>> str_to_bool(\"no\")  False  Works with \"no\", \"false\", \"f\", \"n\", \"0\" (accepted with any letter case) True   Args: bool_str: Bool as a string Returns: bool: Boolean value",
"func":1
},
{
"ref":"sertit.strings.str_to_verbosity",
"url":1,
"doc":"Return a logging level from a string (compared in lower case). -  DEBUG  { debug ,  d ,  10 } -  INFO  { info ,  i ,  20 } -  WARNING  { warning ,  w ,  warn } -  ERROR  { error ,  e ,  err }   >>> str_to_bool(\"d\")  logging.DEBUG  Works with 'debug', 'd', 10 (accepted with any letter case) True >>> str_to_bool(\"i\")  logging.INFO  Works with 'info', 'i', 20 (accepted with any letter case) True >>> str_to_bool(\"w\")  logging.WARNING  Works with 'warning', 'w', 'warn', 30 (accepted with any letter case) True >>> str_to_bool(\"e\")  logging.ERROR  Works with 'error', 'e', 'err', 40 (accepted with any letter case) True   Args: verbosity_str (str): String to be converted Returns: logging level: Logging level (INFO, DEBUG, WARNING, ERROR)",
"func":1
},
{
"ref":"sertit.strings.str_to_list",
"url":1,
"doc":"Convert str to list with  , ,  ; ,   separators.   >>> str_to_list(\"A, B; C D\") [\"A\", \"B\", \"C\", \"D\"]   Args: list_str (Union[str, list]): List as a string additional_separator (str): Additional separators. Base ones are  , ,  ; ,   . case (str): {none, 'lower', 'upper'} Returns: list: A list from split string",
"func":1
},
{
"ref":"sertit.strings.str_to_date",
"url":1,
"doc":"Convert string to a  datetime.datetime . Also accepted date formats: - \"now\": datetime.today() - Usual JSON date format: '%Y-%m-%d' - Already formatted datetimes and dates    Default date format (isoformat) >>> str_to_date(\"2020-05-05T08:05:15\") datetime(2020, 5, 5, 8, 5, 15)  This usual JSON format is also accepted >>> str_to_date(\"2019-08-06\") datetime(2019, 8, 6)  User date's format >>> str_to_date(\"20200909105055\", date_format=\"%Y%m%d%H%M%S\") datetime(2020, 9, 9, 10, 50, 55)   Args: date_str (str): Date as a string date_format (str): Format of the date (as ingested by strptime) Returns: datetime.datetime: A date as a python datetime object",
"func":1
},
{
"ref":"sertit.strings.str_to_list_of_dates",
"url":1,
"doc":"Convert a string containing a list of dates to a list of  datetime.datetime . Also accepted date formats: - \"now\": datetime.today() - Usual JSON date format: '%Y-%m-%d' - Already formatted datetimes and dates   >>>  Default date format (isoformat) >>> str_to_list_of_dates(\"20200909105055, 2019-08-06;19560702121212 2020-08-09\", >>> date_format=\"%Y%m%d%H%M%S\", >>> additional_separator=\" \") [datetime(2020, 9, 9, 10, 50, 55), datetime(2019, 8, 6), datetime(1956, 7, 2, 12, 12, 12), datetime(2020, 8, 9)]   Args: date_str (Union[list, str]): Date as a string date_format (str): Format of the date (as ingested by strptime) additional_separator (str): Additional separator Returns: list: A list containing datetimes objects",
"func":1
},
{
"ref":"sertit.strings.to_cmd_string",
"url":1,
"doc":"Add quotes around the string in order to make the command understand it's a string (useful with tricky symbols like & or white spaces):   >>>  This str wont work in the terminal without quotes (because of the &) >>> pb_str = r\"D:\\Minab_4-DA&VHR\\Minab_4-DA&VHR.shp\" >>> to_cmd_string(pb_str)  D:\\Minab_4-DA&VHR\\Minab_4-DA&VHR.shp   Args: unquoted_str (str): String to update Returns: str: Quoted string",
"func":1
},
{
"ref":"sertit.strings.snake_to_camel_case",
"url":1,
"doc":"Convert a  snake_case string to  CamelCase .   >>> snake_to_camel_case(\"snake_case\") \"SnakeCase\"   Args: snake_str (str): String formatted in snake_case Returns: str: String formatted in CamelCase",
"func":1
},
{
"ref":"sertit.strings.camel_to_snake_case",
"url":1,
"doc":"Convert a  CamelCase string to  snake_case .   >>> camel_to_snake_case(\"CamelCase\") \"camel_case\"   Args: snake_str (str): String formatted in CamelCase Returns: str: String formatted in snake_case",
"func":1
},
{
"ref":"sertit.network",
"url":2,
"doc":"Network control utils"
},
{
"ref":"sertit.network.exponential_backoff",
"url":2,
"doc":"Implementation of the standard Exponential Backoff algorithm (https: en.wikipedia.org/wiki/Exponential_backoff) This algorithm is useful in networking setups where one has to use a potentially unreliable resource over the network, where by unreliable we mean not always available. Every major service provided over the network can't guarantee 100% availability all the time. Therefore if a service fails one should retry a certain number of time and a maximum amount of times. This algorithm is designed to try using the services multiple times with exponentially increasing delays between the tries. The time delays are chosen at random to ensure better theoretical behaviour. No default value is provided on purpose Args: network_request (Callable): Python function taking no arguments which represents a network request which may fail. If it fails it must raise an exception. wait_time_slot (float): Smallest amount of time to wait between retries. Recommended range [0.1 - 10.0] seconds increase_factor (float): Exponent of the expected exponential increase in waiting time. In other words on average by how much should the delay in time increase. Recommended range [1.5 - 3.0] max_wait (float): Maximum of time one will wait for the network request to perform successfully. If the maximum amount of time is reached a timeout exception is thrown. max_retries (int): Number of total tries to perform the network request. Must be at least 2 and maximum  EXP_BACK_OFF_ABS_MAX_RETRIES (or 100 if the environment value is not defined). Recommended range [5 - 25]. If the value exceeds  EXP_BACK_OFF_ABS_MAX_RETRIES (or 100 if the environment value is not defined). The value will be set to  EXP_BACK_OFF_ABS_MAX_RETRIES (or 100 if the environment value is not defined). desc (str): Description of the network request being attempted random_state (int): Seed to the random number generator (optional)",
"func":1
},
{
"ref":"sertit.vectors",
"url":3,
"doc":"Vectors tools You can use this only if you have installed sertit[full] or sertit[vectors]"
},
{
"ref":"sertit.vectors.corresponding_utm_projection",
"url":3,
"doc":"Find the EPSG code of the UTM projection from a lon/lat in WGS84.   >>> corresponding_utm_projection(lon=7.8, lat=48.6)  Strasbourg 'EPSG:32632'   Args: lon (float): Longitude (WGS84) lat (float): Latitude (WGS84) Returns: str: EPSG string",
"func":1
},
{
"ref":"sertit.vectors.from_polygon_to_bounds",
"url":3,
"doc":"Convert a  shapely.polygon to its bounds, sorted as  left, bottom, right, top .   >>> poly = Polygon (0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0. ) >>> from_polygon_to_bounds(poly) (0.0, 0.0, 1.0, 1.0)   Args: polygon (MultiPolygon): polygon to convert Returns: (float, float, float, float): left, bottom, right, top",
"func":1
},
{
"ref":"sertit.vectors.from_bounds_to_polygon",
"url":3,
"doc":"Convert the bounds to a  shapely.polygon .   >>> poly = from_bounds_to_polygon(0.0, 0.0, 1.0, 1.0) >>> print(poly) 'POLYGON  1 0, 1 1, 0 1, 0 0, 1 0 '   Args: left (float): Left coordinates bottom (float): Bottom coordinates right (float): Right coordinates top (float): Top coordinates Returns: Polygon: Polygon corresponding to the bounds",
"func":1
},
{
"ref":"sertit.vectors.get_geodf",
"url":3,
"doc":"Get a GeoDataFrame from a geometry and a crs   >>> poly = Polygon (0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0. ) >>> geodf = get_geodf(poly, crs=WGS84) >>> print(geodf) geometry 0 POLYGON  0.00000 0.00000, 0.00000 1.00000, 1    Args: geometry (Union[Polygon, list]): List of Polygons, or Polygon or bounds crs (str): CRS of the polygon Returns: gpd.GeoDataFrame: Geometry as a geodataframe",
"func":1
},
{
"ref":"sertit.vectors.set_kml_driver",
"url":3,
"doc":"Set KML driver for Fiona data (use it at your own risks !)   >>> path = \"path\\to\\kml.kml\" >>> gpd.read_file(path) fiona.errors.DriverError: unsupported driver: 'LIBKML' >>> set_kml_driver() >>> gpd.read_file(path) Name  . geometry 0 CC679_new_AOI2_3  . POLYGON Z  45.03532 32.49765 0.00000, 46.1947 . [1 rows x 12 columns]  ",
"func":1
},
{
"ref":"sertit.vectors.get_aoi_wkt",
"url":3,
"doc":"Get AOI formatted as a WKT from files that can be read by Fiona (like shapefiles,  .) or directly from a WKT file. The use of KML has been forced (use it at your own risks !). See: https: fiona.readthedocs.io/en/latest/fiona.html fiona.open It is assessed that: - only  one polygon composes the AOI (as only the first one is read) - it should be specified in lat/lon (WGS84) if a WKT file is provided   >>> path = \"path\\to\\vec.geojson\"  OK with ESRI Shapefile, geojson, WKT, KML . >>> get_aoi_wkt(path) 'POLYGON Z  46.1947755465253067 32.4973553439109324 0.0000000000000000, 45.0353174370802520 32.4976496856158974 0.0000000000000000, 45.0355748149750283 34.1139970085580018 0.0000000000000000, 46.1956059695554089 34.1144793800670882 0.0000000000000000, 46.1947755465253067 32.4973553439109324 0.0000000000000000 '   Args: aoi_path (Union[str, CloudPath, Path]): Absolute or relative path to an AOI. Its format should be WKT or any format read by Fiona, like shapefiles. as_str (bool): If True, return WKT as a str, otherwise as a shapely geometry Returns: Union[str, Polygon]: AOI formatted as a WKT stored in lat/lon",
"func":1
},
{
"ref":"sertit.vectors.get_wider_exterior",
"url":3,
"doc":"Get the wider exterior of a MultiPolygon as a Polygon Args: vector (vector: gpd.GeoDataFrame): Polygon to simplify Returns: vector: gpd.GeoDataFrame: Wider exterior",
"func":1
},
{
"ref":"sertit.vectors.shapes_to_gdf",
"url":3,
"doc":"TODO",
"func":1
},
{
"ref":"sertit.vectors.read",
"url":3,
"doc":"Read any vector: - if KML: sets correctly the drivers and open layered KML (you may need  ogr2ogr to make it work !) - if archive (only zip or tar), use a regex to look for the vector inside the archive. You can use this [site](https: regexr.com/) to build your regex. - if GML: manages the empty errors   >>>  Usual >>> path = 'D:\\path\\to\\vector.geojson' >>> vectors.read(path, crs=WGS84) Name  . geometry 0 Sentinel-1 Image Overlay  . POLYGON  0.85336 42.24660, -2.32032 42.65493, . >>>  Archive >>> arch_path = 'D:\\path\\to\\zip.zip' >>> vectors.read(arch_path, archive_regex=\". map-overlay\\.kml\") Name  . geometry 0 Sentinel-1 Image Overlay  . POLYGON  0.85336 42.24660, -2.32032 42.65493, .   Args: path (Union[str, CloudPath, Path]): Path to vector to read. In case of archive, path to the archive. crs: Wanted CRS of the vector. If None, using naive or origin CRS. archive_regex (str): [Archive only] Regex for the wanted vector inside the archive Returns: gpd.GeoDataFrame: Read vector as a GeoDataFrame",
"func":1
},
{
"ref":"sertit.files",
"url":4,
"doc":"Tools for paths and files"
},
{
"ref":"sertit.files.get_root_path",
"url":4,
"doc":"Get the root path of the current disk: - On Linux this returns  / - On Windows this returns  C:\\ or whatever the current drive is   >>> get_root_path() \"/\" on Linux \"C:\\\" on Windows (if you run this code from the C: drive)  ",
"func":1
},
{
"ref":"sertit.files.listdir_abspath",
"url":4,
"doc":"Get absolute path of all files in the given directory. It is the same function than  os.listdir but returning absolute paths.   >>> folder = \".\" >>> listdir_abspath(folder) ['D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\files.py', 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\logs.py', 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\misc.py', 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\network.py', 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\rasters_rio.py', 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\strings.py', 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\vectors.py', 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\version.py', 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\__init__.py']   Args: directory (Union[str, CloudPath, Path]): Relative or absolute path to the directory to be scanned Returns: str: Absolute path of all files in the given directory",
"func":1
},
{
"ref":"sertit.files.to_abspath",
"url":4,
"doc":"Return the absolute path of the specified path and check if it exists If not: - If it is a file (aka has an extension), it raises an exception - If it is a folder, it creates it To be used with argparse to retrieve the absolute path of a file, like:   >>> parser = argparse.ArgumentParser() >>>  Add config file path key >>> parser.add_argument(\" config\", help=\"Config file path (absolute or relative)\", type=to_abspath)   Args: path (Union[str, CloudPath, Path]): Path as a string (relative or absolute) create (bool): Create directory if not existing Returns: Union[CloudPath, Path]: Absolute path",
"func":1
},
{
"ref":"sertit.files.real_rel_path",
"url":4,
"doc":"Gives the real relative path from a starting folder. (and not just adding   \\  between the start and the target)   >>> path = r'D:\\_SERTIT_UTILS\\sertit-utils\\sertit' >>> start = os.path.join(\".\", \" \", \" \") >>> real_rel_path(path, start) 'sertit-utils\\sertit'   Args: path (Union[str, CloudPath, Path]): Path to make relative start (Union[str, CloudPath, Path]): Start, the path being relative from this folder. Returns: Relative path",
"func":1
},
{
"ref":"sertit.files.extract_file",
"url":4,
"doc":"Extract an archived file (zip or others). Overwrites if specified. For zipfiles, in case of multiple folders archived, pay attention that what is returned is the first folder.   >>> file_path = 'D:\\path\\to\\zip.zip' >>> output = 'D:\\path\\to\\output' >>> extract_file(file_path, output, overwrite=True) D:\\path\\to\\output\\zip'   Args: file_path (str): Archive file path output (str): Output where to put the extracted file overwrite (bool): Overwrite found extracted files Returns: Union[list, CloudPath, Path]: Extracted file paths (as str if only one)",
"func":1
},
{
"ref":"sertit.files.extract_files",
"url":4,
"doc":"Extract all archived files. Overwrites if specified.   >>> file_path = ['D:\\path\\to\\zip1.zip', 'D:\\path\\to\\zip2.zip'] >>> output = 'D:\\path\\to\\output' >>> extract_files(file_path, output, overwrite=True) ['D:\\path\\to\\output\\zip1', 'D:\\path\\to\\output\\zip2']   Args: archives (list of str): List of archives to be extracted output (str): Output folder where extracted files will be written overwrite (bool): Overwrite found extracted files Returns: list: Extracted files (even pre-existing ones)",
"func":1
},
{
"ref":"sertit.files.get_archived_file_list",
"url":4,
"doc":"Get the list of all the files contained in an archive.   >>> arch_path = 'D:\\path\\to\\zip.zip' >>> get_archived_file_list(arch_path, file_regex) ['file_1.txt', 'file_2.tif', 'file_3.xml', 'file_4.geojson']   Args: archive_path (Union[str, CloudPath, Path]): Archive path Returns: list: All files contained in the given archive",
"func":1
},
{
"ref":"sertit.files.get_archived_rio_path",
"url":4,
"doc":"Get archived file path from inside the archive, to be read with rasterio: -  zip+file: {zip_path}!{file_name} -  tar+file: {tar_path}!{file_name} See [here](https: rasterio.readthedocs.io/en/latest/topics/datasets.html?highlight=zip dataset-identifiers) for more information.  WARNING It wont be readable by pandas, geopandas or xmltree !  WARNING If  as_list is  False , it will only return the first file matched ! You can use this [site](https: regexr.com/) to build your regex.   >>> arch_path = 'D:\\path\\to\\zip.zip' >>> file_regex = '. dir. file_name'  Use . for any character >>> path = get_archived_tif_path(arch_path, file_regex) 'zip+file: D:\\path\\to\\output\\zip!dir/filename.tif' >>> rasterio.open(path)    Args: archive_path (Union[str, CloudPath, Path]): Archive path file_regex (str): File regex (used by re) as it can be found in the getmembers() list as_list (bool): If true, returns a list (including all found files). If false, returns only the first match Returns: Union[list, str]: Band path that can be read by rasterio",
"func":1
},
{
"ref":"sertit.files.read_archived_xml",
"url":4,
"doc":"Read archived XML from  zip or  tar archives. You can use this [site](https: regexr.com/) to build your regex.   >>> arch_path = 'D:\\path\\to\\zip.zip' >>> file_regex = '. dir. file_name'  Use . for any character >>> read_archived_xml(arch_path, file_regex)    Args: archive_path (Union[str, CloudPath, Path]): Archive path xml_regex (str): XML regex (used by re) as it can be found in the getmembers() list Returns: etree._Element: XML file",
"func":1
},
{
"ref":"sertit.files.archive",
"url":4,
"doc":"Archives a folder recursively.   >>> folder_path = 'D:\\path\\to\\folder_to_archive' >>> archive_path = 'D:\\path\\to\\output' >>> archive = archive(folder_path, archive_path, fmt=\"gztar\") 'D:\\path\\to\\output\\folder_to_archive.tar.gz'   Args: folder_path (Union[str, CloudPath, Path]): Folder to archive archive_path (Union[str, CloudPath, Path]): Archive path, with or without extension fmt (str): Format of the archive, used by  shutil.make_archive . Choose between [zip, tar, gztar, bztar, xztar] Returns: str: Archive filename",
"func":1
},
{
"ref":"sertit.files.add_to_zip",
"url":4,
"doc":"Add folders to an already existing zip file (recursively).   >>> zip_path = 'D:\\path\\to\\zip.zip' >>> dirs_to_add = ['D:\\path\\to\\dir1', 'D:\\path\\to\\dir2'] >>> add_to_zip(zip_path, dirs_to_add) >>>  zip.zip contains 2 more folders, dir1 and dir2   Args: zip_path (Union[str, CloudPath, Path]): Already existing zip file dirs_to_add (Union[list, str]): Directories to add",
"func":1
},
{
"ref":"sertit.files.get_filename",
"url":4,
"doc":"Get file name (without extension) from file path, ie:   >>> file_path = 'D:\\path\\to\\filename.zip' >>> get_file_name(file_path) 'filename'   Args: file_path (Union[str, CloudPath, Path]): Absolute or relative file path (the file doesn't need to exist) Returns: str: File name (without extension)",
"func":1
},
{
"ref":"sertit.files.remove",
"url":4,
"doc":"Deletes a file or a directory (recursively) using  shutil.rmtree or  os.remove .   >>> path_to_remove = 'D:\\path\\to\\remove'  Could also be a file >>> remove(path_to_remove) path_to_remove deleted   Args: path (Union[str, CloudPath, Path]): Path to be removed",
"func":1
},
{
"ref":"sertit.files.remove_by_pattern",
"url":4,
"doc":"Remove files corresponding to a pattern from a directory.   >>> directory = 'D:\\path\\to\\folder' >>> os.listdir(directory) [\"huhu.exe\", \"blabla.geojson\", \"haha.txt\", \"blabla\"] >>> remove(directory, \"blabla \") >>> os.listdir(directory) [\"huhu.exe\", \"haha.txt\"]  Removes also directories >>> remove(directory, \" \", extension=\"txt\") >>> os.listdir(directory) [\"huhu.exe\"]   Args: directory (Union[str, CloudPath, Path]): Directory where to find the files name_with_wildcard (str): Filename (wildcards accepted) extension (str): Extension wanted, optional. With or without point. (yaml or .yaml accepted)",
"func":1
},
{
"ref":"sertit.files.copy",
"url":4,
"doc":"Copy a file or a directory (recursively) with  copytree or  copy2 .   >>> src = 'D:\\path\\to\\copy' >>> dst = 'D:\\path\\to\\output' >>> copy(src, dst) copydir 'D:\\path\\to\\output\\copy' >>> src = 'D:\\path\\to\\copy.txt' >>> dst = 'D:\\path\\to\\output\\huhu.txt' >>> copyfile = copy(src, dst) 'D:\\path\\to\\output\\huhu.txt' but with the content of copy.txt   Args: src (Union[str, CloudPath, Path]): Source Path dst (Union[str, CloudPath, Path]): Destination Path (file or folder) Returns: Union[CloudPath, Path]: New path",
"func":1
},
{
"ref":"sertit.files.find_files",
"url":4,
"doc":"Returns matching files recursively from a list of root paths. Regex are allowed (using glob)   >>> root_path = 'D:\\root' >>> dir1_path = 'D:\\root\\dir1' >>> dir2_path = 'D:\\root\\dir2' >>> os.listdir(dir1_path) [\"haha.txt\", \"huhu.txt\", \"hoho.txt\"] >>> os.listdir(dir2_path) [\"huhu.txt\", \"hehe.txt\"] >>> find_files(\"huhu.txt\", root_path) ['D:\\root\\dir1\\huhu.txt', 'D:\\root\\dir2\\huhu.txt'] >>> find_files(\"huhu.txt\", root_path, max_nof_files=1) ['D:\\root\\dir1\\huhu.txt'] >>> find_files(\"huhu.txt\", root_path, max_nof_files=1, get_as_str=True) found = 'D:\\root\\dir1\\huhu.txt'   Args: names (Union[list, str]): File names. root_paths (Union[list, str]): Root paths max_nof_files (int): Maximum number of files (set to -1 for unlimited) get_as_str (bool): if only one file is found, it can be retrieved as a string instead of a list Returns: list: File name",
"func":1
},
{
"ref":"sertit.files.CustomDecoder",
"url":4,
"doc":"Decoder for JSON with methods for datetimes  object_hook , if specified, will be called with the result of every JSON object decoded and its return value will be used in place of the given  dict . This can be used to provide custom deserializations (e.g. to support JSON-RPC class hinting).  object_pairs_hook , if specified will be called with the result of every JSON object decoded with an ordered list of pairs. The return value of  object_pairs_hook will be used instead of the  dict . This feature can be used to implement custom decoders. If  object_hook is also defined, the  object_pairs_hook takes priority.  parse_float , if specified, will be called with the string of every JSON float to be decoded. By default this is equivalent to float(num_str). This can be used to use another datatype or parser for JSON floats (e.g. decimal.Decimal).  parse_int , if specified, will be called with the string of every JSON int to be decoded. By default this is equivalent to int(num_str). This can be used to use another datatype or parser for JSON integers (e.g. float).  parse_constant , if specified, will be called with one of the following strings: -Infinity, Infinity, NaN. This can be used to raise an exception if invalid JSON numbers are encountered. If  strict is false (true is the default), then control characters will be allowed inside strings. Control characters in this context are those with character codes in the 0-31 range, including  '\\t' (tab),  '\\n' ,  '\\r' and  '\\0' ."
},
{
"ref":"sertit.files.CustomDecoder.object_hook",
"url":4,
"doc":"Overload of object_hook function that deals with  datetime.datetime Args: obj (dict): Dict containing objects to decode from JSON Returns: dict: Dict with decoded object",
"func":1
},
{
"ref":"sertit.files.CustomEncoder",
"url":4,
"doc":"Encoder for JSON with methods for datetimes and np.int64 Constructor for JSONEncoder, with sensible defaults. If skipkeys is false, then it is a TypeError to attempt encoding of keys that are not str, int, float or None. If skipkeys is True, such items are simply skipped. If ensure_ascii is true, the output is guaranteed to be str objects with all incoming non-ASCII characters escaped. If ensure_ascii is false, the output can contain non-ASCII characters. If check_circular is true, then lists, dicts, and custom encoded objects will be checked for circular references during encoding to prevent an infinite recursion (which would cause an OverflowError). Otherwise, no such check takes place. If allow_nan is true, then NaN, Infinity, and -Infinity will be encoded as such. This behavior is not JSON specification compliant, but is consistent with most JavaScript based encoders and decoders. Otherwise, it will be a ValueError to encode such floats. If sort_keys is true, then the output of dictionaries will be sorted by key; this is useful for regression tests to ensure that JSON serializations can be compared on a day-to-day basis. If indent is a non-negative integer, then JSON array elements and object members will be pretty-printed with that indent level. An indent level of 0 will only insert newlines. None is the most compact representation. If specified, separators should be an (item_separator, key_separator) tuple. The default is (', ', ': ') if  indent is  None and (',', ': ') otherwise. To get the most compact JSON representation, you should specify (',', ':') to eliminate whitespace. If specified, default is a function that gets called for objects that can't otherwise be serialized. It should return a JSON encodable version of the object or raise a  TypeError ."
},
{
"ref":"sertit.files.CustomEncoder.default",
"url":4,
"doc":"Overload of the default method",
"func":1
},
{
"ref":"sertit.files.read_json",
"url":4,
"doc":"Read a JSON file   >>> json_path = 'D:\\path\\to\\json.json' >>> read_json(json_path, print_file=False) {\"A\": 1, \"B\": 2}   Args: json_file (Union[str, CloudPath, Path]): Path to JSON file print_file (bool): Print the configuration file Returns: dict: JSON data",
"func":1
},
{
"ref":"sertit.files.save_json",
"url":4,
"doc":"Save a JSON file, with datetime, numpy types and Enum management.   >>> output_json = 'D:\\path\\to\\json.json' >>> json_dict = {\"A\": np.int64(1), \"B\": datetime.today(), \"C\": SomeEnum.some_name} >>> save_json(output_json, json_dict)   Args: output_json (Union[str, CloudPath, Path]): Output file json_dict (dict): Json dictionary",
"func":1
},
{
"ref":"sertit.files.save_obj",
"url":4,
"doc":"Save an object as a pickle (can save any Python objects).   >>> output_pkl = 'D:\\path\\to\\pickle.pkl' >>> pkl_dict = {\"A\": np.ones([3, 3]), \"B\": datetime.today(), \"C\": SomeEnum.some_name} >>> save_json(output_pkl, pkl_dict)   Args: obj (Any): Any object serializable path (Union[str, CloudPath, Path]): Path where to write the pickle",
"func":1
},
{
"ref":"sertit.files.load_obj",
"url":4,
"doc":"Load a pickled object.   >>> output_pkl = 'D:\\path\\to\\pickle.pkl' >>> load_obj(output_pkl) {\"A\": np.ones([3, 3]), \"B\": datetime.today(), \"C\": SomeEnum.some_name}   Args: path (Union[str, CloudPath, Path]): Path of the pickle Returns: object (Any): Pickled object",
"func":1
},
{
"ref":"sertit.files.get_file_in_dir",
"url":4,
"doc":"Get one or all matching files (pattern + extension) from inside a directory. Note that the pattern is a regex with glob's convention, ie.  pattern . If  exact_name is  False , the searched pattern will be  {pattern} .{extension} , else  {pattern}.{extension} .   >>> directory = 'D:\\path\\to\\dir' >>> os.listdir(directory) [\"haha.txt\", \"huhu1.txt\", \"huhu1.geojson\", \"hoho.txt\"] >>> get_file_in_dir(directory, \"huhu\") 'D:\\path\\to\\dir\\huhu1.geojson' >>> get_file_in_dir(directory, \"huhu\", extension=\"txt\") 'D:\\path\\to\\dir\\huhu1.txt' >>> get_file_in_dir(directory, \"huhu\", get_list=True) ['D:\\path\\to\\dir\\huhu1.txt', 'D:\\path\\to\\dir\\huhu1.geojson'] >>> get_file_in_dir(directory, \"huhu\", filename_only=True, get_list=True) ['huhu1.txt', 'huhu1.geojson'] >>> get_file_in_dir(directory, \"huhu\", get_list=True, exact_name=True) []   Args: directory (str): Directory where to find the files pattern_str (str): Pattern wanted as a string, with glob's convention. extension (str): Extension wanted, optional. With or without point. ( yaml or  .yaml accepted) filename_only (bool): Get only the filename get_list (bool): Get the whole list of matching files exact_name (bool): Get the exact name (without adding  before and after the given pattern) Returns: Union[CloudPath, Path, list]: File",
"func":1
},
{
"ref":"sertit.files.hash_file_content",
"url":4,
"doc":"Hash a file into a unique str.   >>> read_json(\"path\\to\\json.json\") {\"A\": 1, \"B\": 2} >>> hash_file_content(str(file_content \"d3fad5bdf9\"   Args: file_content (str): File content len_param (int): Length parameter for the hash (length of the key will be 2x this number) Returns: str: Hashed file content",
"func":1
},
{
"ref":"sertit.misc",
"url":5,
"doc":"Miscellaneous Tools"
},
{
"ref":"sertit.misc.ListEnum",
"url":5,
"doc":"List Enum (enum with function listing names and values)   >>> @unique >>> class TsxPolarization(ListEnum): >>> SINGLE = \"S\"  Single >>> DUAL = \"D\"  Dual >>> QUAD = \"Q\"  Quad >>> TWIN = \"T\"  Twin  "
},
{
"ref":"sertit.misc.ListEnum.list_values",
"url":5,
"doc":"Get the value list of this enum   >>> TsxPolarization.list_values() [\"S\", \"D\", \"Q\", \"T\"]  ",
"func":1
},
{
"ref":"sertit.misc.ListEnum.list_names",
"url":5,
"doc":"Get the name list of this enum:   >>> TsxPolarization.list_values() [\"SINGLE\", \"DUAL\", \"QUAD\", \"TWIN\"]  ",
"func":1
},
{
"ref":"sertit.misc.ListEnum.from_value",
"url":5,
"doc":"Get the enum class from its value:   >>> TsxPolarization.from_value(\"Q\")    Args: val (Any): Value of the Enum Returns: ListEnum: Enum with value",
"func":1
},
{
"ref":"sertit.misc.ListEnum.convert_from",
"url":5,
"doc":"Convert from a list or a string to an enum instance   >>> TsxPolarization.convert_from([\"SINGLE\", \"S\", TsxPolarization.QUAD]) [ ,  ,  ]   Args: to_convert (Union[list, str]): List or string to convert into an enum instance Returns: list: Converted list",
"func":1
},
{
"ref":"sertit.misc.remove_empty_values",
"url":5,
"doc":"Remove empty values from list:   >>> lst = [\"A\", \"T\", \"R\",  , 3, None] >>> list_to_dict(lst) [\"A\", \"T\", \"R\", 3]   Args: list_with_empty_values (list): List with empty values Returns: list: Curated list",
"func":1
},
{
"ref":"sertit.misc.list_to_dict",
"url":5,
"doc":"Return a dictionary from a list  [key, value, key_2, value_2 .]   >>> lst = [\"A\",\"T\", \"R\", 3] >>> list_to_dict(lst) {\"A\": \"T\", \"R\": 3}   Args: dict_list (list[str]): Dictionary as a list Returns: dict: Dictionary",
"func":1
},
{
"ref":"sertit.misc.nested_set",
"url":5,
"doc":"Set value in nested directory:   >>> dct = {\"A\": \"T\", \"R\": 3} >>> nested_set(dct, keys=[\"B\", \"C\", \"D\"], value=\"value\") { \"A\": \"T\", \"R\": 3, \"B\": { \"C\": { \"D\": \"value\" } } }   Args: dic (dict): Dictionary keys (list[str]): Keys as a list value: Value to be set",
"func":1
},
{
"ref":"sertit.misc.check_mandatory_keys",
"url":5,
"doc":"Check all mandatory argument in a dictionary. Raise an exception if a mandatory argument is missing.  Note : nested keys do not work here !   >>> dct = {\"A\": \"T\", \"R\": 3} >>> check_mandatory_keys(dct, [\"A\", \"R\"])  Returns nothing, is OK >>> check_mandatory_keys(dct, [\"C\"]) Traceback (most recent call last): File \" \", line 1, in  File \" \", line 167, in check_mandatory_keys ValueError: Missing mandatory key 'C' among {'A': 'T', 'R': 3}   Args: data_dict (dict): Data dictionary to be checked mandatory_keys (list[str]): List of mandatory keys",
"func":1
},
{
"ref":"sertit.misc.find_by_key",
"url":5,
"doc":"Find a value by key in a dictionary.   >>> dct = { >>> \"A\": \"T\", >>> \"R\": 3, >>> \"B\": { >>> \"C\": { >>> \"D\": \"value\" >>> } >>> } >>> } >>> find_by_key(dct, \"D\") \"value\"   Args: data (dict): Dict to walk through target (str): target key Returns: Any: Value data[ .][target]",
"func":1
},
{
"ref":"sertit.misc.run_cli",
"url":5,
"doc":"Run a command line.   >>> cmd_hillshade = [\"gdaldem\", \" config\", >>> \"NUM_THREADS\", \"1\", >>> \"hillshade\", strings.to_cmd_string(dem_path), >>> \"-compute_edges\", >>> \"-z\", self.nof_threads, >>> \"-az\", azimuth, >>> \"-alt\", zenith, >>> \"-of\", \"GTiff\", >>> strings.to_cmd_string(hillshade_dem)] >>>  Run command >>> run_cli(cmd_hillshade)   Args: cmd (str or list[str]): Command as a list timeout (float): Timeout check_return_value (bool): Check output value of the exe in_background (bool): Run the subprocess in background cwd (str): Working directory Returns: int, str: return value and output log",
"func":1
},
{
"ref":"sertit.misc.get_function_name",
"url":5,
"doc":"Get the name of the function where this one is launched.   >>> def huhuhu(): >>> return get_function_name() >>> huhuhu() \"huhuhu\"   Returns: str: Function's name",
"func":1
},
{
"ref":"sertit.misc.in_docker",
"url":5,
"doc":"Check if the session is running inside a docker   if in_docker(): print(\"OMG we are stock in a Docker ! Get me out of here !\") else: print(\"We are safe\")   Returns: bool: True if inside a docker",
"func":1
},
{
"ref":"sertit.misc.chdir",
"url":5,
"doc":"Change current directory, used as a context manager, ie:   >>> folder = r\"C:\" >>> with chdir(folder): >>> print(os.getcwd( 'C:\\'   Args: newdir (str): New directory",
"func":1
},
{
"ref":"sertit.logs",
"url":6,
"doc":"Logging tools"
},
{
"ref":"sertit.logs.init_logger",
"url":6,
"doc":"Initialize a very basic logger to trace the first lines in the stream. To be done before everything (like parsing log_file etc .)   >>> logger = logging.getLogger(\"logger_test\") >>> init_logger(logger, logging.INFO, '%(asctime)s - [%(levelname)s] - %(message)s') >>> logger.info(\"MESSAGE\") 2021-03-02 16:57:35 - [INFO] - MESSAGE   Args: curr_logger (logging.Logger): Logger to be initialize log_lvl (int): Logging level to be set log_format (str): Logger format to be set",
"func":1
},
{
"ref":"sertit.logs.create_logger",
"url":6,
"doc":"Create file and stream logger at the wanted level for the given logger. - If you have  colorlog installed, it will produce colored logs. - If you do not give any output and name, it won't create any file logger It will also manage the log level of other specified logger that you give.   >>> logger = logging.getLogger(\"logger_test\") >>> create_logger(logger, logging.DEBUG, logging.INFO, \"path\\to\\log\", \"log.txt\") >>> logger.info(\"MESSAGE\") 2021-03-02 16:57:35 - [INFO] - MESSAGE >>>  \"logger_test\" will also log DEBUG messages >>>  to the \"path\\to\\log\\log.txt\" file with the same format   Args: logger (logging.Logger): Logger to create file_log_level (int): File log level stream_log_level (int): Stream log level output_folder (str): Output folder. Won't create File logger if not specified name (str): Name of the log file, prefixed with the date and suffixed with _log. Can be None. other_loggers_names (Union[str, list]): Other existing logger to manage (setting the right format and log level) other_loggers_file_log_level (int): File log level for other loggers other_loggers_stream_log_level (int): Stream log level for other loggers",
"func":1
},
{
"ref":"sertit.logs.shutdown_logger",
"url":6,
"doc":"Shutdown logger (if you need to delete the log file for example)   >>> logger = logging.getLogger(\"logger_test\") >>> shutdown_logger(logger) >>>  \"logger_test\" won't log anything after another init   Args: logger (logging.Logger): Logger to shutdown",
"func":1
},
{
"ref":"sertit.logs.reset_logging",
"url":6,
"doc":"Reset root logger  WARNING MAY BE OVERKILL   >>> reset_logging() Reset root logger  ",
"func":1
},
{
"ref":"sertit.rasters",
"url":7,
"doc":"Raster tools You can use this only if you have installed sertit[full] or sertit[rasters]"
},
{
"ref":"sertit.rasters.PATH_XARR_DS",
"url":7,
"doc":"Types: - Path - rasterio Dataset -  xarray.DataArray and  xarray.Dataset "
},
{
"ref":"sertit.rasters.XDS_TYPE",
"url":7,
"doc":"Xarray types: xr.Dataset and xr.DataArray"
},
{
"ref":"sertit.rasters.path_xarr_dst",
"url":7,
"doc":"Path,  xarray or dataset decorator. Allows a function to ingest: - a path - a  xarray - a  rasterio dataset   >>>  Create mock function >>> @path_or_dst >>> def fct(dst): >>> read(dst) >>> >>>  Test the two ways >>> read1 = fct(\"path\\to\\raster.tif\") >>> with rasterio.open(\"path\\to\\raster.tif\") as dst: >>> read2 = fct(dst) >>> >>>  Test >>> read1  read2 True   Args: function (Callable): Function to decorate Returns: Callable: decorated function",
"func":1
},
{
"ref":"sertit.rasters.to_np",
"url":7,
"doc":"Convert the  xarray to a  np.ndarray with the correct nodata encoded. This is particularly useful when reading with  masked=True .   >>> raster_path = \"path\\to\\mask.tif\"  Classified raster in np.uint8 with nodata = 255 >>>  We read with masked=True so the data is converted to float >>> xds = read(raster_path)  [149408 values with dtype=float64] Coordinates:  band (band) int32 1  y (y) float64 4.798e+06 4.798e+06  . 4.788e+06 4.788e+06  x (x) float64 5.411e+05 5.411e+05  . 5.549e+05 5.55e+05 spatial_ref int32 0 >>> to_np(xds)  Getting back np.uint8 and encoded nodata array( [255, 255, 255,  ., 255, 255, 255], [255, 255, 255,  ., 255, 255, 255], [255, 255, 255,  ., 255, 255, 255],  ., [255, 255, 255,  ., 1, 255, 255], [255, 255, 255,  ., 1, 255, 255], [255, 255, 255,  ., 1, 255, 255 ], dtype=uint8) True   Args: xds (xarray.DataArray):  xarray.DataArray to convert dtype (Any): Dtype to convert to. If None, using the origin dtype if existing or its current dtype. Returns:",
"func":1
},
{
"ref":"sertit.rasters.get_nodata_mask",
"url":7,
"doc":"Get nodata mask from a xarray.   >>> diag_arr = xr.DataArray(data=np.diag([1, 2, 3] >>> diag_arr.rio.write_nodata(0, inplace=True)  array( 1, 0, 0], [0, 2, 0], [0, 0, 3 ) Dimensions without coordinates: dim_0, dim_1 >>> get_nodata_mask(diag_arr) array( 1, 0, 0], [0, 1, 0], [0, 0, 1 , dtype=uint8)   Args: xds (XDS_TYPE): Array to evaluate Returns: np.ndarray: Pixelwise nodata array",
"func":1
},
{
"ref":"sertit.rasters.vectorize",
"url":7,
"doc":"Vectorize a  xarray to get the class vectors. If dissolved is False, it returns a GeoDataFrame with a GeoSeries per cluster of pixel value, with the value as an attribute. Else it returns a GeoDataFrame with a unique polygon.  WARNING - Your data is casted by force into np.uint8, so be sure that your data is classified. - This could take a while as the computing time directly depends on the number of polygons to vectorize. Please be careful.   >>> raster_path = \"path\\to\\raster.tif\" >>> vec1 = vectorize(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> vec2 = vectorize(dst) >>> vec1  vec2 True   Args: xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray values (Union[None, int, list]): Get only the polygons concerning this/these particular values keep_values (bool): Keep the passed values. If False, discard them and keep the others. dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given. default_nodata (int): Default values for nodata in case of non existing in file Returns: gpd.GeoDataFrame: Classes Vector",
"func":1
},
{
"ref":"sertit.rasters.get_valid_vector",
"url":7,
"doc":"Get the valid data of a raster as a vector. Pay attention that every nodata pixel will appear too. If you want only the footprint of the raster, please use  get_footprint .   >>> raster_path = \"path\\to\\raster.tif\" >>> nodata1 = get_nodata_vec(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> nodata2 = get_nodata_vec(dst) >>> nodata1  nodata2 True   Args: xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray default_nodata (int): Default values for nodata in case of non existing in file Returns: gpd.GeoDataFrame: Nodata Vector",
"func":1
},
{
"ref":"sertit.rasters.get_nodata_vector",
"url":7,
"doc":"Get the nodata vector of a raster as a vector. Pay attention that every nodata pixel will appear too. If you want only the footprint of the raster, please use  get_footprint .   >>> raster_path = \"path\\to\\raster.tif\"  Classified raster, with no data set to 255 >>> nodata1 = get_nodata_vec(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> nodata2 = get_nodata_vec(dst) >>> nodata1  nodata2 True   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata default_nodata (int): Default values for nodata in case of non existing in file Returns: gpd.GeoDataFrame: Nodata Vector",
"func":1
},
{
"ref":"sertit.rasters.mask",
"url":7,
"doc":"Masking a dataset: setting nodata outside of the given shapes, but without cropping the raster to the shapes extent. The original nodata is kept and completed with the nodata provided by the shapes. Overload of rasterio mask function in order to create a  xarray . The  mask function docs can be seen [here](https: rasterio.readthedocs.io/en/latest/api/rasterio.mask.html). It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.   >>> raster_path = \"path\\to\\raster.tif\" >>> shape_path = \"path\\to\\shapes.geojson\"  Any vector that geopandas can read >>> shapes = gpd.read_file(shape_path) >>> mask1 = mask(raster_path, shapes) >>>  or >>> with rasterio.open(raster_path) as dst: >>> mask2 = mask(dst, shapes) >>> mask1  mask2 True   Args: xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset (except if a  GeoDataFrame is passed, in which case it will automatically be converted) nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.  kwargs: Other rasterio.mask options Returns: XDS_TYPE: Masked array as a xarray",
"func":1
},
{
"ref":"sertit.rasters.paint",
"url":7,
"doc":"Painting a dataset: setting values inside the given shapes. To set outside the shape, set invert=True. Pay attention that this behavior is the opposite of the  rasterio.mask function. The original nodata is kept. This means if your shapes intersects the original nodata, the value of the pixel will be set to nodata rather than to the wanted value. Overload of rasterio mask function in order to create a  xarray . The  mask function docs can be seen [here](https: rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).   >>> raster_path = \"path\\to\\raster.tif\" >>> shape_path = \"path\\to\\shapes.geojson\"  Any vector that geopandas can read >>> shapes = gpd.read_file(shape_path) >>> paint1 = paint(raster_path, shapes, value=100) >>>  or >>> with rasterio.open(raster_path) as dst: >>> paint2 = paint(dst, shapes, value=100) >>> paint1  paint2 True   Args: xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset (except if a  GeoDataFrame is passed, in which case it will automatically be converted) value (int): Value to set on the shapes. invert (bool): If invert is True, set value outside the shapes.  kwargs: Other rasterio.mask options Returns: XDS_TYPE: Painted array as a xarray",
"func":1
},
{
"ref":"sertit.rasters.crop",
"url":7,
"doc":"Cropping a dataset: setting nodata outside of the given shapes AND cropping the raster to the shapes extent. Overload of [ rioxarray clip](https: corteva.github.io/rioxarray/stable/rioxarray.html rioxarray.raster_array.RasterArray.clip) function in order to create a masked_array.   >>> raster_path = \"path\\to\\raster.tif\" >>> shape_path = \"path\\to\\shapes.geojson\"  Any vector that geopandas can read >>> shapes = gpd.read_file(shape_path) >>> xds2 = crop(raster_path, shapes) >>>  or >>> with rasterio.open(raster_path) as dst: >>> xds2 = crop(dst, shapes) >>> xds1  xds2 True   Args: xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset (except if a  GeoDataFrame is passed, in which case it will automatically be converted) nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.  kwargs: Other rioxarray.clip options Returns: XDS_TYPE: Cropped array as a xarray",
"func":1
},
{
"ref":"sertit.rasters.read",
"url":7,
"doc":"Read a raster dataset from a : -  xarray (compatibility issues) -  rasterio.Dataset -  rasterio opened data (array, metadata) - a path. The resolution can be provided (in dataset unit) as: - a tuple or a list of (X, Y) resolutions - a float, in which case X resolution = Y resolution - None, in which case the dataset resolution will be used   >>> raster_path = \"path\\to\\raster.tif\" >>> xds1 = read(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> xds2 = read(dst) >>> xds1  xds2 True   Args: dst (PATH_ARR_DS): Path to the raster or a rasterio dataset or a xarray resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y) size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided. resampling (Resampling): Resampling method masked (bool): Get a masked array indexes (Union[int, list]): Indexes to load. Load the whole array if None. Returns: Union[XDS_TYPE]: Masked xarray corresponding to the raster data and its meta data",
"func":1
},
{
"ref":"sertit.rasters.write",
"url":7,
"doc":"Write raster to disk. (encapsulation of  rasterio 's function, because for now  rioxarray to_raster doesn't work as expected) Metadata will be created with the  xarray metadata (ie. width, height, count, type .) The driver is  GTiff by default, and no nodata value is provided. The file will be compressed if the raster is a mask (saved as uint8). If not overwritten, sets the nodata according to  dtype : - uint8: 255 - int8: -128 - uint16, uint32, int32, int64, uint64: 65535 - int16, float32, float64, float128, float: -9999 Compress with  LZW option by default. To disable it, add the  compress=None parameter.   >>> raster_path = \"path\\to\\raster.tif\" >>> raster_out = \"path\\to\\out.tif\" >>>  Read raster >>> xds = read(raster_path) >>>  Rewrite it >>> write(xds, raster_out)   Args: xds (XDS_TYPE): Path to the raster or a rasterio dataset or a xarray path (Union[str, CloudPath, Path]): Path where to save it (directories should be existing)  kwargs: Overloading metadata, ie  nodata=255 or  dtype=np.uint8 ",
"func":1
},
{
"ref":"sertit.rasters.collocate",
"url":7,
"doc":"Collocate two georeferenced arrays: forces the  slave raster to be exactly georeferenced onto the  master raster by reprojection. Use it like  OTB SuperImpose .   >>> master_path = \"path\\to\\master.tif\" >>> slave_path = \"path\\to\\slave.tif\" >>> col_path = \"path\\to\\collocated.tif\" >>>  Collocate the slave to the master >>> col_xds = collocate(read(master_path), read(slave_path), Resampling.bilinear) >>>  Write it >>> write(col_xds, col_path)   Args: master_xds (XDS_TYPE): Master xarray slave_xds (XDS_TYPE): Slave xarray resampling (Resampling): Resampling method Returns: XDS_TYPE: Collocated xarray",
"func":1
},
{
"ref":"sertit.rasters.sieve",
"url":7,
"doc":"Sieving, overloads rasterio function with raster shaped like (1, h, w).  WARNING Your data is casted by force into  np.uint8 , so be sure that your data is classified.   >>> raster_path = \"path\\to\\raster.tif\"  classified raster >>>  Rewrite it >>> sieved_xds = sieve(raster_path, sieve_thresh=20) >>>  Write it >>> raster_out = \"path\\to\\raster_sieved.tif\" >>> write(sieved_xds, raster_out)   Args: xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray sieve_thresh (int): Sieving threshold in pixels connectivity (int): Connectivity, either 4 or 8 dtype: Dtype of the xarray (if nodata is set, the xds.dtype is float whereas the values are meant to be ie in np.uint8) Returns: (XDS_TYPE): Sieved xarray",
"func":1
},
{
"ref":"sertit.rasters.get_dim_img_path",
"url":7,
"doc":"Get the image path from a  BEAM-DIMAP data. A  BEAM-DIMAP file cannot be opened by rasterio, although its .img file can.   >>> dim_path = \"path\\to\\dimap.dim\"  BEAM-DIMAP image >>> img_path = get_dim_img_path(dim_path) >>>  Read raster >>> raster, meta = read(img_path)   Args: dim_path (Union[str, CloudPath, Path]): DIM path (.dim or .data) img_name (str): .img file name (or regex), in case there are multiple .img files (ie. for S3 data) Returns: Union[CloudPath, Path]: .img file",
"func":1
},
{
"ref":"sertit.rasters.get_extent",
"url":7,
"doc":"Get the extent of a raster as a  geopandas.Geodataframe .   >>> raster_path = \"path\\to\\raster.tif\" >>> extent1 = get_extent(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> extent2 = get_extent(dst) >>> extent1  extent2 True   Args: xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray Returns: gpd.GeoDataFrame: Extent as a  geopandas.Geodataframe ",
"func":1
},
{
"ref":"sertit.rasters.get_footprint",
"url":7,
"doc":"Get real footprint of the product (without nodata, in french  emprise utile)   >>> raster_path = \"path\\to\\raster.tif\" >>> footprint1 = get_footprint(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> footprint2 = get_footprint(dst) >>> footprint1  footprint2   Args: xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray Returns: gpd.GeoDataFrame: Footprint as a GeoDataFrame",
"func":1
},
{
"ref":"sertit.rasters.merge_vrt",
"url":7,
"doc":"Merge rasters as a VRT. Uses  gdalbuildvrt . See here: https: gdal.org/programs/gdalbuildvrt.html Creates VRT with relative paths !  WARNING They should have the same CRS otherwise the mosaic will be false !   >>> paths_utm32630 = [\"path\\to\\raster1.tif\", \"path\\to\\raster2.tif\", \"path\\to\\raster3.tif\"] >>> paths_utm32631 = [\"path\\to\\raster4.tif\", \"path\\to\\raster5.tif\"] >>> mosaic_32630 = \"path\\to\\mosaic_32630.vrt\" >>> mosaic_32631 = \"path\\to\\mosaic_32631.vrt\" >>>  Create mosaic, one by CRS ! >>> merge_vrt(paths_utm32630, mosaic_32630) >>> merge_vrt(paths_utm32631, mosaic_32631, {\"-srcnodata\":255, \"-vrtnodata\":0})   Args: crs_paths (list): Path of the rasters to be merged with the same CRS crs_merged_path (Union[str, CloudPath, Path]): Path to the merged raster kwargs: Other gdlabuildvrt arguments",
"func":1
},
{
"ref":"sertit.rasters.merge_gtiff",
"url":7,
"doc":"Merge rasters as a GeoTiff.  WARNING They should have the same CRS otherwise the mosaic will be false !   >>> paths_utm32630 = [\"path\\to\\raster1.tif\", \"path\\to\\raster2.tif\", \"path\\to\\raster3.tif\"] >>> paths_utm32631 = [\"path\\to\\raster4.tif\", \"path\\to\\raster5.tif\"] >>> mosaic_32630 = \"path\\to\\mosaic_32630.tif\" >>> mosaic_32631 = \"path\\to\\mosaic_32631.tif\"  Create mosaic, one by CRS ! >>> merge_gtiff(paths_utm32630, mosaic_32630) >>> merge_gtiff(paths_utm32631, mosaic_32631)   Args: crs_paths (list): Path of the rasters to be merged with the same CRS crs_merged_path (Union[str, CloudPath, Path]): Path to the merged raster kwargs: Other rasterio.merge arguments More info [here](https: rasterio.readthedocs.io/en/latest/api/rasterio.merge.html rasterio.merge.merge)",
"func":1
},
{
"ref":"sertit.rasters.unpackbits",
"url":7,
"doc":"Function found here: https: stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types   >>> bit_array = np.random.randint(5, size=[3,3]) array( 1, 1, 3], [4, 2, 0], [4, 3, 2 , dtype=uint8)  Unpack 8 bits (8 1, as itemsize of uint8 is 1) >>> unpackbits(bit_array, 8) array( [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0 ,  0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0 ,  0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0 ], dtype=uint8)   Args: array (np.ndarray): Array to unpack nof_bits (int): Number of bits to unpack Returns: np.ndarray: Unpacked array",
"func":1
},
{
"ref":"sertit.rasters.read_bit_array",
"url":7,
"doc":"Read bit arrays as a succession of binary masks (sort of read a slice of the bit mask, slice number bit_id)   >>> bit_array = np.random.randint(5, size=[3,3]) array( 1, 1, 3], [4, 2, 0], [4, 3, 2 , dtype=uint8)  Get the 2nd bit array >>> read_bit_array(bit_array, 2) array( 0, 0, 0], [1, 0, 0], [1, 0, 0 , dtype=uint8)   Args: bit_mask (np.ndarray): Bit array to read bit_id (int): Bit ID of the slice to be read Example: read the bit 0 of the mask as a cloud mask (Theia) Returns: Union[np.ndarray, list]: Binary mask or list of binary masks if a list of bit_id is given",
"func":1
},
{
"ref":"sertit.rasters.read_uint8_array",
"url":7,
"doc":"Read 8 bit arrays as a succession of binary masks. Forces array to  np.uint8 . See  read_bit_array . Args: bit_mask (np.ndarray): Bit array to read bit_id (int): Bit ID of the slice to be read Example: read the bit 0 of the mask as a cloud mask (Theia) Returns: Union[np.ndarray, list]: Binary mask or list of binary masks if a list of bit_id is given",
"func":1
},
{
"ref":"sertit.rasters.set_metadata",
"url":7,
"doc":"Set metadata from a  xr.DataArray to another (including  rioxarray metadata such as encoded_nodata and crs). Useful when performing operations on xarray that result in metadata loss such as sums.   >>>  xda: some xr.DataArray >>> sum = xda + xda  Sum loses its metadata here  array( [nan, nan, nan,  ., nan, nan, nan], [nan, nan, nan,  ., nan, nan, nan], [nan, nan, nan,  ., nan, nan, nan],  ., [nan, nan, nan,  ., 2., nan, nan], [nan, nan, nan,  ., 2., nan, nan], [nan, nan, nan,  ., 2., nan, nan ]) Coordinates:  band (band) int32 1  y (y) float64 4.798e+06 4.798e+06  . 4.788e+06 4.788e+06  x (x) float64 5.411e+05 5.411e+05  . 5.549e+05 5.55e+05 >>>  We need to set the metadata back (and we can set a new name) >>> sum = set_metadata(sum, xda, new_name=\"sum\")  array( [nan, nan, nan,  ., nan, nan, nan], [nan, nan, nan,  ., nan, nan, nan], [nan, nan, nan,  ., nan, nan, nan],  ., [nan, nan, nan,  ., 2., nan, nan], [nan, nan, nan,  ., 2., nan, nan], [nan, nan, nan,  ., 2., nan, nan ]) Coordinates:  band (band) int32 1  y (y) float64 4.798e+06 4.798e+06  . 4.788e+06 4.788e+06  x (x) float64 5.411e+05 5.411e+05  . 5.549e+05 5.55e+05 spatial_ref int32 0 Attributes: (12/13) grid_mapping: spatial_ref BandName: Band_1 RepresentationType: ATHEMATIC STATISTICS_COVARIANCES: 0.2358157950609785 STATISTICS_MAXIMUM: 2 STATISTICS_MEAN: 1.3808942647686  .  . STATISTICS_SKIPFACTORX: 1 STATISTICS_SKIPFACTORY: 1 STATISTICS_STDDEV: 0.48560665546817 STATISTICS_VALID_PERCENT: 80.07 original_dtype: uint8   Args: naked_xda (xr.DataArray): DataArray to complete mtd_xda (xr.DataArray): DataArray with the correct metadata new_name (str): New name for naked DataArray Returns: xr.DataArray: Complete DataArray",
"func":1
},
{
"ref":"sertit.rasters.set_nodata",
"url":7,
"doc":"Set nodata to a xarray that have no default nodata value. In the data array, the no data will be set to  np.nan . The encoded value can be retrieved with  xda.rio.encoded_nodata .   >>> A = xr.DataArray(dims=(\"x\", \"y\"), data=np.zeros 3,3), dtype=np.uint8 >>> A[0, 0] = 1  array( 1, 0, 0], [0, 0, 0], [0, 0, 0 , dtype=uint8) Dimensions without coordinates: x, y >>> A_nodata = set_nodata(A, 0)  array( 1., nan, nan], [nan, nan, nan], [nan, nan, nan ) Dimensions without coordinates: x, y   Args: xda (xr.DataArray): DataArray nodata_val (Union[float, int]): Nodata value Returns: xr.DataArray: DataArray with nodata set",
"func":1
},
{
"ref":"sertit.rasters.where",
"url":7,
"doc":"Overloads  xr.where with: - setting metadata of  master_xda - preserving the nodata pixels of the  master_xda If  master_xda is None, use it like  xr.where . Else, it outputs a  xarray.DataArray with the same dtype than  master_xda .  WARNING If you don't give a  master_xda , it is better to pass numpy arrays to  if_false and  if_true keywords as passing xarrays interfers with the output metadata (you may lose the CRS and so on). Just pass  if_true=true_xda.data inplace of  if_true=true_xda and the same for  if_false   >>> A = xr.DataArray(dims=(\"x\", \"y\"), data= 1, 0, 5], [np.nan, 0, 0 ) >>> mask_A = rasters.where(A > 3, 0, 1, A, new_name=\"mask_A\")  array( 1., 1., 0.], [nan, 1., 1. ) Dimensions without coordinates: x, y   Args: cond (scalar, array, Variable, DataArray or Dataset): Conditional array if_true (scalar, array, Variable, DataArray or Dataset): What to do if  cond is True if_false (scalar, array, Variable, DataArray or Dataset): What to do if  cond is False master_xda: Master  xr.DataArray used to set the metadata and the nodata new_name (str): New name of the array Returns: xr.DataArray: Where array with correct mtd and nodata pixels",
"func":1
},
{
"ref":"sertit.rasters_rio",
"url":8,
"doc":"Raster tools You can use this only if you have installed sertit[full] or sertit[rasters_rio]"
},
{
"ref":"sertit.rasters_rio.PATH_ARR_DS",
"url":8,
"doc":"Types: - Path - Rasterio open data: (array, meta) - rasterio Dataset -  xarray "
},
{
"ref":"sertit.rasters_rio.path_arr_dst",
"url":8,
"doc":"Path,  xarray , (array, metadata) or dataset decorator. Allows a function to ingest: - a path - a  xarray - a  rasterio dataset -  rasterio open data: (array, meta)   >>>  Create mock function >>> @path_or_dst >>> def fct(dst): >>> read(dst) >>> >>>  Test the two ways >>> read1 = fct(\"path\\to\\raster.tif\") >>> with rasterio.open(\"path\\to\\raster.tif\") as dst: >>> read2 = fct(dst) >>> >>>  Test >>> read1  read2 True   Args: function (Callable): Function to decorate Returns: Callable: decorated function",
"func":1
},
{
"ref":"sertit.rasters_rio.get_new_shape",
"url":8,
"doc":"Get the new shape (height, width) of a resampled raster. Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y) size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided. Returns: (int, int): Height, width",
"func":1
},
{
"ref":"sertit.rasters_rio.update_meta",
"url":8,
"doc":"Basic metadata update from a numpy array. Updates everything that we can find in the array: -  dtype : array dtype, -  count : first dimension of the array if the array is in 3D, else 1 -  height : second dimension of the array -  width : third dimension of the array -  nodata : if a masked array is given, nodata is its fill_value  WARNING The array's shape is interpreted in rasterio's way (count, height, width) !   >>> raster_path = \"path\\to\\raster.tif\" >>> with rasterio.open(raster_path) as dst: >>> meta = dst.meta >>> arr = dst.read() >>> meta { 'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': 300, 'height': 300, 'count': 4, 'crs': CRS.from_epsg(32630), 'transform': Affine(20.0, 0.0, 630000.0,0.0, -20.0, 4870020.0) } >>> new_arr = np.ma.masked_array(arr[:,  2,  2].astype(np.uint8), fill_value=0) >>> new_arr.shape (4, 150, 150) >>> new_arr.dtype dtype('uint8') >>> new_arr.fill_value 0 >>> update_meta(new_arr, meta) { 'driver': 'GTiff', 'dtype': dtype('uint8'), 'nodata': 0, 'width': 150, 'height': 150, 'count': 4, 'crs': CRS.from_epsg(32630), 'transform': Affine(20.0, 0.0, 630000.0, 0.0, -20.0, 4870020.0) }   Args: arr (Union[np.ndarray, np.ma.masked_array]): Array from which to update the metadata meta (dict): Metadata to update Returns: dict: Update metadata",
"func":1
},
{
"ref":"sertit.rasters_rio.get_nodata_mask",
"url":8,
"doc":"Get nodata mask from a masked array. The nodata may not be set before, then pass a nodata value that will be evaluated on the array.   >>> diag_arr = np.diag([1,2,3]) array( 1, 0, 0], [0, 2, 0], [0, 0, 3 ) >>> get_nodata_mask(diag_arr, has_nodata=False) array( 1, 0, 0], [0, 1, 0], [0, 0, 1 , dtype=uint8) >>> get_nodata_mask(diag_arr, has_nodata=False, default_nodata=1) array( 0, 1, 1], [1, 1, 1], [1, 1, 1 , dtype=uint8)   Args: array (np.ma.masked_array): Array to evaluate has_nodata (bool): If the array as its nodata specified. If not, using default_nodata. default_nodata (int): Default nodata used if the array's nodata is not set Returns: np.ndarray: Pixelwise nodata array",
"func":1
},
{
"ref":"sertit.rasters_rio.vectorize",
"url":8,
"doc":"Vectorize a raster to get the class vectors. If dissolved is False, it returns a GeoDataFrame with a GeoSeries per cluster of pixel value, with the value as an attribute. Else it returns a GeoDataFrame with a unique polygon.  WARNING - Please only use this function on a classified raster. - This could take a while as the computing time directly depends on the number of polygons to vectorize. Please be careful.   >>> raster_path = \"path\\to\\raster.tif\"  Classified raster, with no data set to 255 >>> vec1 = vectorize(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> vec2 = vectorize(dst) >>> vec1  vec2 True   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata values (Union[None, int, list]): Get only the polygons concerning this/these particular values keep_values (bool): Keep the passed values. If False, discard them and keep the others. dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given. default_nodata (int): Default values for nodata in case of non existing in file Returns: gpd.GeoDataFrame: Classes Vector",
"func":1
},
{
"ref":"sertit.rasters_rio.get_valid_vector",
"url":8,
"doc":"Get the valid data of a raster as a vector. Pay attention that every nodata pixel will appear too. If you want only the footprint of the raster, please use  get_footprint .   >>> raster_path = \"path\\to\\raster.tif\"  Classified raster, with no data set to 255 >>> nodata1 = get_nodata_vec(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> nodata2 = get_nodata_vec(dst) >>> nodata1  nodata2 True   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata default_nodata (int): Default values for nodata in case of non existing in file Returns: gpd.GeoDataFrame: Nodata Vector",
"func":1
},
{
"ref":"sertit.rasters_rio.get_nodata_vector",
"url":8,
"doc":"Get the nodata vector of a raster as a vector. Pay attention that every nodata pixel will appear too. If you want only the footprint of the raster, please use  get_footprint .   >>> raster_path = \"path\\to\\raster.tif\"  Classified raster, with no data set to 255 >>> nodata1 = get_nodata_vec(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> nodata2 = get_nodata_vec(dst) >>> nodata1  nodata2 True   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata default_nodata (int): Default values for nodata in case of non existing in file Returns: gpd.GeoDataFrame: Nodata Vector",
"func":1
},
{
"ref":"sertit.rasters_rio.mask",
"url":8,
"doc":"Masking a dataset: setting nodata outside of the given shapes, but without cropping the raster to the shapes extent. Overload of rasterio mask function in order to create a masked_array. The  mask function docs can be seen [here](https: rasterio.readthedocs.io/en/latest/api/rasterio.mask.html). It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.   >>> raster_path = \"path\\to\\raster.tif\" >>> shape_path = \"path\\to\\shapes.geojson\"  Any vector that geopandas can read >>> shapes = gpd.read_file(shape_path) >>> masked_raster1, meta1 = mask(raster_path, shapes) >>>  or >>> with rasterio.open(raster_path) as dst: >>> masked_raster2, meta2 = mask(dst, shapes) >>> masked_raster1  masked_raster2 True >>> meta1  meta2 True   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset (except if a  GeoDataFrame is passed, in which case it will automatically be converted. nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.  kwargs: Other rasterio.mask options Returns: (np.ma.masked_array, dict): Masked array as a masked array and its metadata",
"func":1
},
{
"ref":"sertit.rasters_rio.crop",
"url":8,
"doc":"Cropping a dataset: setting nodata outside of the given shapes AND cropping the raster to the shapes extent.  HOW: Overload of rasterio mask function in order to create a masked_array. The  mask function docs can be seen [here](https: rasterio.readthedocs.io/en/latest/api/rasterio.mask.html). It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.   >>> raster_path = \"path\\to\\raster.tif\" >>> shape_path = \"path\\to\\shapes.geojson\"  Any vector that geopandas can read >>> shapes = gpd.read_file(shape_path) >>> cropped_raster1, meta1 = crop(raster_path, shapes) >>>  or >>> with rasterio.open(raster_path) as dst: >>> cropped_raster2, meta2 = crop(dst, shapes) >>> cropped_raster1  cropped_raster2 True >>> meta1  meta2 True   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset (except if a  GeoDataFrame is passed, in which case it will automatically be converted. nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.  kwargs: Other rasterio.mask options Returns: (np.ma.masked_array, dict): Cropped array as a masked array and its metadata",
"func":1
},
{
"ref":"sertit.rasters_rio.read",
"url":8,
"doc":"Read a raster dataset from a  rasterio.Dataset or a path. The resolution can be provided (in dataset unit) as: - a tuple or a list of (X, Y) resolutions - a float, in which case X resolution = Y resolution - None, in which case the dataset resolution will be used Tip: Use index with a list of one element to keep a 3D array   >>> raster_path = \"path\\to\\raster.tif\" >>> raster1, meta1 = read(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> raster2, meta2 = read(dst) >>> raster1  raster2 True >>> meta1  meta2 True   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y) size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided. resampling (Resampling): Resampling method (nearest by default) masked (bool): Get a masked array,  True by default (whereas it is False by default in rasterio)  kwargs: Other dst.read() arguments such as indexes. Returns: np.ma.masked_array, dict: Masked array corresponding to the raster data and its meta data",
"func":1
},
{
"ref":"sertit.rasters_rio.write",
"url":8,
"doc":"Write raster to disk (encapsulation of rasterio's function) Metadata will be copied and updated with raster's information (ie. width, height, count, type .) The driver is GTiff by default, and no nodata value is provided. The file will be compressed if the raster is a mask (saved as uint8)   >>> raster_path = \"path\\to\\raster.tif\" >>> raster_out = \"path\\to\\out.tif\" >>>  Read raster >>> raster, meta = read(raster_path) >>>  Rewrite it on disk >>> write(raster, meta, raster_out)   Args: raster (Union[np.ma.masked_array, np.ndarray]): Raster to save on disk meta (dict): Basic metadata that will be copied and updated with raster's information path (Union[str, CloudPath, Path]): Path where to save it (directories should be existing)  kwargs: Overloading metadata, ie  nodata=255 ",
"func":1
},
{
"ref":"sertit.rasters_rio.collocate",
"url":8,
"doc":"Collocate two georeferenced arrays: forces the  slave raster to be exactly georeferenced onto the  master raster by reprojection. Use it like  OTB SuperImpose .   >>> master_path = \"path\\to\\master.tif\" >>> slave_path = \"path\\to\\slave.tif\" >>> col_path = \"path\\to\\collocated.tif\" >>>  Just open the master data >>> with rasterio.open(master_path) as master_dst: >>>  Read slave >>> slave, slave_meta = read(slave_path) >>>  Collocate the slave to the master >>> col_arr, col_meta = collocate(master_dst.meta, >>> slave, >>> slave_meta, >>> Resampling.bilinear) >>>  Write it >>> write(col_arr, col_path, col_meta)   Args: master_meta (dict): Master metadata slave_arr (np.ma.masked_array): Slave array to be collocated slave_meta (dict): Slave metadata resampling (Resampling): Resampling method Returns: np.ma.masked_array, dict: Collocated array and its metadata",
"func":1
},
{
"ref":"sertit.rasters_rio.sieve",
"url":8,
"doc":"Sieving, overloads rasterio function with raster shaped like (1, h, w). Forces the output to  np.uint8 (as only classified rasters should be sieved)   >>> raster_path = \"path\\to\\raster.tif\"  classified raster >>>  Read raster >>> raster, meta = read(raster_path) >>>  Rewrite it >>> sieved, sieved_meta = sieve(raster, meta, sieve_thresh=20) >>>  Write it >>> raster_out = \"path\\to\\raster_sieved.tif\" >>> write(sieved, raster_out, sieved_meta)   Args: array (Union[np.ma.masked_array, np.ndarray]): Array to sieve out_meta (dict): Metadata to update sieve_thresh (int): Sieving threshold in pixels connectivity (int): Connectivity, either 4 or 8 Returns: (Union[np.ma.masked_array, np.ndarray], dict): Sieved array and updated meta",
"func":1
},
{
"ref":"sertit.rasters_rio.get_dim_img_path",
"url":8,
"doc":"Get the image path from a  BEAM-DIMAP data. A  BEAM-DIMAP file cannot be opened by rasterio, although its .img file can.   >>> dim_path = \"path\\to\\dimap.dim\"  BEAM-DIMAP image >>> img_path = get_dim_img_path(dim_path) >>>  Read raster >>> raster, meta = read(img_path)   Args: dim_path (Union[str, CloudPath, Path]): DIM path (.dim or .data) img_name (str): .img file name (or regex), in case there are multiple .img files (ie. for S3 data) Returns: Union[CloudPath, Path]: .img file",
"func":1
},
{
"ref":"sertit.rasters_rio.get_extent",
"url":8,
"doc":"Get the extent of a raster as a  geopandas.Geodataframe .   >>> raster_path = \"path\\to\\raster.tif\" >>> extent1 = get_extent(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> extent2 = get_extent(dst) >>> extent1  extent2 True   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata Returns: gpd.GeoDataFrame: Extent as a  geopandas.Geodataframe ",
"func":1
},
{
"ref":"sertit.rasters_rio.get_footprint",
"url":8,
"doc":"Get real footprint of the product (without nodata, in french  emprise utile)   >>> raster_path = \"path\\to\\raster.tif\" >>> footprint1 = get_footprint(raster_path) >>>  or >>> with rasterio.open(raster_path) as dst: >>> footprint2 = get_footprint(dst) >>> footprint1  footprint2   Args: dst (PATH_ARR_DS): Path to the raster, its dataset, its  xarray or a tuple containing its array and metadata Returns: gpd.GeoDataFrame: Footprint as a GeoDataFrame",
"func":1
},
{
"ref":"sertit.rasters_rio.merge_vrt",
"url":8,
"doc":"Merge rasters as a VRT. Uses  gdalbuildvrt . See here: https: gdal.org/programs/gdalbuildvrt.html Creates VRT with relative paths !  WARNING They should have the same CRS otherwise the mosaic will be false !   >>> paths_utm32630 = [\"path\\to\\raster1.tif\", \"path\\to\\raster2.tif\", \"path\\to\\raster3.tif\"] >>> paths_utm32631 = [\"path\\to\\raster4.tif\", \"path\\to\\raster5.tif\"] >>> mosaic_32630 = \"path\\to\\mosaic_32630.vrt\" >>> mosaic_32631 = \"path\\to\\mosaic_32631.vrt\" >>>  Create mosaic, one by CRS ! >>> merge_vrt(paths_utm32630, mosaic_32630) >>> merge_vrt(paths_utm32631, mosaic_32631, {\"-srcnodata\":255, \"-vrtnodata\":0})   Args: crs_paths (list): Path of the rasters to be merged with the same CRS) crs_merged_path (Union[str, CloudPath, Path]): Path to the merged raster kwargs: Other gdlabuildvrt arguments",
"func":1
},
{
"ref":"sertit.rasters_rio.merge_gtiff",
"url":8,
"doc":"Merge rasters as a GeoTiff.  WARNING They should have the same CRS otherwise the mosaic will be false !   >>> paths_utm32630 = [\"path\\to\\raster1.tif\", \"path\\to\\raster2.tif\", \"path\\to\\raster3.tif\"] >>> paths_utm32631 = [\"path\\to\\raster4.tif\", \"path\\to\\raster5.tif\"] >>> mosaic_32630 = \"path\\to\\mosaic_32630.tif\" >>> mosaic_32631 = \"path\\to\\mosaic_32631.tif\"  Create mosaic, one by CRS ! >>> merge_gtiff(paths_utm32630, mosaic_32630) >>> merge_gtiff(paths_utm32631, mosaic_32631)   Args: crs_paths (list): Path of the rasters to be merged with the same CRS) crs_merged_path (Union[str, CloudPath, Path]): Path to the merged raster kwargs: Other rasterio.merge arguments More info [here](https: rasterio.readthedocs.io/en/latest/api/rasterio.merge.html rasterio.merge.merge)",
"func":1
},
{
"ref":"sertit.rasters_rio.unpackbits",
"url":8,
"doc":"Function found here: https: stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types   >>> bit_array = np.random.randint(5, size=[3,3]) array( 1, 1, 3], [4, 2, 0], [4, 3, 2 , dtype=uint8)  Unpack 8 bits (8 1, as itemsize of uint8 is 1) >>> unpackbits(bit_array, 8) array( [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0 ,  0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0 ,  0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0 ], dtype=uint8)   Args: array (np.ndarray): Array to unpack nof_bits (int): Number of bits to unpack Returns: np.ndarray: Unpacked array",
"func":1
},
{
"ref":"sertit.rasters_rio.read_bit_array",
"url":8,
"doc":"Read bit arrays as a succession of binary masks (sort of read a slice of the bit mask, slice number bit_id)   >>> bit_array = np.random.randint(5, size=[3,3]) array( 1, 1, 3], [4, 2, 0], [4, 3, 2 , dtype=uint8)  Get the 2nd bit array >>> read_bit_array(bit_array, 2) array( 0, 0, 0], [1, 0, 0], [1, 0, 0 , dtype=uint8)   Args: bit_mask (np.ndarray): Bit array to read bit_id (int): Bit ID of the slice to be read Example: read the bit 0 of the mask as a cloud mask (Theia) Returns: Union[np.ndarray, list]: Binary mask or list of binary masks if a list of bit_id is given",
"func":1
},
{
"ref":"sertit.display",
"url":9,
"doc":"Display tools"
},
{
"ref":"sertit.display.scale",
"url":9,
"doc":"Scale a raster given as a np.ndarray between 0 and 1. The min max are computed with percentiles (2 by default), but can be true min/max if  perc=0 .  WARNING If 3D, the raster should be in rasterio's convention:  (count, height, width) Args: array (Union[np.ndarray, numpy.ma.masked_array]): Matrix to be scaled perc (int): Percentile to cut. 0 = min/max, 2 by default Returns: numpy array: Scaled matrix",
"func":1
},
{
"ref":"sertit.ci",
"url":10,
"doc":"CI tools You can use  assert_raster_equal only if you have installed sertit[full] or sertit[rasters]"
},
{
"ref":"sertit.ci.get_mnt_path",
"url":10,
"doc":"Return mounting directory  /mnt .  WARNING This won't work on Windows !   >>> get_mnt_path() '/mnt'   Returns: str: Mounting directory",
"func":1
},
{
"ref":"sertit.ci.get_db2_path",
"url":10,
"doc":"Return mounted directory  /mnt/ds2_db2 which corresponds to  \\ds2\\database02 .  WARNING Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !   >>> get_db2_path() '/mnt/ds2_db2'   Returns: str: Mounted directory",
"func":1
},
{
"ref":"sertit.ci.get_db3_path",
"url":10,
"doc":"Return mounted directory  /mnt/ds2_db3 which corresponds to  \\ds2\\database03 .  WARNING Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !   >>> get_db3_path() '/mnt/ds2_db3'   Returns: str: Mounted directory",
"func":1
},
{
"ref":"sertit.ci.get_db4_path",
"url":10,
"doc":"Return mounted directory  /mnt/ds2_db4 which corresponds to  \\ds2\\database04 .  WARNING Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !   >>> get_db4_path() '/mnt/ds2_db4'   Returns: str: Mounted directory",
"func":1
},
{
"ref":"sertit.ci.assert_raster_equal",
"url":10,
"doc":"Assert that two rasters are equal. -> Useful for pytests.   >>> path = r\"CI\\DATA asters aster.tif\" >>> assert_raster_equal(path, path) >>>  Raises AssertionError if sth goes wrong   Args: path_1 (Union[str, CloudPath, Path]): Raster 1 path_2 (Union[str, CloudPath, Path]): Raster 2",
"func":1
},
{
"ref":"sertit.ci.assert_raster_almost_equal",
"url":10,
"doc":"Assert that two rasters are almost equal. (everything is equal except the transform and the arrays that are almost equal) Accepts an offset of  1E{decimal} on the array and a precision of 10^-9 on the transform -> Useful for pytests.   >>> path = r\"CI\\DATA asters aster.tif\" >>> path2 = r\"CI\\DATA asters aster_almost.tif\" >>> assert_raster_equal(path, path2) >>>  Raises AssertionError if sth goes wrong   Args: path_1 (Union[str, CloudPath, Path]): Raster 1 path_2 (Union[str, CloudPath, Path]): Raster 2",
"func":1
},
{
"ref":"sertit.ci.assert_dir_equal",
"url":10,
"doc":"Assert that two directories are equal.  Useful for pytests.   >>> path = r\"CI\\DATA asters\" >>> assert_dir_equal(path, path) >>>  Raises AssertionError if sth goes wrong   Args: path_1 (str): Directory 1 path_2 (str): Directory 2",
"func":1
},
{
"ref":"sertit.ci.assert_geom_equal",
"url":10,
"doc":"Assert that two geometries are equal (do not check equality between geodataframe as they may differ on other fields). -> Useful for pytests.   >>> path = r\"CI\\DATA ectors\u0007oi.geojson\" >>> assert_geom_equal(path, path) >>>  Raises AssertionError if sth goes wrong    WARNING Only checks: - valid geometries - length of GeoDataFrame - CRS Args: geom_1 (gpd.GeoDataFrame): Geometry 1 geom_2 (gpd.GeoDataFrame): Geometry 2",
"func":1
},
{
"ref":"sertit.ci.assert_xml_equal",
"url":10,
"doc":"Assert that 2 XML (as etree Elements) are equal. -> Useful for pytests. Args: xml_elem_1 (etree._Element): 1st Element xml_elem_2 (etree._Element): 2nd Element",
"func":1
},
{
"ref":"sertit.snap",
"url":11,
"doc":"SNAP tools"
},
{
"ref":"sertit.snap.bytes2snap",
"url":11,
"doc":"Convert nof bytes into snap-compatible Java options.   >>> bytes2snap(32000) '31K'   Args: nof_bytes (int): Byte nb Returns: str: Human-readable in bits",
"func":1
},
{
"ref":"sertit.snap.get_gpt_cli",
"url":11,
"doc":"Get GPT command line with system OK optimizations. To see options, type this command line with  diag (but it won't run the graph)   >>> get_gpt_cli(\"graph_path\", other_args=[], display_snap_opt=True) SNAP Release version 8.0 SNAP home: C:\\Program Files\\snap\bin\\/ SNAP debug: null SNAP log level: WARNING Java home: c:\\program files\\snap\\jre\\jre Java version: 1.8.0_242 Processors: 16 Max memory: 53.3 GB Cache size: 30.0 GB Tile parallelism: 14 Tile size: 2048 x 2048 pixels To configure your gpt memory usage: Edit snap/bin/gpt.vmoptions To configure your gpt cache size and parallelism: Edit .snap/etc/snap.properties or gpt -c ${cachesize-in-GB}G -q ${parallelism} ['gpt', '\"graph_path\"', '-q', 14, '-J-Xms2G -J-Xmx60G', '-J-Dsnap.log.level=WARNING', '-J-Dsnap.jai.defaultTileSize=2048', '-J-Dsnap.dataio.reader.tileWidth=2048', '-J-Dsnap.dataio.reader.tileHeigh=2048', '-J-Dsnap.jai.prefetchTiles=true', '-c 30G']   Args: graph_path (str): Graph path other_args (list): Other args as a list such as  ['-Pfile=\"in_file.zip\", '-Pout=\"out_file.dim\"'] display_snap_opt (bool): Display SNAP options via  diag Returns: list: GPT command line as a list",
"func":1
}
]