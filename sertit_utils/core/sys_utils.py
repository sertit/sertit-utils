""" System tools """
import os
import platform
import sys
import logging
import subprocess
from contextlib import contextmanager
from typing import Union
from sertit_utils.core.log_utils import SU_NAME

LOGGER = logging.getLogger(SU_NAME)


def __set_env_var_proj_lib__():
    """ Set-up PROJ_LIB environment variables """
    platform_string = platform.platform()
    if 'Windows' in platform_string:
        if 'CONDA_PREFIX' in os.environ.keys():
            os.environ['PROJ_LIB'] = os.path.join(os.environ['CONDA_PREFIX'], 'Library\\share\\proj')
        else:
            os.environ['PROJ_LIB'] = os.path.join('..', 'Library\\share\\proj')
    elif 'Linux' in platform_string:
        if 'CONDA_PREFIX' in os.environ.keys():
            os.environ['PROJ_LIB'] = os.path.join(os.environ['CONDA_PREFIX'], 'share/proj')
        else:
            os.environ['PROJ_LIB'] = os.path.join('..', 'share/proj')


def setup_environ():
    """ Set-up PROJ_LIB environment variables """
    if 'PROJ_LIB' not in os.environ:
        __set_env_var_proj_lib__()
    LOGGER.debug("Env var '%s' is set to '%s'", 'PROJ_LIB', os.environ['PROJ_LIB'])


def run_command(cmd: Union[str, list],
                timeout: float = None,
                check_return_value: bool = True,
                in_background: bool = True,
                cwd='/') -> (int, str):
    """
    Run a command line.

    ```python
    cmd_hillshade = ["gdaldem", "--config",
                     "NUM_THREADS", "1",
                     "hillshade", type_utils.to_cmd_string(dem_path),
                     "-compute_edges",
                     "-z", self.nof_threads,
                     "-az", azimuth,
                     "-alt", zenith,
                     "-of", "GTiff",
                     type_utils.to_cmd_string(hillshade_dem)]
    # Run command
    sys_utils.run_command(cmd_hillshade)
    ```

    Args:
        cmd (str or list[str]): Command as a list
        timeout (float): Timeout
        check_return_value (bool): Check output value of the exe
        in_background (bool): Run the subprocess in background
        cwd (str): Working directory

    Returns:
        int, str: return value and output log
    """
    if isinstance(cmd, list):
        cmd = [str(cmd_i) for cmd_i in cmd]
        cmd_line = ' '.join(cmd)
    elif isinstance(cmd, str):
        cmd_line = cmd
    else:
        raise TypeError('The command line should be given as a str or a list')

    # Background
    LOGGER.debug(cmd_line)
    if in_background:
        stdout = None
        stderr = None
        close_fds = True
    else:
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
        close_fds = False

    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    with subprocess.Popen(cmd_line,
                          shell=True,
                          stdout=stdout,
                          stderr=stderr,
                          cwd=cwd,
                          start_new_session=True,
                          close_fds=close_fds) as process:
        output = ''
        if not in_background:
            for line in process.stdout:
                line = line.decode(encoding=sys.stdout.encoding,
                                   errors='replace' if sys.version_info < (3, 5)
                                   else 'backslashreplace').rstrip()
                LOGGER.info(line)
                output += line

        # Get return value
        retval = process.wait(timeout)

        # Kill process
        process.kill()

    # Check return value
    if check_return_value and retval != 0:
        raise Exception("Exe {} has failed.".format(cmd[0]))

    return retval, output


def get_function_name() -> str:
    """
    Get the name of the function where this one is launched.

    Returns:
        str: Function's name
    """
    # pylint: disable=W0212
    return sys._getframe(1).f_code.co_name


def in_docker():
    """
    Check if the session is running inside a docker

    Returns:
        bool: True if inside a docker

    """
    try:
        with open('/proc/1/cgroup', 'rt') as ifh:
            in_dck = 'docker' in ifh.read()
    # pylint: disable=W0703
    except Exception:
        in_dck = False

    return in_dck


@contextmanager
def chdir(newdir: str) -> None:
    """
    Change current directory as a context manage, ie:

    ```python
    from sertit_utils.core import sys_utils
    folder = "."
    with chdir(folder):
        # Current directory
        pwd = os.getcwd()
    ```

    Args:
        newdir (str): New directory
    """
    prevdir = os.getcwd()
    newdir = newdir.replace("\\", "/")
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
