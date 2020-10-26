""" Script testing the sys_utils """
import os
from sertit_utils.core import sys_utils


def test_setup_environ():
    """ Test setup_environ """
    sys_utils.setup_environ()  # Just ensure no exception is thrown


def test_run_command():
    """ Test run_command """
    cmd = ["cd", ".."]
    sys_utils.run_command(cmd, in_background=True, cwd='/')  # Just ensure no exception is thrown


def test_get_function_name():
    """ Test get_function_name """
    assert sys_utils.get_function_name() == "test_get_function_name"


def test_in_docker():
    """ Test in_docker """
    # Hack: in docker if ni linux
    sys_utils.in_docker()  # Just hope it doesn't crash


def test_chdir():
    """ Testing chdir functions """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    old_pwd = os.getcwd()
    with sys_utils.chdir(curr_dir):
        pwd = os.getcwd()
        assert pwd == curr_dir

    assert os.getcwd() == old_pwd
