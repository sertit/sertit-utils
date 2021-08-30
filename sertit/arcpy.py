# flake8: noqa
def init_conda_arcpy_env():
    """
    Initialize conda environment with Arcgis Pro
    """
    # Try importing lxml
    try:
        from lxml import etree
    except ImportError:
        import os
        import sys

        if "python" in sys.executable:
            root_dir = os.path.dirname(sys.executable)
        else:
            import subprocess

            try:
                conda_env_list = subprocess.run(
                    "conda env list", capture_output=True, shell=True, encoding="UTF-8"
                ).stdout
                conda_env_list = conda_env_list.split("\n")
                curr_env = [env for env in conda_env_list if "*" in env][0]
                root_dir = [elem for elem in curr_env.split(" ") if elem][-1]
            except Exception:
                raise ImportError(
                    "Cannot import lxml. Please try 'pip uninstall lxml -y' then 'pip install lxml'."
                )

        os.environ["PATH"] = root_dir + r"\Library\bin;"
        print(f"Missing lxml DLLs. Completing PATH: {os.environ['PATH']}")
        from lxml import etree  # Try again
