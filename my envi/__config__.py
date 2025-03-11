# This file is generated by SciPy's build process
# It contains system_info results at the time of building this package.
from enum import Enum

__all__ = ["show"]
_built_with_meson = True


class DisplayModes(Enum):
    stdout = "stdout"
    dicts = "dicts"


def _cleanup(d):
    """
    Removes empty values in a `dict` recursively
    This ensures we remove values that Meson could not provide to CONFIG
    """
    if isinstance(d, dict):
        return { k: _cleanup(v) for k, v in d.items() if v != '' and _cleanup(v) != '' }
    else:
        return d


CONFIG = _cleanup(
    {
        "Compilers": {
            "c": {
                "name": "gcc",
                "linker": r"ld.bfd",
                "version": "10.3.0",
                "commands": r"cc",
                "args": r"",
                "linker args": r"",
            },
            "cython": {
                "name": r"cython",
                "linker": r"cython",
                "version": r"3.0.12",
                "commands": r"cython",
                "args": r"",
                "linker args": r"",
            },
            "c++": {
                "name": "gcc",
                "linker": r"ld.bfd",
                "version": "10.3.0",
                "commands": r"c++",
                "args": r"",
                "linker args": r"",
            },
            "fortran": {
                "name": "gcc",
                "linker": r"ld.bfd",
                "version": "10.3.0",
                "commands": r"gfortran",
                "args": r"",
                "linker args": r"",
            },
            "pythran": {
                "version": r"0.17.0",
                "include directory": r"C:\Users\runneradmin\AppData\Local\Temp\pip-build-env-vnvsvx_3\overlay\Lib\site-packages/pythran"
            },
        },
        "Machine Information": {
            "host": {
                "cpu": r"x86_64",
                "family": r"x86_64",
                "endian": r"little",
                "system": r"windows",
            },
            "build": {
                "cpu": r"x86_64",
                "family": r"x86_64",
                "endian": r"little",
                "system": r"windows",
            },
            "cross-compiled": bool("False".lower().replace('false', '')),
        },
        "Build Dependencies": {
            "blas": {
                "name": "scipy-openblas",
                "found": bool("True".lower().replace('false', '')),
                "version": "0.3.28",
                "detection method": "pkgconfig",
                "include directory": r"C:/Users/runneradmin/AppData/Local/Temp/cibw-run-yfa1j2v9/cp313-win_amd64/build/venv/Lib/site-packages/scipy_openblas32/include",
                "lib directory": r"C:/Users/runneradmin/AppData/Local/Temp/cibw-run-yfa1j2v9/cp313-win_amd64/build/venv/Lib/site-packages/scipy_openblas32/lib",
                "openblas configuration": r"OpenBLAS 0.3.28 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24",
                "pc file directory": r"D:/a/scipy/scipy",
            },
            "lapack": {
                "name": "scipy-openblas",
                "found": bool("True".lower().replace('false', '')),
                "version": "0.3.28",
                "detection method": "pkgconfig",
                "include directory": r"C:/Users/runneradmin/AppData/Local/Temp/cibw-run-yfa1j2v9/cp313-win_amd64/build/venv/Lib/site-packages/scipy_openblas32/include",
                "lib directory": r"C:/Users/runneradmin/AppData/Local/Temp/cibw-run-yfa1j2v9/cp313-win_amd64/build/venv/Lib/site-packages/scipy_openblas32/lib",
                "openblas configuration": r"OpenBLAS 0.3.28 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24",
                "pc file directory": r"D:/a/scipy/scipy",
            },
            "pybind11": {
                "name": "pybind11",
                "version": "2.13.6",
                "detection method": "config-tool",
                "include directory": r"unknown",
            },
        },
        "Python Information": {
            "path": r"C:\Users\runneradmin\AppData\Local\Temp\cibw-run-yfa1j2v9\cp313-win_amd64\build\venv\Scripts\python.exe",
            "version": "3.13",
        },
    }
)


def _check_pyyaml():
    import yaml

    return yaml


def show(mode=DisplayModes.stdout.value):
    """
    Show libraries and system information on which SciPy was built
    and is being used

    Parameters
    ----------
    mode : {`'stdout'`, `'dicts'`}, optional.
        Indicates how to display the config information.
        `'stdout'` prints to console, `'dicts'` returns a dictionary
        of the configuration.

    Returns
    -------
    out : {`dict`, `None`}
        If mode is `'dicts'`, a dict is returned, else None

    Notes
    -----
    1. The `'stdout'` mode will give more readable
       output if ``pyyaml`` is installed

    """
    if mode == DisplayModes.stdout.value:
        try:  # Non-standard library, check import
            yaml = _check_pyyaml()

            print(yaml.dump(CONFIG))
        except ModuleNotFoundError:
            import warnings
            import json

            warnings.warn("Install `pyyaml` for better output", stacklevel=1)
            print(json.dumps(CONFIG, indent=2))
    elif mode == DisplayModes.dicts.value:
        return CONFIG
    else:
        raise AttributeError(
            f"Invalid `mode`, use one of: {', '.join([e.value for e in DisplayModes])}"
        )
