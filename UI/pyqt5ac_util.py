import pyqt5ac

ioPaths = [
    ["UI/*.ui", "PY/%%FILENAME%%.py"],
    ["UI/*.qrc", "PY/%%FILENAME%%_rc.py"],
]

pyqt5ac.main(
    rccOptions="",
    uicOptions="--from-imports",
    force=False,
    config="",
    ioPaths=ioPaths,
    variables=None,
    initPackage=True,
)
