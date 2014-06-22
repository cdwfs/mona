import os
import os.path
import shutil
import subprocess
import sys
import zipfile

if __name__ == "__main__":
    dist_dir = "mona-win32"
    dist_zip_path = dist_dir + ".zip"

    # Start from a clean slate!
    if os.path.exists(dist_zip_path):
        os.unlink(dist_zip_path)
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    # Create distribution directory
    os.makedirs(dist_dir)

    # Build executables
    msvc_tools_dir = os.getenv("VS100COMNTOOLS")
    if msvc_tools_dir is None:
        raise RuntimeError("Visual Studio 10.0 not detected!")
    devenv_path = msvc_tools_dir + "..\\IDE\\devenv.com"
    rebuildCmd = [devenv_path, "src/mona.sln", "/projectconfig", "Release", "/rebuild"]
    returnCode = subprocess.call(rebuildCmd)
    if returnCode != 0:
        raise RuntimeError("build failed!")
    # Add executable
    shutil.copyfile("mona.exe", os.path.join(dist_dir, "mona.exe"))
    shutil.copyfile("minimona.exe", os.path.join(dist_dir, "minimona.exe"))

    # Copy distributable files into dist_dir
    qt_dll_dir = "C:\\Qt\\5.3\\msvc2010_opengl\\bin"
    msvc_dll_dir = os.path.normpath(os.path.join(msvc_tools_dir, "..\\..\\VC\\redist\\x86\\Microsoft.VC100.CRT"))
    dist_paths = [
        # QT DLLs
        os.path.join(qt_dll_dir, "icudt52.dll"),
        os.path.join(qt_dll_dir, "icuin52.dll"),
        os.path.join(qt_dll_dir, "icuuc52.dll"),
        os.path.join(qt_dll_dir, "Qt5Core.dll"),
        os.path.join(qt_dll_dir, "Qt5Gui.dll"),
        os.path.join(qt_dll_dir, "Qt5OpenGL.dll"),
        os.path.join(qt_dll_dir, "Qt5Widgets.dll"),
        # Visual Studio DLLs
        os.path.join(msvc_dll_dir, "msvcp100.dll"),
        os.path.join(msvc_dll_dir, "msvcr100.dll"),
        # TODO: Add README, licenses, etc.
        ]
    print("\nCopying files to %s..." % dist_dir)
    for path in dist_paths:
        print("\t%s" % path)
        shutil.copyfile(path, os.path.join(dist_dir, os.path.split(path)[1]))

    # Create zipfile
    print("\nCompressing to %s..." % dist_zip_path)
    if sys.version_info.major >= 3:
        compressMethod = zipfile.ZIP_LZMA
    else:
        compressMethod = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(dist_zip_path, "w", compression=compressMethod) as dist_zip:
        for f in os.listdir(dist_dir):
            print("\t%s" % f)
            dist_zip.write(os.path.join(dist_dir, f))
    print("Wrote %d bytes to %s" % (os.path.getsize(dist_zip_path), dist_zip_path))
