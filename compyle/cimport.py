import os
import io
import importlib
import shutil
import sys
from filelock import FileLock

from os.path import exists, expanduser, isdir, join

import pybind11
from distutils.extension import Extension
from distutils.command import build_ext
from setuptools import setup
from distutils.errors import CompileError, LinkError
from distutils.sysconfig import customize_compiler

from .ext_module import get_platform_dir, get_ext_extension, get_openmp_flags
from .capture_stream import CaptureMultipleStreams  # noqa: 402
from distutils.ccompiler import new_compiler


class Cmodule:
    def __init__(self, src, hash_fn, root=None, verbose=False, openmp=False,
                 extra_inc_dir=[pybind11.get_include()],
                 extra_link_args=[], extra_compile_args=[]):
        self.src = src
        self.hash = hash_fn
        self.name = f'm_{self.hash}'
        self.verbose = verbose
        self.openmp = openmp
        # self.extra_inc_dir = extra_inc_dir + ['/usr/include']
        self.extra_inc_dir = extra_inc_dir + [
        '/usr/include',
        '/usr/include/c++/{}'.format(self._get_gcc_version()),
        '/usr/lib/gcc/{}-linux-gnu/{}'.format(self._get_machine(), self._get_gcc_version())
    ]
        self.extra_link_args = extra_link_args
        self.extra_compile_args = extra_compile_args
        self._use_cpp11()

        self._setup_root(root)
        self._setup_filenames()
        self.lock = FileLock(self.lock_path, timeout=120)

    def _get_gcc_version(self):
        import subprocess
        try:
            gcc_version = subprocess.check_output(['gcc', '-dumpversion']).decode().strip()
            return gcc_version
        except:
            return '11'  # fallback to a common version

    def _get_machine(self):
        import platform
        return platform.machine()

    def _setup_root(self, root):
        if root is None:
            plat_dir = get_platform_dir()
            self.root = expanduser(join('~', '.compyle', 'source', plat_dir))
        else:
            self.root = root

        self.build_dir = join(self.root, 'build')

        if not isdir(self.build_dir):
            try:
                os.makedirs(self.build_dir)
            except OSError:
                pass

    def _write_source(self):
        if not exists(self.src_path):
            with io.open(self.src_path, 'w', encoding='utf-8') as f:
                f.write(self.src)

    def _setup_filenames(self):
        self.src_path = join(self.root, self.name + '.cpp')
        self.ext_path = join(self.root, self.name + get_ext_extension())
        self.lock_path = join(self.root, self.name + '.lock')

    def is_build_needed(self):
        return not exists(self.ext_path)

    def build(self):
        self._include_openmp()
        ext = Extension(
            name=self.name,
            sources=[self.src_path],
            language='c++',  # Change to c++ since we're using pybind11
            include_dirs=self.extra_inc_dir,
            extra_link_args=self.extra_link_args,
            extra_compile_args=self.extra_compile_args + [
                '-D_COMPLEX_H',
                '-D_GNU_SOURCE',
                '-fpermissive',  # Allow mixing C and C++ code
                '-include', 'complex.h',
                '-include', 'complex',
                '-x', 'c++',
            ]
        )
        args = [
            "build_ext",
            "--build-lib=" + self.build_dir,
            "--build-temp=" + self.build_dir,
            "-v",
        ]

        try:
            with CaptureMultipleStreams() as stream:
                setup(name=self.name,
                      ext_modules=[ext],
                      script_args=args,
                      cmdclass={"build_ext": build_ext.build_ext})
                shutil.move(join(self.build_dir, self.name +
                            get_ext_extension()), self.ext_path)

        except(CompileError, LinkError, SystemExit):
            hline = "*"*80
            print(hline + "\nERROR")
            s_out = stream.get_output()
            print(s_out[0])
            print(s_out[1])
            msg = "Compilation of code failed, please check "\
                "error messages above."
            print(hline + "\n" + msg)
            sys.exit(1)

    def write_and_build(self):
        """Write source and build the extension module"""
        if self.is_build_needed():
            with self.lock:
                self._write_source()
                self.build()
        else:
            self._message("Precompiled code from:", self.src_path)

    def load(self):
        self.write_and_build()
        spec = importlib.util.spec_from_file_location(self.name, self.ext_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _include_openmp(self):
        if self.openmp:
            ec, el = get_openmp_flags()
            self.extra_compile_args += ec
            self.extra_link_args += el

    def _use_cpp11(self):
        self.extra_compile_args += ['-std=c++11']

    def _message(self, *args):
        msg = ' '.join(args)
        if self.verbose:
            print(msg)


# def wget_tpnd_headers():
#     import requests
#     baseurl = 'https://gitlab.inria.fr/tapenade/tapenade/-/raw/master/ADFirstAidKit/{}?ref_type=heads&inline=false'
#     files = ['adComplex.h', 'adStack.c', 'adStack.h']
#     reqs = [requests.get(baseurl.format(file)) for file in files]
#     saveloc = get_tpnd_obj_dir()
#     if not os.path.exists(saveloc):
#         os.mkdir(saveloc)

#     for file, r in zip(files, reqs):
#         with open(join(saveloc, file), 'wb') as f:
#             f.write(r.content)

def wget_tpnd_headers():
    import requests
    "https://gitlab.inria.fr/tapenade/tapenade/-/raw/3.16-v2/ADFirstAidKit/adBuffer.c?inline=false"
    "https://gitlab.inria.fr/tapenade/tapenade/-/raw/3.16-v2/ADFirstAidKit/adBuffer.h?inline=false"
    baseurl = 'https://gitlab.inria.fr/tapenade/tapenade/-/raw/3.16-v2/ADFirstAidKit/{}?inline=false'
    files = ['adBuffer.c', 'adBuffer.h', 'adStack.c', 'adStack.h']
    saveloc = get_tpnd_obj_dir()
    os.makedirs(saveloc, exist_ok=True)

    print(f"Downloading Tapenade source code to {saveloc}")
    for file in files:
        r = requests.get(baseurl.format(file), timeout=30)
        r.raise_for_status()
        with open(join(saveloc, file), 'wb') as f:
            f.write(r.content)

def get_tpnd_obj_dir():
    plat_dir = get_platform_dir()
    root = expanduser(join('~', '.compyle', 'source', plat_dir))
    tpnd_dir = join(root, 'tapenade_src')
    return tpnd_dir


def compile_tapenade_source(verbose=0):
    print("Setting up Tapenade source code...")
    try:
        obj_dir_tpnd = get_tpnd_obj_dir()
        with CaptureMultipleStreams() as stream:
            wget_tpnd_headers()
            os.environ["CC"] = 'gcc'  # Use gcc instead of g++
            compiler = new_compiler(verbose=1)
            customize_compiler(compiler)
            compiler.compile(
                [join(obj_dir_tpnd, 'adStack.c'),
                 join(obj_dir_tpnd, 'adBuffer.c')],
                output_dir=obj_dir_tpnd,
                extra_preargs=[
                    '-c',
                    '-fPIC',
                ],
                include_dirs=[obj_dir_tpnd, '/usr/include'],
            )
            s_out = stream.get_output()
            print(s_out[0])
            print(s_out[1])
            objdir = join(obj_dir_tpnd, obj_dir_tpnd[1:])
            shutil.move(join(objdir, 'adStack.o'),
                        join(obj_dir_tpnd, 'adStack.o'))
            shutil.move(join(objdir, 'adBuffer.o'),
                        join(obj_dir_tpnd, 'adBuffer.o'))
    except (CompileError, LinkError):
        hline = "*"*80
        print(hline + "\nERROR")
        s_out = stream.get_output()
        print(s_out[0])
        print(s_out[1])
        msg = "Compilation of tapenade source failed, please check "\
              "error messages above."
        print(hline + "\n" + msg)
        sys.exit(1)
