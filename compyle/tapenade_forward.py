from operator import mod
from tabnanny import verbose
import pybind11
import cppimport
import inspect
import os, subprocess
import functools
import re
import numpy as np
from mako.template import Template

from compyle.pushpop_c import PUSHPOP_C

from .profile import profile
from .translator import CConverter
from .c_backend import CBackend
from .transpiler import Transpiler, convert_to_float_if_needed
from . import array
from compyle.api import get_config

from .ext_module import ExtModule
from .cimport import Cmodule, get_tpnd_obj_dir, compile_tapenade_source

import compyle

# <%
# cfg['compiler_args'] = ['-std=c++11', '-fopenmp']
# cfg['linker_args'] = ['-fopenmp']
# setup_pybind11(cfg)
# %>
pyb11_setup_header = '''

// c code for with PyBind11 binding
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

\n
'''
# <%
# cfg['linker_args'] = ['/home/rohit/Documents/Pypr/compyle/compyle/tapenade_src/adBuffer.o', '/home/rohit/Documents/Pypr/compyle/compyle/tapenade_src/adStack.o']
# setup_pybind11(cfg)
# %>

pyb11_setup_header_rev = '''

// c code for with PyBind11 binding
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

\n
'''

pyb11_wrap_template = '''
PYBIND11_MODULE(${name}_fdiff, m) {
    m.def("${name}${FUNC_SUFFIX_F}", [](${pyb_call}){
        return ${name}${FUNC_SUFFIX_F}(${c_call});
    });
}
'''

pyb11_wrap_template_rev = '''
PYBIND11_MODULE(${name}_rdiff, m) {
    m.def("${name}${FUNC_SUFFIX_F}", [](${pyb_call}){
        return ${name}${FUNC_SUFFIX_F}(${c_call});
    });
}
'''

c_backend_template = '''
${c_kernel_defn}

void elementwise_${fn_name}(size_t SIZE, ${fn_args})
{
    %if openmp:
        #pragma omp parallel for
    %endif
    for(size_t iter = 0; iter < SIZE; iter++)
    {
        ${fn_name}(iter, ${fn_call}) ;
    }
}
'''
VAR_SUFFIX_F = '__d'
FUNC_SUFFIX_F = '_d'

VAR_SUFFIX_R = '__b'
FUNC_SUFFIX_R = '_b'


def get_source(f):
    c = CConverter()
    source = c.parse(f)
    return source


def sig_to_pyb_call(par, typ):
    if typ[-1] == "*":
        call = "py::array_t<{}> {}".format(typ[:-1], par)
    else:
        call = typ + " " + str(par)
    return call


def sig_to_c_call(par, typ):
    if typ[-1] == '*':
        call = "({ctype}) {arg}.request().ptr".format(ctype=typ, arg=par)
    else:
        call = "{arg}".format(arg=par)
    return call


def get_diff_signature(f, active, mode='forward'):
    if mode == 'forward':
        VAR_SUFFIX = VAR_SUFFIX_F
    elif mode == 'reverse':
        VAR_SUFFIX = VAR_SUFFIX_R

    sig = inspect.signature(f)
    pyb_c = []
    pyb_py = []
    pure_c = []
    pure_py = []

    for s in sig.parameters:
        typ = str(sig.parameters[s].annotation.type)
        if s not in active:
            pyb_py.append([sig_to_pyb_call(s, typ)])
            pyb_c.append([sig_to_c_call(s, typ)])
            pure_c.append(["{typ} {i}".format(typ=typ, i=s)])
            pure_py.append([s])
        else:
            typstar = typ if typ[-1] == '*' else typ + '*'
            pyb_py.append([
                sig_to_pyb_call(s, typ),
                sig_to_pyb_call(s + VAR_SUFFIX, typstar)
            ])
            pyb_c.append([
                sig_to_c_call(s, typ),
                sig_to_c_call(s + VAR_SUFFIX, typstar)
            ])
            pure_c.append([
                "{typ} {i}".format(typ=typ, i=s),
                "{typ} {i}".format(typ=typstar, i=s + VAR_SUFFIX)
            ])
            pure_py.append([s, s + VAR_SUFFIX])

    pyb_py_all = functools.reduce(lambda x, y: x + y, pyb_py)
    pyb_c_all = functools.reduce(lambda x, y: x + y, pyb_c)
    pure_c = functools.reduce(lambda x, y: x + y, pure_c)
    pure_py = functools.reduce(lambda x, y: x + y, pure_py)

    return pyb_py_all, pyb_c_all, pure_py, pure_c


class GradBase:
    def __init__(self, func, wrt, gradof, mode='forward', backend='tapenade'):
        self.backend = backend
        self.func = func
        self.args = None
        self.wrt = wrt
        self.gradof = gradof
        self.active = wrt + gradof
        self.mode = mode
        self.name = func.__name__
        self._config = get_config()
        self.source = 'Not yet generated'
        self.grad_source = 'Not yet generated'
        self.grad_all_source = 'Not yet generated'
        self.tapenade_op = 'Not yet generated'
        self.c_func = self.c_gen_error
        self._get_sources()

    def _get_sources(self):

        self.source = get_source(self.func)
        with open(self.name + '.c', 'w') as f:
            f.write(self.source)
            
        if self.mode == 'forward':
            command = [
                "tapenade", f"{self.name}.c", "-d", "-o",
                f"{self.name}_forward_diff", "-tgtvarname",
                f"{VAR_SUFFIX_F}", "-tgtfuncname",
                f"{FUNC_SUFFIX_F}",
                "-head", f'{self.name}({" ".join(self.wrt)})\({" ".join(self.gradof)})'
            ]
        elif self.mode == 'reverse':
            command = [
                "tapenade", f"{self.name}.c", "-b", "-o",
                f"{self.name}_reverse_diff", "-tgtvarname",
                f"{VAR_SUFFIX_R}", "-tgtfuncname",
                f"{FUNC_SUFFIX_R}",
                "-head", f'{self.name}({" ".join(self.wrt)})\({" ".join(self.gradof)})'
            ]
        
        op_tapenade = ""
        try:
            proc = subprocess.run(command, capture_output=True, text=True)
            op_tapenade += proc.stdout
        except:
            raise RuntimeError(
                "Encountered errors while differentiating through Tapenade.")
        self.tapenade_op = op_tapenade
        print(" ".join(command))

        if self.mode == 'forward':
            f_extn = "_forward_diff_d.c"
        elif self.mode == 'reverse':
            f_extn = "_reverse_diff_b.c"

        with open(self.name + f_extn, 'r') as f:
            self.grad_source = f.read()

    def c_gen_error(*args):
        raise RuntimeError("Differentiated function not yet generated")

    def _massage_arg(self, x):
        if isinstance(x, array.Array):
            return x.dev
        elif self.backend != 'cuda' or isinstance(x, np.ndarray):
            return x
        else:
            return np.asarray(x)

    @profile
    def __call__(self, *args, **kw):
        c_args = [self._massage_arg(x) for x in args]

        if self.backend == 'tapenade':
            self.c_func(*c_args)

        elif self.backend == 'cuda':
            import pycuda.driver as drv
            event = drv.Event()
            self.c_func(*c_args, **kw)
            event.record()
            event.synchronize()

        elif self.backend == 'c':
            self.c_func(*c_args)

        else:
            raise RuntimeError("Given backend not supported, got '{}'".format(
                self.backend))


class ForwardGrad(GradBase):
    def __init__(self, func, wrt, gradof):
        super(ForwardGrad, self).__init__(func,
                                           wrt,
                                           gradof,
                                           mode='forward',
                                           backend='tapenade')
        self.c_func = self.get_c_forward_diff()

    def get_c_forward_diff(self):
        self.grad_source = pyb11_setup_header + self.grad_source

        pyb_all, c_all, _, _ = get_diff_signature(self.func,
                                                  self.active,
                                                  mode='forward')
        pyb_call = ", ".join(pyb_all)
        c_call = ", ".join(c_all)

        pyb_temp = Template(pyb11_wrap_template)
        pyb_bind = pyb_temp.render(name=self.name,
                                   FUNC_SUFFIX_F=FUNC_SUFFIX_F,
                                   pyb_call=pyb_call,
                                   c_call=c_call)

        self.grad_all_source = self.grad_source + pyb_bind

        with open(self.name + "_fdiff.cpp", 'w') as f:
            f.write(self.grad_all_source)

        
        # module = cppimport.imp(self.name + '_fdiff')
        mod = Cmodule(self.name + '_fdiff', self.grad_all_source, extra_inc_dir=[pybind11.get_include()])
        return getattr(mod.load(), self.name + FUNC_SUFFIX_F)
        

class ElementwiseGrad(GradBase):
    def __init__(self, func, wrt, gradof, backend='c'):
        super(ElementwiseGrad, self).__init__(func,
                                               wrt,
                                               gradof,
                                               mode='forward',
                                               backend=backend)
        self._config = get_config()
        self.c_func = self._generate()

    def _generate(self):
        if self.backend == 'c':
            return self._c_gen()
        elif self.backend == 'cuda':
            return self._cuda_gen()

    def _c_gen(self):
        pyb_args, pyb_c_args, py_args, c_args = get_diff_signature(
            self.func, self.active)

        c_templt = Template(c_backend_template)
        c_code = c_templt.render(c_kernel_defn=self.grad_source,
                                 fn_name='{fname}{suff}'.format(
                                     fname=self.name, suff=FUNC_SUFFIX_F),
                                 fn_args=", ".join(c_args[1:]),
                                 fn_call=", ".join(py_args[1:]),
                                 openmp=self._config.use_openmp)
        pyb_templt = Template(pyb11_wrap_template)
        elwise_name = 'elementwise_' + self.name
        size = "{}.request().size".format(py_args[1])
        pyb_code = pyb_templt.render(name=elwise_name,
                                     FUNC_SUFFIX_F=FUNC_SUFFIX_F,
                                     pyb_call=", ".join(pyb_args[1:]),
                                     c_call=", ".join([size] + pyb_c_args[1:]))
        self.grad_all_source = pyb11_setup_header + c_code + pyb_code

        with open(elwise_name + "_fdiff.cpp", 'w') as f:
            f.write(self.grad_all_source)

        module = cppimport.imp(elwise_name + '_fdiff')
        return getattr(module, elwise_name + FUNC_SUFFIX_F)

    def _cuda_gen(self):
        from .cuda import set_context
        set_context()
        from pycuda.elementwise import ElementwiseKernel
        from pycuda._cluda import CLUDA_PREAMBLE

        _, _, py_args, c_args = get_diff_signature(self.func, self.active)

        self.grad_source = self.convert_to_device_code(self.grad_source)
        expr = '{func}({args})'.format(func=self.name + FUNC_SUFFIX_F,
                                       args=", ".join(py_args))

        arguments = convert_to_float_if_needed(", ".join(c_args[1:]))
        preamble = convert_to_float_if_needed(self.grad_source)

        cluda_preamble = Template(text=CLUDA_PREAMBLE).render(
            double_support=True)
        knl = ElementwiseKernel(name=self.name,
                                arguments=arguments,
                                operation=expr,
                                preamble="\n".join([cluda_preamble, preamble]))
        self.grad_all_source = cluda_preamble + preamble
        return knl

    def convert_to_device_code(self, code):
        code = re.sub(r'\bvoid\b', 'WITHIN_KERNEL void', code)
        code = re.sub(r'\bfloat\b', 'GLOBAL_MEM float ', code)
        code = re.sub(r'\bdouble\b', 'GLOBAL_MEM double ', code)
        return code


class ReverseGrad(GradBase):
    def __init__(self, func, wrt, gradof, backend='tapenade'):
        super().__init__(func, wrt, gradof, mode='reverse', backend=backend)
        self.c_func = self._c_reverse_diff()

    def _c_reverse_diff(self):
        from .pushpop_c import PUSHPOP_C
        self.grad_source = pyb11_setup_header_rev + self.grad_source

        pyb_all, c_all, self.args, _ = get_diff_signature(self.func,
                                                  self.active,
                                                  mode=self.mode)
        pyb_call = ", ".join(pyb_all)
        c_call = ", ".join(c_all)

        pyb_temp = Template(pyb11_wrap_template_rev)
        pyb_bind = pyb_temp.render(name=self.name,
                                   FUNC_SUFFIX_F=FUNC_SUFFIX_R,
                                   pyb_call=pyb_call,
                                   c_call=c_call)

        self.grad_all_source = self.grad_source + pyb_bind


        with open(self.name + "_rdiff.cpp", 'w') as f:
            f.write(self.grad_all_source)
            
        if not os.path.exists(os.path.join(get_tpnd_obj_dir(), 'adBuffer.o')) or not os.path.exists(os.path.join(get_tpnd_obj_dir(), 'adStack.o')):
            compile_tapenade_source()
            
        mod = Cmodule(self.name + "_rdiff", self.grad_all_source, extra_inc_dir=[pybind11.get_include(), get_tpnd_obj_dir(), '/home/rohit/Documents/Pypr/compyle/compyle/tapenade_src'], extra_link_args=[os.path.join(get_tpnd_obj_dir(), 'adBuffer.o'), os.path.join(get_tpnd_obj_dir(), 'adStack.o')])
        module = mod.load()
        
        return getattr(module, self.name + FUNC_SUFFIX_R)
    

class Grad(ReverseGrad):
    def __init__(self, func, wrt, gradof, backend='tapenade'):
        super().__init__(func, wrt, gradof, backend=backend)        
            
