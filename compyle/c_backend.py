from textwrap import dedent
import pybind11
from .translator import CConverter
from mako.template import Template
from .translator import ocl_detect_type, OpenCLConverter, KnownType
from .cython_generator import CythonGenerator, get_func_definition, getsourcelines

pybind11_wrap_fn = '''
PYBIND11_MODULE(${name}, m) {
    m.def("${name}", &${name}, "${doc}");
}
'''


class CBackend(CythonGenerator):
    def __init__(self, detect_type=ocl_detect_type, known_types=None):
        super(CBackend, self).__init__()
        # self.function_address_space = 'WITHIN_KERNEL '


    def get_func_signature_pyb11(self, func):
        sourcelines = getsourcelines(func)[0]
        defn, lines = get_func_definition(sourcelines)
        f_name, returns, args = self._analyze_method(func, lines)
        pyb11_args = []
        pyb11_call = []
        c_args = []
        c_call = []
        for arg, value in args:
            c_type = self.detect_type(arg, value)
            c_args.append('{type} {arg}'.format(type=c_type, arg=arg))

            c_call.append(arg)
            pyb11_type = self.ctype_to_pyb11(c_type)
            pyb11_args.append('{type} {arg}'.format(type=pyb11_type, arg=arg))
            if c_type.endswith('*'):
                pyb11_call.append('({ctype}){arg}.request().ptr'.format(arg = arg, ctype = c_type))
            else:
                pyb11_call.append('{arg}'.format(arg=arg))

        return (pyb11_args, pyb11_call), (c_args, c_call)

    def ctype_to_pyb11(self, c_type):
        if c_type[-1] == '*':
            return 'py::array_t<{}>'.format(c_type[:-1])
        else:
            return c_type

    def _get_self_type(self):
        return KnownType('GLOBAL_MEM %s*' % self._class_name)

