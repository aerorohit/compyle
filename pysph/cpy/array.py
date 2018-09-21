import numpy as np

from .config import get_config


def get_backend(backend=None):
    if not backend:
        cfg = get_config()
        if cfg.use_opencl:
            return 'opencl'
        elif cfg.use_cuda:
            return 'cuda'
        else:
            return 'cython'
    else:
        return backend


def wrap(*args, **kw):
    '''
    Parameters
    ----------

    *args: any numpy arrays to be wrapped.

    **kw: only one keyword arg called `backend` is supported.

    backend: str: use appropriate backend for arrays.
    '''
    backend = get_backend(kw.get('backend'))
    if len(args) == 1:
        return Array(ary=args[0], backend=backend)
    else:
        return [Array(ary=x, backend=backend) for x in args]


def ones_like(array, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        dev_array = 1 + gpuarray.zeros_like(array)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        dev_array = gpuarray.ones_like(array)
    else:
        return Array(np.ones_like(array))
    wrapped_array = Array()
    wrapped_array.set_dev_array(dev_array)
    return wrapped_array


def ones(n, dtype, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        dev_array = 1 + gpuarray.zeros(get_queue(), n, dtype)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        dev_array = np.array(1, dtype=dtype) + gpuarray.zeros(n, dtype)
    else:
        return Array(np.ones(n, dtype=dtype))
    wrapped_array = Array()
    wrapped_array.set_dev_array(dev_array)
    return wrapped_array


def empty(n, dtype, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        return Array(dev_ary=gpuarray.empty(get_queue(), n, dtype))
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        return Array(dev_ary=gpuarray.empty(n, dtype))
    else:
        return Array(ary=np.empty(n, dtype=dtype))


def zeros(n, dtype, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        return Array(dev_ary=gpuarray.zeros(get_queue(), n, dtype))
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        return Array(dev_ary=gpuarray.zeros(n, dtype))
    else:
        return Array(ary=np.zeros(n, dtype=dtype))


def arange(start, stop, step, dtype=None, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        return Array(dev_ary=gpuarray.arange(get_queue(),
                     start, stop,step, dtype=dtype))
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        return Array(dev_ary=gpuarray.arange(start,
                     stop, step, dtype=dtype))
    else:
        return Array(np.arange(start, stop, step, dtype=dtype))


def minimum(ary, backend='cython'):
    if backend == 'cython':
        return ary.dev.min()
    elif backend == 'opencl':
        import pyopencl.array as gpuarray
        return gpuarray.min(ary.dev).get()
    elif backend == 'cuda':
        import pycuda.array as gpuarray
        return gpuarray.min(ary.dev).get()


def maximum(ary, backend='cython'):
    if backend == 'cython':
        return ary.dev.max()
    elif backend == 'opencl':
        import pyopencl.array as gpuarray
        return gpuarray.max(ary.dev).get()
    elif backend == 'cuda':
        import pycuda.array as gpuarray
        return gpuarray.max(ary.dev).get()


def take(ary, indices, backend='cython'):
    if backend == 'opencl':
        import pyopencl.array as gpuarray
        out = gpuarray.take(ary.dev, indices.dev)
    elif backend == 'cuda':
        import pycuda.gpuarray as gpuarray
        out = gpuarray.take(ary.dev, indices.dev)
    elif backend == 'cython':
        return Array(ary=np.take(ary.dev, indices.dev))
    return Array(dev_ary=out)


class Array(object):
    """A simple wrapper for numpy arrays.

    It has two attributes,

    `data` is the raw numpy array.

    `dev` is the device array if needed.

    Use the `pull()` method to get the data from device.
    Use `push()` to push the data to the device.

    """
    def __init__(self, ary=None, dev_ary=None, backend=None):
        self.backend = get_backend(backend)
        self.data = ary
        self._convert = False
        if self.data is not None:
            self._initialize()
        else:
            if dev_ary is not None:
                self.dev = dev_ary
            else:
                self.dev = None

    def _initialize(self):
        if self.backend == 'opencl' or self.backend == 'cuda':
            use_double = get_config().use_double
            self._dtype = np.float64 if use_double else np.float32
            if np.issubdtype(self.data.dtype, np.float):
                self._convert = True
            self.q = None
            if self.backend == 'opencl':
                from .opencl import get_queue
                from pyopencl.array import to_device
                self.q = get_queue()
                if self.data is not None:
                    self.dev = to_device(self.q, self._get_data())
            elif self.backend == 'cuda':
                from .cuda import set_context
                set_context()
                from pycuda.gpuarray import to_gpu
                if self.data is not None:
                    self.dev = to_gpu(self._get_data())
            else:
                self.dev = None
        else:
            self.dev = self.data

    def _get_data(self):
        if self._convert:
            return self.data.astype(self._dtype)
        else:
            return self.data

    def set_dev_array(self, dev_array):
        self.dev = dev_array

    def pull(self):
        if self.backend == 'opencl' or self.backend == 'cuda':
            if self.data is None:
                self.data = np.empty(len(self.dev), dtype=self._dtype)
            self.data[:] = self.dev.get()

    def push(self):
        if self.backend == 'opencl' or self.backend == 'cuda':
            self.dev.set(self._get_data())


