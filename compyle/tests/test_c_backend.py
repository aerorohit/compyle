from operator import ge
import pytest
import numpy as np

# from ..array import Array
# from ..config import get_config
import compyle.array as array
from compyle.api import annotate, wrap, Elementwise, get_config, declare
from compyle.transpiler import Transpiler

get_config().use_openmp = True

@annotate(i = 'int', doublep = 'x, y, ans')
def add(i, x, y, ans):
    """
    Add two integers x and y
    return: integer (x + y)
    """
    ans[i] = x[i] + y[i]

@annotate(x = 'doublep', n = 'int', return_='double')
def sum(x, n):
    ans = 0
    i = declare('int')
    for i in range(n):
        ans += x[i]

    return ans

@annotate(int = 'i, n' , doublep = 'x, ans')
def addar(i, x, ans, n):
    j = declare('int')
    for j in range(n):
        ans[i] = sum(x, n)


if __name__ == '__main__':
    n = 1000
    x = np.ones(n, dtype = np.float64)
    y = np.arange(n, dtype=np.float64)
    ans = np.empty(n, dtype=np.float64)
    x, y, ans = wrap(x, y, ans, backend = 'cython')
    # x, y, ans = wrap(x, y, ans, backend = 'cuda ')

    e = Elementwise(addar, backend='C')
    print(e.all_source)
    

    # ecy = Elementwise(add, backend='cython')
    # ecy(x, y, ans)

    
    # e(y, ans, n)
    # print(ans)
np.sum(x)



















# tp = Transpiler(backend=backend)
# tp.add(add_annotated)
# code = tp.get_code()
# tp.add_code()

# print(code)
# for i in tp.blocks:
#     print(i.code)
#     print('#'*80)
#     print('#'*80)
# if __name__ == '__main__':
#     backend = 'C'
#     x = array.ones(10, np.int32, backend=backend)
#     print(x)