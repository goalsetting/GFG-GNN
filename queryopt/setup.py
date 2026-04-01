#!/usr/bin/env python
from distutils.core import setup, Extension
import numpy as np
from pathlib import Path

source_dir = Path("/home/user/GFGGNN/queryopt")
example_module = Extension('_pyabcore',
    sources=[        str(source_dir / 'pyabcore.cpp'),
        str(source_dir / 'pyabcore_wrap.cxx'),
        str(source_dir / 'bigraph.cpp'),
        str(source_dir / 'kcore.cpp')  ],
   include_dirs=['/home/user/GFDN/queryopt/boost_1_81_0'],  # 头文件路径
   library_dirs=['/home/user/GFDN/queryopt/boost_1_81_0/stage/lib'],  # 库文件路径
   # libraries=['boost_system-vc143-mt-x64-1_84'],  # 具体库名（根据实际编译结果调整）
   # extra_compile_args=['/EHsc']  # 针对MSVC的编译选项
    extra_compile_args=['-fexceptions']  # 启用C++异常处理

)

setup (
    name = 'pyabcore',
    version = '0.1',
    author = "yujianke",
    description = """alpha beta core for python""",
    ext_modules = [example_module],
    py_modules = ["pyabcore"],
    include_dirs=[np.get_include()]
)