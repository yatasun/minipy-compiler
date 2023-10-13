import os
import sys

# 没有 __init__.py, 所以用这种机制吗 ?
sys.path.append('../python-student-support-code')
sys.path.append('../python-student-support-code/interp_x86')
sys.path.append('../iucompiler')
sys.path.append('../iucompiler/interp_x86')

import compiler
import compiler_register_allocator
import interp_Lvar
import type_check_Lvar
from utils import run_tests, run_one_test, enable_tracing
from interp_x86.eval_x86 import interp_x86

enable_tracing()

# compiler = compiler.Compiler()
compiler = compiler_register_allocator.Compiler()

typecheck_Lvar = type_check_Lvar.TypeCheckLvar().type_check

typecheck_dict = {
    'source': typecheck_Lvar,
    'remove_complex_operands': typecheck_Lvar,
}
interpLvar = interp_Lvar.InterpLvar().interp
interp_dict = {
    'remove_complex_operands': interpLvar,
    'select_instructions': interp_x86,
    'assign_homes': interp_x86,
    'patch_instructions': interp_x86,
}

if True:
    run_tests('var', compiler, 'var',
              typecheck_dict,
              interp_dict)
else:
    run_one_test(os.getcwd() + '/tests/var/sub-input.py',
                 'var',
                 compiler,
                 'var',
                 typecheck_dict,
                 interp_dict)

