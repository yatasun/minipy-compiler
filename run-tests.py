import os
import sys

# 没有 __init__.py, 所以用这种机制吗 ?
sys.path.append('../iucompiler')
sys.path.append('../iucompiler/interp_x86')

import compiler
import compiler_if
import compiler_while
import interp_Lvar
import interp_Lif
import interp_Cif
import interp_Lwhile
import type_check_Lvar
import type_check_Lif
import type_check_Lwhile
from utils import run_tests, run_one_test, enable_tracing
from interp_x86.eval_x86 import interp_x86

enable_tracing()

compiler = compiler.Compiler()
compiler = compiler_if.Compiler()
compiler = compiler_while.Compiler()

typecheck_Lvar = type_check_Lvar.TypeCheckLvar().type_check
typecheck_Lif = type_check_Lif.TypeCheckLif().type_check
typecheck_Lwhile = type_check_Lwhile.TypeCheckLwhile().type_check

typecheck_dict = {
    'source': typecheck_Lwhile,
    'shrink': typecheck_Lwhile,
    'remove_complex_operands': typecheck_Lwhile,
    'explicate_control': typecheck_Lwhile
}
interpLvar = interp_Lvar.InterpLvar().interp
interpLif = interp_Lif.InterpLif().interp
interpCif = interp_Cif.InterpCif().interp
interpLwhile = interp_Lwhile.InterpLwhile().interp
interp_dict = {
    'shrink': interpLwhile,
    'remove_complex_operands': interpLwhile,
    'explicate_control': interpCif,
    'select_instructions': interp_x86,
    'assign_homes': interp_x86,
    'patch_instructions': interp_x86,
    'prelude_and_conclusion': interp_x86,
}

if True:
    # run_tests('var', compiler, 'var',
    #           typecheck_dict,
    #           interp_dict)
    # run_tests('if', compiler, 'if',
    #           typecheck_dict,
    #           interp_dict)
    run_tests('while', compiler, 'while',
              typecheck_dict,
              interp_dict)
else:
    run_one_test(os.getcwd() + '/tests/if/my4.py',
                 'if',
                 compiler,
                 'if',
                 typecheck_dict,
                 interp_dict)

