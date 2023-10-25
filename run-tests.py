import os
import sys

# 没有 __init__.py, 所以用这种机制吗 ?
sys.path.append('../iucompiler')
sys.path.append('../iucompiler/interp_x86')

import compiler
import compiler_if
import compiler_while
import compiler_tup
import compiler_fun
import interp_Lvar
import interp_Lif
import interp_Cif
import interp_Lwhile
import interp_Ltup
import interp_Ctup
import type_check_Lvar
import type_check_Lif
import type_check_Lwhile
import type_check_Ltup
from utils import run_tests, run_one_test, enable_tracing
from interp_x86.eval_x86 import interp_x86

enable_tracing()

compiler = compiler.Compiler()
compiler = compiler_if.Compiler()
compiler = compiler_while.Compiler()
compiler = compiler_tup.Compiler()
compiler = compiler_fun.Compiler()

typecheck_Lvar = type_check_Lvar.TypeCheckLvar().type_check
typecheck_Lif = type_check_Lif.TypeCheckLif().type_check
typecheck_Lwhile = type_check_Lwhile.TypeCheckLwhile().type_check
typecheck_Ltup = type_check_Ltup.TypeCheckLtup().type_check

typecheck_dict = {
    'source': typecheck_Ltup,
    'shrink': typecheck_Ltup,
    'expose_allocation': typecheck_Ltup,
    'remove_complex_operands': typecheck_Ltup,
    'explicate_control': typecheck_Ltup,
}
interpLvar = interp_Lvar.InterpLvar().interp
interpLif = interp_Lif.InterpLif().interp
interpCif = interp_Cif.InterpCif().interp
interpLwhile = interp_Lwhile.InterpLwhile().interp
interpLtup = interp_Ltup.InterpLtup().interp
interpCtup = interp_Ctup.InterpCtup().interp
interp_dict = {
    'shrink': interpLtup,
    'expose_allocation': interpLtup,
    'remove_complex_operands': interpLtup,
    'explicate_control': interpCtup,
    'select_instructions': interp_x86,
    'assign_homes': interp_x86,
    'patch_instructions': interp_x86,
    'prelude_and_conclusion': interp_x86,
}

if True:
    run_tests('var', compiler, 'var',
              typecheck_dict,
              interp_dict)
    run_tests('if', compiler, 'if',
              typecheck_dict,
              interp_dict)
    run_tests('while', compiler, 'while',
              typecheck_dict,
              interp_dict)
    run_tests('tup', compiler, 'tup',
              typecheck_dict,
              interp_dict)
else:
    run_one_test(os.getcwd() + '/tests/tup/my2.py',
                 'tup',
                 compiler,
                 'tup',
                 typecheck_dict,
                 interp_dict)

