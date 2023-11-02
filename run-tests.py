import os
import sys

sys.path.append('../minipy-compiler')
sys.path.append('../minipy-compiler/interp_x86')

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
import interp_Lfun
import interp_Cfun
import type_check_Lvar
import type_check_Lif
import type_check_Lwhile
import type_check_Ltup
import type_check_Lfun
import type_check_Cfun
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
typecheck_Lfun = type_check_Lfun.TypeCheckLfun().type_check
typecheck_Cfun = type_check_Cfun.TypeCheckCfun().type_check

typecheck_dict = {
    'source': typecheck_Lfun,
    'shrink': typecheck_Lfun,
    'reveal_functions': typecheck_Lfun,
    'limit_functions': typecheck_Lfun,
    'expose_allocation': typecheck_Lfun,
    'remove_complex_operands': typecheck_Lfun,
    'explicate_control': typecheck_Cfun,
}
interpLvar = interp_Lvar.InterpLvar().interp
interpLif = interp_Lif.InterpLif().interp
interpCif = interp_Cif.InterpCif().interp
interpLwhile = interp_Lwhile.InterpLwhile().interp
interpLtup = interp_Ltup.InterpLtup().interp
interpCtup = interp_Ctup.InterpCtup().interp
interpLfun = interp_Lfun.InterpLfun().interp
interpCfun = interp_Cfun.InterpCfun().interp
interp_dict = {
    'shrink': interpLfun,
    'reveal_functions': interpLfun,
    'limit_functions': interpLfun,
    'expose_allocation': interpLfun,
    'remove_complex_operands': interpLfun,
    'explicate_control': interpCfun,
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
    # run_tests('while', compiler, 'while',
    #           typecheck_dict,
    #           interp_dict)
    # run_tests('tup', compiler, 'tup',
    #           typecheck_dict,
    #           interp_dict)
    run_tests('fun', compiler, 'fun',
              typecheck_dict,
              interp_dict)
else:
    run_one_test(os.getcwd() + '/tests/fun/my3.py',
                 'fun',
                 compiler,
                 'fun',
                 typecheck_dict,
                 interp_dict)

