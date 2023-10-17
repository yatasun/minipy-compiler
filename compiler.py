import ast
from ast import *
from utils import *
from x86_ast import *
import os
from typing import List, Tuple, Set, Dict
from functools import reduce

Binding = Tuple[Name, expr]
Temporaries = List[Binding]


class Compiler:
    ############################################################################
    # Remove Complex Operands
    ############################################################################

    # L_Var -> L_Var^mon

    def is_atomic(self, e: expr) -> bool:
        match e:
            case Constant(_) | Name(_):
                return True
            case _:
                return False

    # need_atomic == True: 则返回的是 (Name(tmp), [...]) 的形式
    def rco_exp(self, e: expr, need_atomic: bool) -> Tuple[expr, Temporaries]:
        result_expr = expr()
        result_temps = []
        match e:
            case Constant(_) | Name(_):
                return (e, [])

            case Call(Name("input_int"), []):
                result_expr, result_temps = e, []

            case UnaryOp(op, exp):
                # 先处理 subexpression
                (new_exp, temps) = self.rco_exp(exp, True)
                result_expr, result_temps = UnaryOp(op, new_exp), temps

            case BinOp(exp1, op, exp2):
                (new_exp1, temps1) = self.rco_exp(exp1, True)
                (new_exp2, temps2) = self.rco_exp(exp2, True)
                result_expr, result_temps = (
                    BinOp(new_exp1, op, new_exp2),
                    temps1 + temps2,
                )

            case _:
                raise Exception("rco_exp unexpected: " + repr(e))

        if need_atomic:
            tmp = generate_name("tmp")
            result_expr, result_temps = Name(tmp), result_temps + [
                (Name(tmp), result_expr)
            ]

        return result_expr, result_temps

    def rco_stmt(self, s: stmt) -> List[stmt]:
        match s:
            case Expr(Call(Name("print"), [atm])):
                new_exp, temps = self.rco_exp(atm, True)
                return make_assigns(temps) + [Expr(Call(Name("print"), [new_exp]))]  # type: ignore
            case Expr(exp):
                new_exp, temps = self.rco_exp(exp, False)
                return make_assigns(temps) + [Expr(new_exp)]  # type: ignore
            case Assign([Name(var)], exp):
                new_exp, temps = self.rco_exp(exp, False)
                return make_assigns(temps) + [Assign([Name(var)], new_exp)]  # type: ignore
            case _:
                raise Exception("rco_stmt unexpected: " + repr(s))

    def remove_complex_operands(self, p: Module) -> Module:
        match p:
            case Module(body):
                return Module([new_s for s in body for new_s in self.rco_stmt(s)])
            case _:
                raise Exception("remove_complex_operands unexpected: " + repr(p))

    ############################################################################
    # Select Instructions
    ############################################################################

    # L_Var^mon -> x86_Var

    def select_arg(self, e: expr) -> arg:
        match e:
            case Constant(n):
                return Immediate(n)
            case Name(v):
                return Variable(v)
            case _:
                raise Exception("select_arg unexpected: " + repr(e))

    def select_stmt(self, s: stmt) -> List[instr]:
        match s:
            case Expr(Call(Name("print"), [atm])):
                arg = self.select_arg(atm)
                return [
                    Instr("movq", [arg, Reg("rdi")]),
                    Callq(label_name("print_int"), 1),
                ]
            # side effect
            case Expr(Call(Name("input_int"), [])):
                return [Callq(label_name("read_int"), 0)]
            case Assign([Name(_)], _):
                return self.select_stmt_assign(s)
            case _:
                raise Exception("select_stmt unexpected: " + repr(s))

    def select_stmt_assign(self, s: Assign) -> List[instr]:
        match s:
            case Assign([Name(var)], exp):
                match exp:
                    case Constant(_) | Name(_) as atm:
                        return [Instr("movq", [self.select_arg(atm), Variable(var)])]
                    case UnaryOp(USub(), atm):
                        return [
                            Instr("movq", [self.select_arg(atm), Variable(var)]),
                            Instr("negq", [Variable(var)]),
                        ]
                    case BinOp(atm1, Add(), atm2):
                        arg1, arg2 = self.select_arg(atm1), self.select_arg(atm2)
                        lhs = Variable(var)
                        if lhs == arg1:
                            return [Instr("addq", [arg2, lhs])]
                        elif lhs == arg2:
                            return [Instr("addq", [arg1, lhs])]
                        else:
                            return [
                                Instr("movq", [arg1, lhs]),
                                Instr("addq", [arg2, lhs]),
                            ]
                    case BinOp(atm1, Sub(), atm2):
                        arg1, arg2 = self.select_arg(atm1), self.select_arg(atm2)
                        lhs = Variable(var)
                        if lhs == arg1:
                            return [Instr("subq", [arg2, lhs])]
                        elif lhs == arg2:
                            return [Instr("negq", [lhs]), Instr("addq", [arg1, lhs])]
                        else:
                            return [
                                Instr("movq", [arg1, lhs]),
                                Instr("subq", [arg2, lhs]),
                            ]
                    case Call(Name("input_int"), []):
                        return [
                            Callq(label_name("read_int"), 0),
                            Instr("movq", [Reg("rax"), Variable(var)]),
                        ]

                    case _:
                        raise Exception(
                            "select_stmt_assign unexpected exp: " + repr(exp)
                        )
            case _:
                raise Exception("select_stmt_assign unexpected: " + repr(s))

    def select_instructions(self, p: Module) -> X86Program:
        return X86Program(
            reduce(
                lambda acc, curr: acc + curr, [self.select_stmt(s) for s in p.body], []
            )
        )

    ############################################################################
    # Assign Homes
    ############################################################################

    # x86_Var -> x86_Int

    def assign_homes_arg(self, a: arg, home: Dict[Variable, arg]) -> arg:
        match a:
            case Immediate(_):
                return a
            case Reg(_):
                return a
            case Variable(_) as var:
                if var in home:
                    return home[var]
                else:
                    self.spilled_size += 8
                    home[var] = Deref("rbp", -self.spilled_size)
                    return home[var]
            case _:
                raise Exception("assign_homes_arg unexpected " + repr(a))

    def assign_homes_instr(self, i: instr, home: Dict[Variable, arg]) -> instr:
        match i:
            case Instr(op, [arg]):
                return Instr(op, [self.assign_homes_arg(arg, home)])
            case Instr(op, [arg1, arg2]):
                return Instr(
                    op,
                    [
                        self.assign_homes_arg(arg1, home),
                        self.assign_homes_arg(arg2, home),
                    ],
                )
            case Callq(_, _):
                return i
            case _:
                return i
                # raise Exception("assign_homes_instr unexpected " + repr(i))

    def assign_homes(self, p: X86Program) -> X86Program:
        self.spilled_size = 0
        home = {}
        return X86Program([self.assign_homes_instr(i, home) for i in p.body])  # type: ignore

    ############################################################################
    # Patch Instructions
    ############################################################################

    def patch_instr(self, i: instr) -> List[instr]:
        match i:
            case Instr(op, [Immediate(n) as imm]) if n > 2**16:
                return [Instr("movq", [imm, Reg("rax")]), Instr(op, [Reg("rax")])]
            case Instr(op, [_]):
                return [i]
            case Instr(op, [arg1, arg2]):
                match (arg1, arg2):
                    case (Deref(_, _) as deref1, Deref(_, _) as deref2):
                        return [
                            Instr("movq", [deref1, Reg("rax")]),
                            Instr(op, [Reg("rax"), deref2]),
                        ]
                    case (Immediate(n) as imm, _) if n > 2**16:
                        return [
                            Instr("movq", [imm, Reg("rax")]),
                            Instr(op, [Reg("rax"), arg2]),
                        ]
                    # TODO: 存在这种情况吗？
                    case (_, Immediate(n) as imm) if n > 2**16:
                        return [
                            Instr("movq", [imm, Reg("rax")]),
                            Instr(op, [arg1, Reg("rax")]),
                        ]
                    case _:
                        return [i]
            case Callq(_, _):
                return [i]
            case _:
                return [i]
                # raise Exception("patch_instr unexpected: " + repr(i))

    def patch_instructions(self, p: X86Program) -> X86Program:
        return X86Program([ni for i in p.body for ni in self.patch_instr(i)])  # type: ignore

    ############################################################################
    # Prelude & Conclusion
    ############################################################################

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        new_body = (
            [
                Instr("pushq", [Reg("rbp")]),
                Instr("movq", [Reg("rsp"), Reg("rbp")]),
                # return address + pushq rbp 已经使得 rsp 按 16 byte 对齐.
                Instr(
                    "subq", [Immediate(align(self.spilled_size, 16)), Reg("rsp")]
                ),
            ]
            + p.body  # type: ignore
            + [
                Instr(
                    "addq", [Immediate(align(self.spilled_size, 16)), Reg("rsp")]
                ),
                Instr("popq", [Reg("rbp")]),
                Instr("retq", []),
            ]
        )
        return X86Program(new_body)
