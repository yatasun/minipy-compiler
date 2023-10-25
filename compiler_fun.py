from ast import Assign, Dict, List, Module, Set, arg, expr, stmt, cmpop, Tuple
from dataflow_analysis import analyze_dataflow, build_cfg
from graph import (
    UndirectedAdjList,
    transpose,
)
from priority_queue import PriorityQueue
from type_check_Ctup import TypeCheckCtup
from type_check_Ltup import TypeCheckLtup
from utils import Assign, Dict, List, Module, Set, arg, expr, stmt
from utils import *
from typing import List, Tuple, Set, Dict

from x86_ast import *
import x86_ast
from x86_ast import X86Program, arg, instr, location, Global

Binding = Tuple[Name, expr]
Temporaries = List[Binding]

def cmpop_to_cc(cmp: cmpop) -> str:
    match cmp:
        case Eq():
            return "e"
        case NotEq():
            return "ne"
        case Lt():
            return "l"
        case LtE():
            return "le"
        case Gt():
            return "g"
        case GtE():
            return "ge"
        # L_Tup
        case Is(): 
            return "e"
        case _:
            raise Exception("cmpop_to_cc unexpected: " + repr(cmp))

caller_saved_regs = ["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"]
callee_saved_regs = ["rsp", "rbp", "rbx", "r12", "r13", "r14", "r15"]
# argument_passing_regs is a subset of caller_saved_regs
arg_passing_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]

id_to_regs = {
    # 0: "rcx",
    # 1: "rdx",
    # 2: "rsi",
    # 3: "rdi",
    # 4: "r8",
    # 5: "r9",
    # 6: "r10",
    # 7: "rbx",
    # 8: "r12",
    # 9: "r13",
    # 10: "r14",

    0: "rbx",
    1: "r12",
    2: "r13",
    3: "r14",
}
not_reg_alloc_regs = {
    -1: "rax",
    -2: "rsp",
    -3: "rbp",
    -4: "r11",
    -5: "r15",
    -6: "rcx",
    -7: "rdx",
    -8: "rsi",
    -9: "rdi",
    -10: "r8",
    -11: "r9",
    -12: "r10",
}


def id_to_reg(i: int) -> str:
    if i in id_to_regs:
        return id_to_regs[i]
    if i in not_reg_alloc_regs:
        return not_reg_alloc_regs[i]
    raise Exception(f"id_to_reg: invalid id: {i}")


def reg_to_id(r: str) -> int:
    for k, v in id_to_regs.items():
        if v == r:
            return k
    for k, v in not_reg_alloc_regs.items():
        if v == r:
            return k
    raise Exception(f"reg_to_id: invalid reg: {r}")


def is_not_reg_alloc_reg(r: str) -> bool:
    return r in not_reg_alloc_regs.values()


def islocation(a: arg) -> bool:
    match a:
        case Variable(_) | Reg(_) | ByteReg(_):
            return True
        case _:
            return False


def arg_to_locations(a: arg) -> Set[location]:
    match a:
        case Variable(_) | Reg(_) | ByteReg(_):
            return {a}
        # FIXME: Deref 似乎不算 location, 但是 build_interference 需要用
        # case Deref(r, _):
        #     return {Reg(r)}
        case _:
            return set()

class Compiler:
    ############################################################################
    # Shrink
    ############################################################################

    def shrink_exp(self, e: expr) -> expr:
        match e:
            case Constant(_) | Name(_) | Call(Name("input_int"), []):
                return e
            case UnaryOp(op, exp):
                return UnaryOp(op, self.shrink_exp(exp))
            case BinOp(exp1, op, exp2):
                return BinOp(self.shrink_exp(exp1), op, self.shrink_exp(exp2))

            # desugar `and`, `or`
            case BoolOp(And(), [exp1, exp2]):
                return IfExp(
                    self.shrink_exp(exp1), self.shrink_exp(exp2), Constant(False)
                )
            case BoolOp(Or(), [exp1, exp2]):
                return IfExp(
                    self.shrink_exp(exp1), Constant(True), self.shrink_exp(exp2)
                )

            case IfExp(exp1, exp2, exp3):
                return IfExp(
                    self.shrink_exp(exp1), self.shrink_exp(exp2), self.shrink_exp(exp3)
                )
            case Compare(left, [cmp], [right]):
                return Compare(self.shrink_exp(left), [cmp], [self.shrink_exp(right)])
            
            case ast.Tuple(exps, Load()):
                return ast.Tuple([self.shrink_exp(e) for e in exps], Load())
            case Subscript(exp, Constant(n), Load()):
                return Subscript(self.shrink_exp(exp), Constant(n), Load())
            case Call(Name("len"), [exp]):
                return Call(Name("len"), [self.shrink_exp(exp)])
            
            case _:
                raise Exception("shrink_exp unexpected: " + repr(e))

    def shrink_stmt(self, s: stmt) -> stmt:
        match s:
            # 处理 subexpression
            case Expr(Call(Name("print"), [exp])):
                return Expr(Call(Name("print"), [self.shrink_exp(exp)]))
            case Expr(e):
                return Expr(self.shrink_exp(e))
            case Assign([Name(var)], exp):
                return Assign([Name(var)], self.shrink_exp(exp))
            case If(exp, body, orelse):
                return If(
                    self.shrink_exp(exp),
                    [self.shrink_stmt(ss) for ss in body],
                    [self.shrink_stmt(ss) for ss in orelse],
                )
            case While(test, body, []):
                new_body = [self.shrink_stmt(s) for s in body]
                return While(self.shrink_exp(test), new_body, [])
            case _:
                raise Exception("shrink_stmt unexpected: " + repr(s))

    def shrink(self, p: Module) -> Module:
        return Module([self.shrink_stmt(s) for s in p.body])

    ############################################################################
    # Expose Allocation
    ############################################################################

    def expose_exp(self, e: expr) -> expr:
        match e:
            case UnaryOp(op, exp):
                return UnaryOp(op, self.expose_exp(exp))
            case BinOp(left, op, right):
                return BinOp(self.expose_exp(left), op, self.expose_exp(right))
            case BoolOp(op, [exp1, exp2]):
                return BoolOp(op, [self.expose_exp(exp1), self.expose_exp(exp2)])
            case Compare(exp1, [cmp], [exp2]):
                return Compare(self.expose_exp(exp1), [cmp], [self.expose_exp(exp2)])
            case IfExp(exp1, exp2, exp3):
                return IfExp(
                    self.expose_exp(exp1), self.expose_exp(exp2), self.expose_exp(exp3)
                )
            case ast.Tuple(exps, Load()):
                new_exps = [self.expose_exp(e) for e in exps]
                assigns: list[Assign] = []
                for complex_e in new_exps:
                    tmp = generate_name("init")
                    assigns.append(Assign([Name(tmp)], self.expose_exp(complex_e)))

                # tag + 8 * len(exps)
                alloc_size = 8 + 8 * len(exps)

                test_l = BinOp(
                    GlobalValue(label_name("free_ptr")), Add(), Constant(alloc_size)
                )
                test_r = GlobalValue(label_name("fromspace_end"))
                test = Compare(test_l, [Lt()], [test_r])
                checks = If(test, [], [Collect(alloc_size)])

                result = Name(generate_name("alloc"))
                alloc = Assign(
                    [result], Allocate(len(exps), e.has_type)
                )  # has_type 在 type checker 收集
                sub_assigns = []
                for i, tmp_assign in enumerate(assigns):
                    sub_assigns.append(
                        Assign(
                            [Subscript(result, Constant(i), Store())],
                            tmp_assign.targets[0],
                        )
                    )

                return Begin(assigns + [checks, alloc] + sub_assigns, result)  # type: ignore
            case Subscript(exp, Constant(n), Load()):
                return Subscript(self.expose_exp(exp), Constant(n), Load())
            case _:
                return e

    def expose_stmt(self, s: stmt) -> stmt:
        match s:
            case Expr(Call(Name("print"), [exp])):
                return Expr(Call(Name("print"), [self.expose_exp(exp)]))
            case Expr(exp):
                return Expr(self.expose_exp(exp))
            case Assign([Name(v)], exp):
                return Assign([Name(v)], self.expose_exp(exp))
            case Assign([Subscript(tup, Constant(i), Store())], rhs):
                return Assign(
                    [Subscript(self.expose_exp(tup), Constant(i), Store())],
                    self.expose_exp(rhs),
                )
            case If(test, body, orelse):
                return If(
                    self.expose_exp(test),
                    [self.expose_stmt(s) for s in body],
                    [self.expose_stmt(s) for s in orelse],
                )
            case While(test, body, []):
                return While(
                    self.expose_exp(test), [self.expose_stmt(s) for s in body], []
                )
            case _:
                raise Exception("expose_stmt unexpected: " + repr(s))

    def expose_allocation(self, p: Module) -> Module:
        TypeCheckLtup().type_check(p)  # 收集 has_type
        assert isinstance(p.body, list)
        return Module([self.expose_stmt(s) for s in p.body])

    ############################################################################
    # Remove Complex Operands
    ############################################################################

    def isatomic(self, e: expr) -> bool:
        match e:
            case Constant(_) | Name(_) | GlobalValue(_):
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

            case Compare(left, [cmp], [right]):
                (new_left, temps1) = self.rco_exp(left, True)
                (new_right, temps2) = self.rco_exp(right, True)
                result_expr, result_temps = (
                    Compare(new_left, [cmp], [new_right]),
                    temps1 + temps2,
                )
            case IfExp(test, body, orelse):
                (new_test, temps1) = self.rco_exp(test, False)
                (new_body, temps2) = self.rco_exp(body, False)
                (new_orelse, temps3) = self.rco_exp(orelse, False)

                # Begin 适合处理嵌套 if expression 的情况.
                then_branch = make_begin(temps2, new_body)
                else_branch = make_begin(temps3, new_orelse)
                result_expr, result_temps = (
                    IfExp(new_test, then_branch, else_branch),
                    temps1,
                )

            case GlobalValue(_):
                result_expr, result_temps = e, []
            # L_If 并没有处理 Begin
            case Begin(body, ret):
                new_body = [ns for s in body for ns in self.rco_stmt(s)]
                new_ret, new_temps = self.rco_exp(ret, False)
                result_expr, result_temps = (
                    Begin(new_body + make_assigns(new_temps), new_ret),
                    [],
                )
            case Allocate(_, _):
                result_expr, result_temps = e, []
            case Subscript(exp1, exp2, ctx):
                new_exp1, new_temps1 = self.rco_exp(exp1, True)
                new_exp2, new_temps2 = self.rco_exp(exp2, True)
                result_expr, result_temps = (
                    Subscript(new_exp1, new_exp2, ctx),
                    new_temps1 + new_temps2,
                )
            case Call(Name("len"), [exp]):
                new_exp, new_temps = self.rco_exp(exp, True)
                result_expr, result_temps = Call(Name("len"), [new_exp]), new_temps

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

            case If(exp, body, orelse):
                new_exp, temps = self.rco_exp(exp, False)
                new_body = [new_ss for ss in body for new_ss in self.rco_stmt(ss)]
                new_orelse = [new_ss for ss in orelse for new_ss in self.rco_stmt(ss)]
                return make_assigns(temps) + [If(new_exp, new_body, new_orelse)]  # type: ignore
            case While(test, body, []):
                new_test, temps = self.rco_exp(test, False)
                new_body = [new_s for s in body for new_s in self.rco_stmt(s)]
                return make_assigns(temps) + [While(new_test, new_body, [])] # type: ignore

            case Assign([Subscript(lhs, index, Store())], rhs):
                lhs_atm, lhs_temps = self.rco_exp(lhs, True)
                index_atm, index_temps = self.rco_exp(index, True)
                rhs_atm, rhs_temps = self.rco_exp(rhs, True)
                return make_assigns(lhs_temps + index_temps + rhs_temps) + [Assign([Subscript(lhs_atm, index_atm, Store())], rhs_atm)]  # type: ignore
            case Collect(_):
                return [s]
            case _:
                raise Exception("rco_stmt unexpected: " + repr(s))

    def remove_complex_operands(self, p: Module) -> Module:
        match p:
            case Module(body):
                return Module([new_s for s in body for new_s in self.rco_stmt(s)])
            case _:
                raise Exception("remove_complex_operands unexpected: " + repr(p))

    
    ############################################################################
    # Explicate Control
    ############################################################################

    def create_block(
        self, stmts: List[stmt], basic_block: Dict[str, List[stmt]]
    ) -> List[stmt]:
        match stmts:
            case [Goto(l)]:
                return stmts
            case _:
                label = label_name(generate_name("block"))
                basic_block[label] = stmts
                return [Goto(label)]

    # generates code for an if expression or statement by analyzing the condition expression
    # 返回 `stmt` ... If(Compare(atm, [cmp], [atm]), [Goto(label)], [Goto(label)])
    def explicate_pred(
        self, cnd: expr, thn: List[stmt], els: List[stmt], basic_blocks: Dict[str, List[stmt]]
    ) -> List[stmt]:
        match cnd:
            case Compare(left, [op], [right]):
                goto_thn = self.create_block(thn, basic_blocks)
                goto_els = self.create_block(els, basic_blocks)
                return [If(cnd, goto_thn, goto_els)]
            case Constant(True):
                return thn
            case Constant(False):
                return els
            case UnaryOp(Not(), operand):
                return self.explicate_pred(operand, els, thn, basic_blocks)
            # y+2 if (x==0 if x<1 else x==2) else y+10
            case IfExp(test, body, orelse) as if_exp:
                goto_thn = self.create_block(thn, basic_blocks)
                goto_els = self.create_block(els, basic_blocks)
                body_ss = self.explicate_pred(body, goto_thn, goto_els, basic_blocks)
                orelse_ss = self.explicate_pred(
                    orelse, goto_thn, goto_els, basic_blocks
                )
                return self.explicate_pred(test, body_ss, orelse_ss, basic_blocks)
            # if Begin([res=100], res)
            case Begin(body, result):
                result_ss = self.explicate_pred(result, thn, els, basic_blocks)
                for s in reversed(body):
                    result_ss = self.explicate_stmt(s, result_ss, basic_blocks)
                return result_ss

            case Subscript(tup, index, Load()):
                tmp = generate_name("tmp")
                return self.explicate_assign(
                    cnd,
                    Name(tmp),
                    self.explicate_pred(Name(tmp), thn, els, basic_blocks),
                    basic_blocks,
                )

            case _:
                return [
                    If(
                        Compare(cnd, [Eq()], [Constant(False)]),
                        self.create_block(els, basic_blocks),
                        self.create_block(thn, basic_blocks),
                    )
                ]

    # generates code for expressions as statments, so their result is ignored and only their side effects matter
    # 处理 Expr(e) 的 e, 过滤掉纯的表达式
    # 返回 C_If stmts
    def explicate_effect(
        self, e: expr, cont: List[stmt], basic_blocks: Dict[str, List[stmt]]
    ) -> List[stmt]:
        match e:
            case Begin(body, result):
                begin_block = self.explicate_effect(result, cont, basic_blocks) + cont
                for s in reversed(body):
                    begin_block = self.explicate_stmt(s, begin_block, basic_blocks)
                return begin_block
            case IfExp(test, body, orelse):
                assert isinstance(body, Begin)
                assert isinstance(orelse, Begin)
                goto_cont = self.create_block(cont, basic_blocks)
                body_ss = self.explicate_effect(body, goto_cont, basic_blocks)
                orelse_ss = self.explicate_effect(orelse, goto_cont, basic_blocks)
                return self.explicate_pred(test, body_ss, orelse_ss, basic_blocks)
            case Call(func, args):
                return [Expr(e)] + cont

            case GlobalValue(_) | Allocate(_, _):
                return cont
            case _:
                return cont

    # generates code for expressions on the right-hand side of an assignment
    # 处理 Assign statement
    # 返回 C_If stmts
    def explicate_assign(
        self,
        rhs: expr,
        lhs: expr,
        cont: List[stmt],
        basic_blocks: Dict[str, List[stmt]],
    ) -> List[stmt]:
        match rhs:
            case IfExp(test, body, orelse):
                goto_cont = self.create_block(cont, basic_blocks)
                body_assign = self.explicate_assign(body, lhs, goto_cont, basic_blocks)
                orelse_assign = self.explicate_assign(
                    orelse, lhs, goto_cont, basic_blocks
                )
                return self.explicate_pred(
                    test, body_assign, orelse_assign, basic_blocks
                )

            case Begin(body, result):
                result_ss = self.explicate_assign(result, lhs, cont, basic_blocks)
                for s in reversed(body):
                    result_ss = self.explicate_stmt(s, result_ss, basic_blocks)
                return result_ss
            case _:
                return [Assign([lhs], rhs)] + cont

    # generates code for statements
    def explicate_stmt(
        self, s: stmt, cont: List[stmt], basic_blocks: Dict[str, List[stmt]]
    ) -> List[stmt]:
        match s:
            case Collect(_) | Assign([Subscript(_, _, Store())], _):
                return [s] + cont
            case Assign([lhs], rhs):
                return self.explicate_assign(rhs, lhs, cont, basic_blocks)
            # expression-statement
            case Expr(value):
                return self.explicate_effect(value, cont, basic_blocks)
            case If(test, body, orelse):
                goto_cont = self.create_block(cont, basic_blocks)
                body_ss = goto_cont
                for s in reversed(body):
                    body_ss = self.explicate_stmt(s, body_ss, basic_blocks)
                orelse_ss = goto_cont
                for s in reversed(orelse):
                    orelse_ss = self.explicate_stmt(s, orelse_ss, basic_blocks)
                return self.explicate_pred(test, body_ss, orelse_ss, basic_blocks)

            case While(test, body, []):
                goto_cont = self.create_block(cont, basic_blocks)

                label = label_name(generate_name('loop'))

                body_ss: list[stmt] = [Goto(label)]
                for s in reversed(body):
                    body_ss = self.explicate_stmt(s, body_ss, basic_blocks)

                test_ss =  self.explicate_pred(test, body_ss, goto_cont, basic_blocks)
                basic_blocks[label] = test_ss
                return [Goto(label)]

            case _:
                raise Exception("explicate_stmt unexpected: " + repr(s))

    # Expr(Call(Name('print'), [Name('tmp.0')]))
    def explicate_control(self, p: X86Program) -> CProgram:
        match p:
            case Module(body):
                basic_blocks: dict[str, list[stmt]] = {}
                new_body: list[stmt] = [Return(Constant(0))]
                # 倒序编译
                for s in reversed(body):
                    new_body = self.explicate_stmt(s, new_body, basic_blocks)
                basic_blocks[label_name("start")] = new_body
                return CProgram(basic_blocks)
            case _:
                raise Exception("explicate_control unexpected: " + repr(p))


    ############################################################################
    # Select Instructions
    ############################################################################

    def select_arg(self, e: expr) -> arg:
        match e:
            case Constant(True):
                return Immediate(1)
            case Constant(False):
                return Immediate(0)
            case Constant(n) if isinstance(n, int) and not isinstance(n, bool):
                return Immediate(n)
            case Name(v):
                return Variable(v)
            case GlobalValue(gv):
                return Global(gv)
            case _:
                raise Exception("select_arg unexpected: " + repr(e))

    def select_stmt(self, s: stmt) -> List[instr]:
        match s:
            # L_Tup
            case Assign([Name(var)], GlobalValue(gv)):
                return [Instr("movq", [Global(gv), Variable(var)])]  # type: ignore
            case Assign([Name(var)], Subscript(tup, Constant(n), Load())):
                var_arg = self.select_arg(Name(var))
                tup_arg = self.select_arg(tup)
                return [
                    Instr("movq", [tup_arg, Reg("r11")]),
                    Instr("movq", [Deref("r11", 8 * (n + 1)), var_arg]),
                ]
            case Assign([Subscript(tup, Constant(n), Store())], rhs):
                tup_arg = self.select_arg(tup)
                rhs_arg = self.select_arg(rhs)
                return [
                    Instr("movq", [tup_arg, Reg("r11")]),
                    Instr("movq", [rhs_arg, Deref("r11", 8 * (n + 1))]),
                ]
            case Assign([Name(var)], Call(Name("len"), [tup])):
                tup_arg = self.select_arg(tup)
                mask = 0b1111110
                var_arg = self.select_arg(Name(var))
                return [
                    Instr("movq", [tup_arg, Reg("r11")]),
                    Instr("movq", [Deref("r11", 0), Reg("r11")]),
                    Instr("andq", [Immediate(mask), Reg("r11")]),
                    Instr("sarq", [Immediate(1), Reg("r11")]),
                    Instr("movq", [Reg("r11"), var_arg]),
                ]
            case Assign([Name(var)], Allocate(n, TupleType(tup_types))):
                var_arg = self.select_arg(Name(var))
                pointer_mask = 0
                for i in range(len(tup_types)):
                    if isinstance(tup_types[i], TupleType):
                        pointer_mask |= 1 << i
                tag = (pointer_mask << 7) + (n << 1) + 1
                return [
                    Instr("movq", [Global(label_name("free_ptr")), Reg("r11")]),  # type: ignore
                    Instr("addq", [Immediate(8 * (n + 1)), Global(label_name("free_ptr"))]),  # type: ignore
                    Instr("movq", [Immediate(tag), Deref("r11", 0)]),
                    Instr("movq", [Reg("r11"), var_arg]),
                ]

            case Collect(n):
                return [
                    Instr("movq", [Reg("r15"), Reg("rdi")]),
                    Instr("movq", [Immediate(n), Reg("rsi")]),
                    Callq(label_name("collect"), 2),
                ]

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

            case Goto(label):
                return [Jump(label)]
            case If(Compare(left_atm, [cmp], [right_atm]), [Goto(thn)], [Goto(els)]):
                leaft_arg = self.select_arg(left_atm)
                right_arg = self.select_arg(right_atm)
                cc = cmpop_to_cc(cmp)
                return [
                    Instr("cmpq", [right_arg, leaft_arg]),
                    JumpIf(cc, thn),
                    Jump(els),
                ]
            # TODO: type error
            case Return(exp):
                return [
                    Instr("movq", [self.select_arg(exp), Reg("rax")]),  # type: ignore
                    Jump(label_name("conclusion")),
                ]
            case _:
                raise Exception("select_stmt unexpected: " + repr(s))

    def select_stmt_assign(self, s: Assign) -> List[instr]:
        match s:
            case Assign([Name(var)], UnaryOp(Not(), Name(var2))) if var == var2:
                return [Instr("xorq", [Immediate(1), Variable(var2)])]
            case Assign([Name(var)], UnaryOp(Not(), atm)):
                arg = self.select_arg(atm)
                return [
                    Instr("movq", [arg, Variable(var)]),
                    Instr("xorq", [Immediate(1), Variable(var)]),
                ]
            case Assign([Name(var)], Compare(left_atm, [cmp], [right_atm])):
                left_arg = self.select_arg(left_atm)
                right_arg = self.select_arg(right_atm)
                cc = cmpop_to_cc(cmp)
                return [
                    Instr("cmpq", [right_arg, left_arg]),
                    Instr(f"set{cc}", [ByteReg("al")]),
                    Instr("movzbq", [ByteReg("al"), Variable(var)]),
                ]

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
        TypeCheckCtup().type_check(p)
        self.var_types = p.var_types  # type: ignore

        new_body = {}
        for bb, stmts in p.body.items():  # type: ignore
            instrs = [i for s in stmts for i in self.select_stmt(s)]
            new_body[bb] = instrs
        return X86Program(new_body)
    
    ###########################################################################
    # Uncover Live
    ###########################################################################
    def read_vars(self, i: instr) -> Set[location]:
        match i:
            case Instr("movq", [arg1, _]):
                return arg_to_locations(arg1)
            case Instr("addq", [arg1, arg2]) | Instr("subq", [arg1, arg2]):
                return arg_to_locations(arg1) | arg_to_locations(arg2)
            case Instr("negq", [arg]):
                return arg_to_locations(arg)
            case Instr("pushq", [arg]):
                return arg_to_locations(arg)
            case Callq(_, narg):
                return set([Reg(r) for r in arg_passing_regs[:narg]])

            case Instr("xorq", [arg1, arg2]):
                return arg_to_locations(arg1) | arg_to_locations(arg2)
            case Instr("cmpq", [arg1, arg2]):
                return arg_to_locations(arg1) | arg_to_locations(arg2)
            case Instr("movzbq", [arg1, _]):
                return arg_to_locations(arg1)
            case Instr(setcc, [_]) if setcc.startswith("set"):
                return set()  # 不用考虑  EFLAGS
            case Jump(_) | JumpIf(_, _):
                return set()

            case Instr("andq", [s, d]):
                return arg_to_locations(s) | arg_to_locations(d)
            case Instr("sarq", [s, d]):
                return arg_to_locations(s) | arg_to_locations(d)
            case _:
                return set()

    def write_vars(self, i: instr) -> Set[location]:
        match i:
            case Instr("movq", [_, arg2]):
                return arg_to_locations(arg2)
            case Instr("addq", [_, arg2]) | Instr("subq", [_, arg2]):
                return arg_to_locations(arg2)
            case Instr("negq", [arg]):
                return arg_to_locations(arg)
            case Instr("popq", [arg]):
                return arg_to_locations(arg)
            case Callq(_, _):
                return set([Reg(r) for r in caller_saved_regs])

            case Instr("xorq", [_, arg2]):
                return arg_to_locations(arg2)
            case Instr("cmpq", [_, arg2]):
                return set()  # 不用考虑 EFLAGS
            case Instr("movzbq", [_, arg2]):
                return arg_to_locations(arg2)
            case Instr(setcc, [arg]) if setcc.startswith("set"):
                return arg_to_locations(arg)
            case Jump(_) | JumpIf(_, _):
                return set()

            case Instr("andq", [s, d]):
                return arg_to_locations(d)
            case Instr("sarq", [s, d]):
                return arg_to_locations(d)
            case _:
                return set()

    # live_after 是 basic blocks CFG 中当前结点后继结点的 live_before_block 的 join
    def transfer(self, ss: List[instr], label: str, live_after_block: Set[location]) -> Set[location]:
        if len(ss) == 0:
            return set()

        # live_before_instr: set[location] = set()
        match ss[-1]:
            case Jump(_):
                self.live_after[ss[-1]] = live_after_block
                self.live_before[ss[-1]] = live_after_block
            case JumpIf(_, _):
                self.live_after[ss[-1]] = live_after_block
                tmp = (self.live_after[ss[-1]] - self.write_vars(ss[-1])) | self.read_vars(ss[-1])
                self.live_before[ss[-1]] = tmp.union(live_after_block)
            case _:
                # self.live_after[ss[-1]] = live_after_block
                self.live_after[ss[-1]] = set()
                self.live_before[ss[-1]] = (self.live_after[ss[-1]] - self.write_vars(ss[-1])) | self.read_vars(ss[-1])
        
        for i, inst in list(reversed(list(enumerate(ss))))[1:]:
            self.live_after[inst] = self.live_before[ss[i+1]]
            match inst:
                case Jump(_):
                    self.live_before[inst] = live_after_block
                case JumpIf(_, _):
                    tmp = (self.live_after[inst] - self.write_vars(inst)) | self.read_vars(inst)
                    self.live_before[inst] = tmp.union(live_after_block)
                case _:
                    self.live_before[inst] = (self.live_after[inst] - self.write_vars(inst)) | self.read_vars(inst)

        return self.live_before[ss[0]]

    def uncover_live(self, p: X86Program) -> Dict[instr, Set[location]]:
        assert isinstance(p.body, dict)
        basic_blocks = p.body
        cfg = build_cfg(p.body)

        self.live_before: dict[instr, set[location]] = {}
        self.live_after: dict[instr, set[location]] = {}

        analyze_dataflow(
            transpose(cfg), # 得是转置的
            lambda label, live_after: self.transfer(basic_blocks[label], label, live_after),
            set(),
            lambda x, y: x.union(y),
        )

        return self.live_after

    ############################################################################
    # Build Interference
    ############################################################################

    def build_interference(
        self, p: X86Program, live_after: Dict[instr, Set[location]]
    ) -> UndirectedAdjList:
        assert isinstance(p.body, dict)

        g = UndirectedAdjList()

        for _, locs in live_after.items():
            for loc in locs:
                g.add_vertex(loc)

        for _, instrs in p.body.items():
            for inst in instrs:
                match inst:
                    case Instr("movq", [s, d]) | Instr("movzbq", [s, d]) if islocation(
                        d
                    ):
                        for v in live_after[inst]:
                            if v != d and v != s:
                                g.add_edge(d, v)
                    case _:
                        for d in self.write_vars(inst):  # type: ignore
                            for v in live_after[inst]:  # type: ignore
                                if v != d:
                                    g.add_edge(d, v)

                        match inst:
                            case Callq(func, _):
                                for v in live_after[inst]:
                                    for r in caller_saved_regs:
                                        g.add_edge(v, Reg(r))
                                if func == "collect":
                                    for v in live_after[inst]:
                                        match v:
                                            case Variable(var):
                                                match self.var_types[var]:
                                                    case TupleType(_):
                                                        for r in callee_saved_regs:
                                                            g.add_edge(v, Reg(r))

        return g

    ############################################################################
    # Allocate Registers
    ############################################################################

    def color_graph(
        self, graph: UndirectedAdjList, variables: Set[location]
    ) -> Tuple[Dict[location, int], Set[location]]:
        def mex(s: Set[int]) -> int:  # type: ignore
            i = 0
            while i in s:  # type: ignore
                i += 1
            return i

        vertices: list[location] = list(graph.vertices())
        saturation: dict[location, set[int]] = {v: set() for v in vertices}
        color: dict[location, int] = {}
        for v in vertices:
            match v:
                case ByteReg(r):
                    def byte_reg_to_reg(r: str) -> str:
                        if r.startswith("a"):
                            return "rax"
                        if r.startswith("b"):
                            return "rbx"
                        if r.startswith("c"):
                            return "rcx"
                        if r.startswith("d"):
                            return "rdx"
                        raise Exception(f"byte_reg_to_reg: invalid byte reg: {r}")
                    color[v] = reg_to_id(byte_reg_to_reg(r))
                    for vv in set(graph.adjacent(v)):
                        saturation[vv] = {color[v]}
                case Reg(r):
                    color[v] = reg_to_id(r)
                    for vv in set(graph.adjacent(v)):
                        saturation[vv] = {color[v]}
                case _:
                    continue

        # isinstance(x, KeyWithPosition)
        worklist = PriorityQueue(
            lambda x, y: len(saturation[x.key]) < len(saturation[y.key])
        )
        for v in vertices:
            # 这里不要过滤 Reg/ByteReg, 否则 worklist.increase_key(v) 会 KeyError
            # 或者下面的 propagate 也要过滤 Reg/ByteReg
            if not isinstance(v, Variable):
                continue
            worklist.push(v)

        while not worklist.empty():
            u = worklist.pop()
            # 过滤掉寄存器, 不参与分配到普通栈的 Variable
            # if u in color or (isinstance(u, Variable) and u not in variables):
            #     continue
            if u not in variables:
                continue
            c = mex(saturation[u])
            color[u] = c
            # propagate
            for v in graph.adjacent(u):
                if not isinstance(v, Variable):
                    continue
                saturation[v].add(c)
                worklist.increase_key(v)

        # coloring 只包含 Variable
        coloring = {}
        spilled = set()
        for loc, col in color.items():
            match loc:
                case Variable(v):
                    assert loc in variables
                    assert col >= 0
                    if col > max(id_to_regs.keys()):
                        spilled.add(loc)
                    else:
                        coloring[loc] = col
                case _:
                    continue
                    # coloring[loc] = col

        return coloring, spilled

    def filter_alloc_var(self, v: Variable) -> bool:
        match v:
            case Variable(var):
                match self.var_types[var]:  # type: ignore
                    case TupleType(_):
                        # if var.startswith("alloc"):
                        return False
                    case _:
                        return True
            case _:
                raise Exception("filter_var unexpected: " + repr(v))

    def allocate_registers(
        self, p: X86Program, g: UndirectedAdjList
    ) -> Tuple[Dict[location, arg], Dict[Variable, arg]]:
        coloring_reg, spilled = self.color_graph(
            g,
            set(
                filter(
                    lambda x: isinstance(x, Variable) and self.filter_alloc_var(x),
                    g.vertices(),
                )
            ),
        )

        # callee
        self.used_callee: list[str] = []
        coloring_reg_home: dict[location, arg] = {
            var: Reg(id_to_reg(reg_id)) for var, reg_id in coloring_reg.items()
        }
        for _, a in coloring_reg_home.items():
            match a:
                case Reg(r):
                    if r in callee_saved_regs:
                        self.used_callee.append(r)
        self.used_callee = list(set(self.used_callee))  # 去重

        # variable -> stack
        spilled_home: dict[Variable, arg] = {}
        for i, var in enumerate(spilled):
            match var:
                case Variable(v):
                    # 减去 used_callee size
                    spilled_home[var] = Deref("rbp", -8 * (i + 1) - 8 * len(self.used_callee))
                case _:
                    raise Exception("allocate_registers: spilled should be Variable")
        self.spilled_size = len(spilled_home) * 8

        # variable -> rootstack
        rootstack_home: dict[Variable, arg] = {}
        rootstack_count = 0
        for v in list(g.vertices()):
            match v:
                case Variable(var):
                    # 可能生成临时变量指向 tuple, 但实际上应该只在 rootstack 上分配一次
                    # 比较 tricky 的做法
                    # TupleType 的都放到 rootstack 上
                    # if not var.startswith("alloc"):
                    #     continue
                    match self.var_types[var]:
                        case TupleType(_):
                            assert (v not in spilled_home)
                            if v in rootstack_home:
                                continue
                            rootstack_home[v] = Deref("r15", 8 * rootstack_count)
                            rootstack_count += 1
                            # print(f"{v} => {rootstack_home[v]}")
        self.roostack_size = rootstack_count * 8

        return coloring_reg_home, {**spilled_home, **rootstack_home}  # type: ignore

    ############################################################################
    # Assign Homes
    ############################################################################

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
                    # raise Exception("unreachable")
                    self.spilled_size += 8
                    home[var] = Deref("rbp", -self.spilled_size)
                    print(f"assign_homes_arg: ({var}) => ({home[var]})")
                    return home[var]
            case _:
                return a
                # raise Exception("assign_homes_arg unexpected " + repr(a))

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
    
    def assign_homes(self, pseudo_x86: X86Program) -> X86Program:
        assert isinstance(pseudo_x86.body, dict)

        # 手动补充一个 conclusion block
        pseudo_x86.body[label_name("conclusion")] = []

        live_after = self.uncover_live(pseudo_x86)
        g = self.build_interference(pseudo_x86, live_after)
        coloring_home, spilled_home = self.allocate_registers(pseudo_x86, g)
        home = {**coloring_home, **spilled_home}

        new_body: dict[str, list[instr]] = {}
        for bb, instrs in pseudo_x86.body.items():
            new_body[bb] = [self.assign_homes_instr(i, home) for i in instrs]  # type: ignore

        return X86Program(new_body)

    
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
                    # TODO: Global 类似于 Deref?
                    case (Deref(_, _) | x86_ast.Global(_), Deref(_, _) | x86_ast.Global(_)):
                        return [
                            Instr("movq", [arg1, Reg("rax")]),
                            Instr(op, [Reg("rax"), arg2])
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

            case Instr("cmpq", [arg, Immediate(n)]):
                return [
                    Instr("movq", [Immediate(n), Reg("rax")]),
                    Instr("cmpq", [arg, Reg("rax")])
                ]
            case Instr("movbzq", [arg, Deref(reg, off)]):
                return [
                    Instr("movbzq", [arg, Reg("rax")]),
                    Instr("movq", [Reg("rax"), Deref(reg, off)])
                ]
            case Jump(_):
                return [i]
            case _:
                return [i]
                # raise Exception("patch_instr unexpected: " + repr(i))

    def patch_instructions(self, p: X86Program) -> X86Program:
        assert isinstance(p.body, dict)

        new_body = {}
        for bb, instrs in p.body.items():
            filtered_instrs = []
            for inst in instrs:
                match inst:
                    case Instr("movq", [Deref(_, _) as d1, Deref(_, _) as d2]) if d1 == d2:
                        continue
                    case Instr("movq", [Reg(r1), Reg(r2)]) if r1 == r2:
                        continue
                    case _:
                        filtered_instrs.append(inst)
            new_body[bb] = [ii for i in filtered_instrs for ii in self.patch_instr(i)]

        return X86Program(new_body)

    ###########################################################################
    # Prelude & Conclusion
    ###########################################################################

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        # print(f"===used_callee: {self.used_callee}")
        # print(f"===spilled_size: {self.spilled_size}")
        # print(f"===rootstack_size: {self.roostack_size}")
        assert isinstance(p.body, dict)
        used_callee_size = len(self.used_callee) * 8
        rsp_aligned = align(self.spilled_size + used_callee_size, 16) - used_callee_size
        # print(f"===rsp_aligned: {rsp_aligned}")

        prelude = [
            Instr("pushq", [Reg("rbp")]),
            Instr("movq", [Reg("rsp"), Reg("rbp")]),
        ]
        prelude += [Instr("pushq", [Reg(r)]) for r in self.used_callee]
        if rsp_aligned > 0:
            prelude += [Instr("subq", [Immediate(rsp_aligned), Reg("rsp")])]

        call_init = [
            Instr("movq", [Immediate(65536), Reg("rdi")]),
            Instr("movq", [Immediate(65536), Reg("rsi")]),
            Callq(label_name("initialize"), 2),
        ]
        init_rootstack = [
            Instr("movq", [Global(label_name("rootstack_begin")), Reg("r15")]),
        ]
        if self.roostack_size > 0:
            for i in range(self.roostack_size // 8):
                init_rootstack.append(
                    # Instr("movq", [Immediate(0), Deref("r15", 8 * i)])
                    Instr("movq", [Immediate(0), Deref("r15", 0)])
                )
                init_rootstack.append(
                    Instr("addq", [Immediate(8), Reg("r15")])
                )
            # init_rootstack.append(
            #     Instr("addq", [Immediate(self.roostack_size), Reg("r15")])
            # )
        p.body[label_name("main")] = (
            prelude + call_init + init_rootstack + [Jump(label_name("start"))]
        )

        conclusion = [Instr("popq", [Reg(r)]) for r in reversed(self.used_callee)] + [
            Instr("popq", [Reg("rbp")]),
            Instr("retq", []),
        ]
        if rsp_aligned > 0:
            conclusion.insert(0, Instr("addq", [Immediate(rsp_aligned), Reg("rsp")]))
        if self.roostack_size > 0:
            conclusion.insert(
                0, Instr("subq", [Immediate(self.roostack_size), Reg("r15")])
            )
        p.body[label_name("conclusion")] = conclusion

        return X86Program(p.body)  # type: ignore
