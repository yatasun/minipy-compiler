from ast import *
from ast import Assign, List, Module, Set, arg, expr, stmt, cmpop
import select
from compiler import Temporaries
import compiler_register_allocator
from compiler_register_allocator import *
from graph import DirectedAdjList, Vertex, topological_sort, transpose
from utils import Assign, List, Module
from x86_ast import *
from utils import *
from typing import List, Tuple, Set, Dict

from x86_ast import X86Program, arg, instr, location


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
        case _:
            raise Exception("cmpop_to_cc unexpected: " + repr(cmp))


class Compiler(compiler_register_allocator.Compiler):
    ############################################################################
    # Shrink
    ############################################################################

    # L_If -> L_If(without `and`, `or`)
    # desugar

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
            case _:
                raise Exception("shrink_stmt unexpected: " + repr(s))

    def shrink(self, p: Module) -> Module:
        return Module([self.shrink_stmt(s) for s in p.body])

    ############################################################################
    # Remove Complex Operands
    ############################################################################

    # L_If(without `and`, `or`) -> L_If^mon

    def rco_exp(self, e: expr, need_atomic: bool) -> Tuple[expr, Temporaries]:
        result_expr = expr()
        result_temps = []
        match e:
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

            case _:
                return super().rco_exp(e, need_atomic)  # type: ignore
        if need_atomic:
            tmp = generate_name("tmp")
            result_expr, result_temps = Name(tmp), result_temps + [
                (Name(tmp), result_expr)
            ]
        return (result_expr, result_temps)

    def rco_stmt(self, s: stmt) -> List[stmt]:
        match s:
            case If(exp, body, orelse):
                new_exp, temps = self.rco_exp(exp, False)
                new_body = [new_ss for ss in body for new_ss in self.rco_stmt(ss)]
                new_orelse = [new_ss for ss in orelse for new_ss in self.rco_stmt(ss)]
                return make_assigns(temps) + [If(new_exp, new_body, new_orelse)]  # type: ignore
            case _:
                return super().rco_stmt(s)

    ############################################################################
    # Explicate Control
    ############################################################################

    # L_If^mon -> C_If

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
        self, cnd, thn, els, basic_blocks: Dict[str, List[stmt]]
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
                begin_block = self.explicate_effect(result, cont, basic_blocks)
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
                # print(f"======Call(func, args): {func}, {args}")
                return [Expr(e)] + cont
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
                # print(f"======body:{body}")
                # print(f"======orelse:{orelse}")
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
            case Assign([lhs], rhs):
                return self.explicate_assign(rhs, lhs, cont, basic_blocks)
            # expression-statement
            case Expr(value):
                # print(f"======Expr(value): {value}")
                return self.explicate_effect(value, cont, basic_blocks)
            case If(test, body, orelse):
                # print(f"====If stmt: {s}")
                goto_cont = self.create_block(cont, basic_blocks)
                body_ss = goto_cont
                for s in reversed(body):
                    body_ss = self.explicate_stmt(s, body_ss, basic_blocks)
                orelse_ss = goto_cont
                for s in reversed(orelse):
                    orelse_ss = self.explicate_stmt(s, orelse_ss, basic_blocks)
                return self.explicate_pred(test, body_ss, orelse_ss, basic_blocks)
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

    # C_If -> x86_If^Var
    def select_arg(self, e: expr) -> arg:
        match e:
            case Constant(True):
                return Immediate(1)
            case Constant(False):
                return Immediate(0)
            case _:
                return super().select_arg(e)

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
            case _:
                return super().select_stmt_assign(s)

    def select_stmt(self, s: stmt) -> List[instr]:
        match s:
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
                return super().select_stmt(s)

    def select_instructions(self, p: Module) -> X86Program:
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
            case _:
                return super().read_vars(i)

    def write_vars(self, i: instr) -> Set[location]:
        match i:
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
        return super().write_vars(i)

    def build_cfg(self, basic_blocks: Dict[str, List[instr]]) -> DirectedAdjList:
        cfg = DirectedAdjList()

        for bb, stmts in basic_blocks.items():
            for i in stmts:
                match i:
                    case Jump(label) | JumpIf(_, label):
                        cfg.add_edge(bb, label)
                    case _:
                        pass
        return cfg

    def uncover_live_inner_bb(
        self, ss: List[instr], live_before_block: Dict[str, Set[location]]
    ) -> Tuple[Dict[instr, Set[location]], Dict[instr, Set[location]]]:
        live_before = {i: set() for i in range(len(ss))}
        live_after = {i: set() for i in range(len(ss))}
        match ss[len(ss) - 1]:
            case Jump(label):
                live_after[len(ss) - 1] = live_before_block[label]
            case JumpIf(cc, label):
                live_after[len(ss) - 1] = live_before_block[label]
            case _:
                live_after[len(ss) - 1] = set()

        for i, inst in reversed(list(enumerate(ss))):
            match inst:
                case Jump(label):
                    live_before[i] = live_before_block[label]
                case JumpIf(_, label):
                    live_before[i] = (
                        live_after[i] - self.write_vars(inst)
                    ) | self.read_vars(inst)
                    live_before[i] = live_before[i].union(live_before_block[label])
                case _:
                    live_before[i] = (
                        live_after[i] - self.write_vars(inst)
                    ) | self.read_vars(inst)
            if i - 1 >= 0:
                live_after[i - 1] = live_before[i]

        result_live_before =  {inst: live_before[i] for i, inst in enumerate(ss)}
        result_live_after =  {inst: live_after[i] for i, inst in enumerate(ss)}
        return (result_live_before, result_live_after)

    # 基于 cfg 来做 liveness 分析
    def uncover_live(self, p: X86Program) -> Dict[instr, Set[location]]:
        assert isinstance(p.body, dict)
        basic_blocks = p.body

        cfg = self.build_cfg(p.body)
        rcfg = transpose(cfg)
        rtopo_order: list[Vertex] = topological_sort(rcfg)  # type: ignore
        # print(f"======p.body: {p.body}")
        # print(f"======rtopo_order: {rtopo_order}")

        live_before_block: dict[str, set[location]] = {}
        live_after = {}
        for label in rtopo_order:
            stmts = basic_blocks[label]
            if label == label_name("conclusion"):
                live_before_block[label] = set([Reg("rax"), Reg("rsp")])
                continue

            live_before_inner_bb, live_after_inner_bb = self.uncover_live_inner_bb(stmts, live_before_block)
            live_before_block[label] = live_before_inner_bb[stmts[0]]
            live_after.update(live_after_inner_bb)

        return live_after

    ############################################################################
    # Build Interference
    ############################################################################

    def build_interference(
        self, p: X86Program, live_after: Dict[instr, Set[location]]
    ) -> UndirectedAdjList:
        assert isinstance(p.body, dict)

        g = UndirectedAdjList()
        for _, instrs in p.body.items():
            for inst in instrs:
                match inst:
                    case Instr("movq", [s, d]) | Instr("movzbq", [s, d]) if islocation(
                        d
                    ):
                        for v in live_after[inst]:
                            if v != d and v != s:
                                g.add_edge(d, v)
                    case Callq(_, _):
                        for v in live_after[inst]:
                            for r in caller_saved_regs:
                                g.add_edge(v, Reg(r))
                    case _:
                        for d in self.write_vars(inst):  # type: ignore
                            for v in live_after[inst]:  # type: ignore
                                if v != d:
                                    g.add_edge(d, v)

        return g

    ############################################################################
    # Assign Homes
    ############################################################################

    # x86_Var -> x86_If

    def assign_homes(self, pseudo_x86: X86Program) -> X86Program:
        assert isinstance(pseudo_x86.body, dict)

        # 手动补充一个 conclusion block
        pseudo_x86.body[label_name("conclusion")] = []

        live_after = self.uncover_live(pseudo_x86)
        g = self.build_interference(pseudo_x86, live_after)
        coloring_home, spilled_home = self.allocate_registers(pseudo_x86, g)
        home = {**coloring_home, **spilled_home}
        self.spilled_size = len(spilled_home) * 8

        new_body: dict[str, list[instr]]= {}
        for bb, instrs in pseudo_x86.body.items():
            new_body[bb] = [self.assign_homes_instr(i, home) for i in instrs] # type: ignore

        return X86Program(new_body)

    ###########################################################################
    # Patch Instructions
    ###########################################################################

    def patch_instr(self, i: instr) -> List[instr]:
        match i:
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
                return super().patch_instr(i)

    def patch_instructions(self, p: X86Program) -> X86Program:
        assert isinstance(p.body, dict)

        new_body = {}
        for bb, instrs in p.body.items():
            new_body[bb] = [ii for i in instrs for ii in self.patch_instr(i)]

        return X86Program(new_body)
    
    ###########################################################################
    # Prelude & Conclusion
    ###########################################################################

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        # print(self.spilled_size)
        # print(self.used_callee)
        assert isinstance(p.body, dict)
        used_callee_size = len(self.used_callee) * 8
        rsp_aligned = align(self.spilled_size + used_callee_size, 16) - used_callee_size

        prelude = [
            Instr("pushq", [Reg("rbp")]),
            Instr("movq", [Reg("rsp"), Reg("rbp")]),
        ]
        prelude += [Instr("pushq", [Reg(r)]) for r in self.used_callee]
        if rsp_aligned > 0:
            prelude += [Instr("subq", [Immediate(rsp_aligned), Reg("rsp")])]
        p.body[label_name('main')] = prelude + [Jump(label_name("start"))]

        conclusion = [
            Instr("popq", [Reg(r)]) for r in reversed(self.used_callee)] + [
            Instr("popq", [Reg("rbp")]),
            Instr("retq", []),
        ]
        if rsp_aligned > 0:
            conclusion.insert(0, Instr("addq", [Immediate(rsp_aligned), Reg("rsp")]))
        p.body[label_name('conclusion')] = conclusion

        return X86Program(p.body)  # type: ignore
