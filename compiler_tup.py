from ast import *
from ast import Assign, Dict, List, Module, Set, arg, expr, stmt, cmpop, Tuple
import select
from compiler import Temporaries
import compiler_while
from compiler_register_allocator import (
    Dict,
    List,
    Module,
    Set,
    Variable,
    X86Program,
    arg,
    arg_to_locations,
    expr,
    id_to_reg,
    instr,
    location,
    reg_to_id,
    stmt,
    caller_saved_regs,
    callee_saved_regs,
    islocation,
    id_to_regs,
)
from dataflow_analysis import analyze_dataflow
from graph import (
    DirectedAdjList,
    UndirectedAdjList,
    Vertex,
    topological_sort,
    transpose,
)
from priority_queue import PriorityQueue
from type_check_Ctup import TypeCheckCtup
from type_check_Ltup import TypeCheckLtup
from utils import Assign, Dict, List, Module, Set, arg, expr, stmt
from x86_ast import *
from utils import *
from typing import List, Tuple, Set, Dict

from x86_ast import X86Program, arg, instr, location, Global


class Compiler(compiler_while.Compiler):
    var_types = {}

    ############################################################################
    # Shrink
    ############################################################################

    # L_Tup -> L_Tup

    def shrink_exp(self, e: expr) -> expr:
        match e:
            case ast.Tuple(exps, Load()):
                return ast.Tuple([self.shrink_exp(e) for e in exps], Load())
            case Subscript(exp, Constant(n), Load()):
                return Subscript(self.shrink_exp(exp), Constant(n), Load())
            case Call(Name("len"), [exp]):
                return Call(Name("len"), [self.shrink_exp(exp)])
            case _:
                return super().shrink_exp(e)

    ############################################################################
    # Expose Allocation
    ############################################################################

    # L_Tup -> L_Alloc

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

    # L_Alloc -> L_Alloc^mon

    def rco_exp(self, e: expr, need_atomic: bool) -> Tuple[expr, Temporaries]:
        result_expr = expr()
        result_temps = []
        match e:
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
                return super().rco_exp(e, need_atomic)

        if need_atomic:
            tmp = generate_name("tmp")
            result_expr, result_temps = Name(tmp), result_temps + [
                (Name(tmp), result_expr)
            ]
        return (result_expr, result_temps)

    def rco_stmt(self, s: stmt) -> List[stmt]:
        match s:
            case Assign([Subscript(lhs, index, Store())], rhs):
                lhs_atm, lhs_temps = self.rco_exp(lhs, True)
                index_atm, index_temps = self.rco_exp(index, True)
                rhs_atm, rhs_temps = self.rco_exp(rhs, True)
                return make_assigns(lhs_temps + index_temps + rhs_temps) + [Assign([Subscript(lhs_atm, index_atm, Store())], rhs_atm)]  # type: ignore
            case Collect(_):
                return [s]
            case _:
                return super().rco_stmt(s)

    ############################################################################
    # Explicate Control
    ############################################################################

    # L_Alloc^mon -> C_Tup

    def explicate_effect(
        self, e: expr, cont: List[stmt], basic_blocks: Dict[str, List[stmt]]
    ) -> List[stmt]:
        match e:
            case GlobalValue(_) | Allocate(_, _):
                return cont
            case _:
                return super().explicate_effect(e, cont, basic_blocks)

    # IfExp / If statement
    def explicate_pred(
        self,
        cnd: expr,
        thn: List[stmt],
        els: List[stmt],
        basic_blocks: Dict[str, List[stmt]],
    ) -> List[stmt]:
        match cnd:
            case Subscript(tup, index, Load()):
                tmp = generate_name("tmp")
                return self.explicate_assign(
                    cnd,
                    Name(tmp),
                    self.explicate_pred(Name(tmp), thn, els, basic_blocks),
                    basic_blocks,
                )
            case _:
                return super().explicate_pred(cnd, thn, els, basic_blocks)

    def explicate_stmt(
        self, s: stmt, cont: List[stmt], basic_blocks: Dict[str, List[stmt]]
    ) -> List[stmt]:
        match s:
            case Collect(_) | Assign([Subscript(_, _, Store())], _):
                return [s] + cont
            case _:
                return super().explicate_stmt(s, cont, basic_blocks)

    ############################################################################
    # Select Instructions
    ############################################################################

    # C_Tup -> x86_Global

    def select_arg(self, e: expr) -> arg:
        match e:
            case GlobalValue(gv):
                return Global(gv)
            case _:
                return super().select_arg(e)

    def select_stmt(self, s: stmt) -> List[instr]:
        match s:
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
            case _:
                return super().select_stmt(s)

    def select_instructions(self, p: Module) -> X86Program:
        TypeCheckCtup().type_check(p)
        self.var_types = p.var_types  # type: ignore
        # print(f"===self.var_types: {self.var_types}")

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
            case Instr("andq", [s, d]):
                return arg_to_locations(s) | arg_to_locations(d)
            case Instr("sarq", [s, d]):
                return arg_to_locations(s) | arg_to_locations(d)
            case _:
                return super().read_vars(i)

    def write_vars(self, i: instr) -> Set[location]:
        match i:
            case Instr("andq", [s, d]):
                return arg_to_locations(d)
            case Instr("sarq", [s, d]):
                return arg_to_locations(d)
            case _:
                return super().write_vars(i)

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
        cfg = self.build_cfg(p.body)

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
                # print(f"===inst: {inst}")
                match inst:
                    case Instr("movq", [s, d]) | Instr("movzbq", [s, d]) if islocation(
                        d
                    ):
                        # print(f"===inst: {inst}; live_after: {live_after[inst]}")
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

        # print(f"===g: {g.edges()}")
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
        # print(color)
        # print(saturation)

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
        # print(list(g.vertices()))
        # print(
        #     set(
        #         filter(
        #             lambda x: isinstance(x, Variable) and self.filter_alloc_var(x),
        #             g.vertices(),
        #         )
        #     )
        # )

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
        # print(f"===coloring_reg_home: {coloring_reg_home}")
        # print(f"===used_callee: {self.used_callee}")

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
        # print(f"===spilled_home: {spilled_home}")

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
        # print(f"===rootstack_home: {rootstack_home}")
        self.roostack_size = rootstack_count * 8

        return coloring_reg_home, {**spilled_home, **rootstack_home}  # type: ignore

    ############################################################################
    # Assign Homes
    ############################################################################

    # x86_Global^Var -> x86_Global^Var

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

    ###########################################################################
    # Patch Instructions
    ###########################################################################

    def patch_instr(self, i: instr) -> List[instr]:
        # print(f"^^^^^^^instr: {i}")
        match i:
            case _:
                return super().patch_instr(i)

    ###########################################################################
    # Prelude & Conclusion
    ###########################################################################

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        # print(self.used_callee)
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
