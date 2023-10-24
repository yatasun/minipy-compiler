import compiler
from graph import UEdge, UndirectedAdjList
from typing import List, Tuple, Set, Dict
from ast import *
from x86_ast import *
from typing import Set, Dict, Tuple
from priority_queue import PriorityQueue
from utils import align

# Skeleton code for the chapter on Register Allocation

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


class Compiler(compiler.Compiler):
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
            case _:
                return set()

    """
    live_after(n) = {}                              # 初始化
    live_before(k) = (live_after(k) - W(k)) ∪ R(k)  # 迭代
    live_after(k-1) = live_before(k)                # 传播
    """

    def uncover_live(self, p: X86Program) -> Dict[instr, Set[location]]:
        live_before = {i: set() for i in range(len(p.body))}
        live_after = {i: set() for i in range(len(p.body))}

        for i, inst in reversed(list(enumerate(p.body))):
            live_before[i] = (live_after[i] - self.write_vars(inst)) | self.read_vars(inst)  # type: ignore
            if i - 1 >= 0:
                live_after[i - 1] = live_before[i]

        return {inst: live_after[i] for i, inst in enumerate(p.body)}  # type: ignore

    ############################################################################
    # Build Interference
    ############################################################################

    """
    ## rule for every instruction
    If I_k == Instr('movq', [s, d]): 
      for v in live_after(k):
        if v != d && v != s:
          add_edge(d, v)
    else:
      for d in W(k):
        for v in live_after(k):
          if v!=d:
            add_edge(d, v)

    ## rule for callq I_k
    for v in live_after(k):
      for r in caller_saved_regs:
        add_edge(v, r)
    """

    def build_interference(
        self, p: X86Program, live_after: Dict[instr, Set[location]]
    ) -> UndirectedAdjList:
        g = UndirectedAdjList()
        for inst in p.body:
            match inst:
                case Instr("movq", [s, d]) if islocation(d):
                    for v in live_after[inst]:
                        if v != d and v != s:
                            g.add_edge(d, v)
                # For the callq instruction, we consider all the caller-saved registers to have been written to
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
    # Allocate Registers
    ############################################################################

    # Returns the coloring and the set of spilled variables.
    def color_graph(
        self, graph: UndirectedAdjList, variables: Set[location]
    ) -> Tuple[Dict[location, int], Set[location]]:
        def mex(s: Set[int]) -> int:  # type: ignore
            i = 0
            while i in s:  # type: ignore
                i += 1
            return i

        vertices: list[location] = list(graph.vertices())
        # print(vertices)
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
            # 这里不要过滤 Reg, 否则 worklist.increase_key(v) 会 KeyError
            worklist.push(v)

        while not worklist.empty():
            u = worklist.pop()
            if u in color:
                continue
            c = mex(saturation[u])
            color[u] = c
            # propagate
            for v in graph.adjacent(u):
                saturation[v].add(c)
                worklist.increase_key(v)

        # coloring 只包含 Variable
        coloring = {}
        spilled = set()
        for loc, col in color.items():
            match loc:
                case Variable(v):
                    assert col >= 0
                    if col > max(id_to_regs.keys()):
                        spilled.add(loc)
                    else:
                        coloring[loc] = col
                case Reg(_):
                    continue
                    # coloring[loc] = col

        return coloring, spilled

    # NOTE: 修改了返回值类型, X86Program 改为 Tuple[Dict[location, arg], Dict[Variable, arg]]
    def allocate_registers(
        self, p: X86Program, g: UndirectedAdjList
    ) -> Tuple[Dict[location, arg], Dict[Variable, arg]]:
        coloring, spilled = self.color_graph(
            g, set(filter(lambda x: isinstance(x, Variable), g.vertices()))
        )

        # pushq rbp: rbp 不参与寄存器分配
        self.used_callee: list[str] = []

        coloring_home: dict[location, arg] = {
            var: Reg(id_to_reg(reg_id)) for var, reg_id in coloring.items()
        }

        for _, a in coloring_home.items():
            match a:
                case Reg(r):
                    if r in callee_saved_regs:
                        self.used_callee.append(r)
        self.used_callee = list(set(self.used_callee))  # 去重
        assert "rbp" not in self.used_callee

        spilled_home: dict[Variable, arg] = {}
        for i, var in enumerate(spilled):
            match var:
                case Variable(v):
                    spilled_home[var] = Deref("rbp", -8 * (i + 1))
                case _:
                    raise Exception("allocate_registers: spilled should be Variable")
        self.spilled_size = len(spilled_home) * 8

        return coloring_home, spilled_home  # type: ignore

    ############################################################################
    # Assign Homes
    ############################################################################

    def assign_homes(self, pseudo_x86: X86Program) -> X86Program:
        live_after = self.uncover_live(pseudo_x86)
        g = self.build_interference(pseudo_x86, live_after)
        coloring_home, spilled_home = self.allocate_registers(pseudo_x86, g)
        home = {**coloring_home, **spilled_home}
        return X86Program([self.assign_homes_instr(i, home) for i in pseudo_x86.body])  # type: ignore

    ###########################################################################
    # Patch Instructions
    ###########################################################################

    def patch_instructions(self, p: X86Program) -> X86Program:
        filtered_instrs = []
        for inst in p.body:
            match inst:
                case Instr("movq", [Deref(_, _) as d1, Deref(_, _) as d2]) if d1 == d2:
                    continue
                case Instr("movq", [Reg(r1), Reg(r2)]) if r1 == r2:
                    # print(f"---->>>>>filter: {inst}")
                    continue
                case _:
                    filtered_instrs.append(inst)
        return super().patch_instructions(X86Program(filtered_instrs))

    ###########################################################################
    # Prelude & Conclusion
    ###########################################################################

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        # print(self.spilled_size)
        # print(self.used_callee)
        used_callee_size = len(self.used_callee) * 8
        rsp_aligned = align(self.spilled_size + used_callee_size, 16) - used_callee_size

        new_body = [
            Instr("pushq", [Reg("rbp")]),
            Instr("movq", [Reg("rsp"), Reg("rbp")]),
        ] + [Instr("pushq", [Reg(r)]) for r in self.used_callee]
        if rsp_aligned > 0:
            new_body += [Instr("subq", [Immediate(rsp_aligned), Reg("rsp")])]
        new_body += p.body  # type: ignore
        if rsp_aligned > 0:
            new_body += [Instr("addq", [Immediate(rsp_aligned), Reg("rsp")])]
        new_body += [Instr("popq", [Reg(r)]) for r in reversed(self.used_callee)] + [
            Instr("popq", [Reg("rbp")]),
            Instr("retq", []),
        ]
        return X86Program(new_body)  # type: ignore


def test_uncover_live_1():
    live_after = {}
    i1 = Instr("movq", [Immediate(1), Variable("v")])
    live_after[i1] = {Variable("v")}

    i2 = Instr("movq", [Immediate(42), Variable("w")])
    live_after[i2] = {Variable("w"), Variable("v")}

    i3 = Instr("movq", [Variable("v"), Variable("x")])
    live_after[i3] = {Variable("w"), Variable("x")}

    i4 = Instr("addq", [Immediate(7), Variable("x")])
    live_after[i4] = {Variable("w"), Variable("x")}

    i5 = Instr("movq", [Variable("x"), Variable("y")])
    live_after[i5] = {Variable("w"), Variable("x"), Variable("y")}

    i6 = Instr("movq", [Variable("x"), Variable("z")])
    live_after[i6] = {Variable("w"), Variable("y"), Variable("z")}

    i7 = Instr("addq", [Variable("w"), Variable("z")])
    live_after[i7] = {Variable("y"), Variable("z")}

    i8 = Instr("movq", [Variable("y"), Variable("tmp_0")])
    live_after[i8] = {Variable("tmp_0"), Variable("z")}

    i9 = Instr("negq", [Variable("tmp_0")])
    live_after[i9] = {Variable("tmp_0"), Variable("z")}

    i10 = Instr("movq", [Variable("z"), Variable("tmp_1")])
    live_after[i10] = {Variable("tmp_0"), Variable("tmp_1")}

    i11 = Instr("addq", [Variable("tmp_0"), Variable("tmp_1")])
    live_after[i11] = {Variable("tmp_1")}

    i12 = Instr("movq", [Variable("tmp_1"), Reg("rdi")])
    live_after[i12] = {Reg("rdi")}

    i13 = Callq("print_int", 1)
    live_after[i13] = set()

    p = X86Program([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13])
    c = Compiler()
    assert live_after == c.uncover_live(p)


# sub-int, select_instructions
def test_uncover_live_2():
    live_after = {}
    i1 = Callq("read_int", 0)
    i2 = Instr("movq", [Reg("rax"), Variable("tmp.0")])
    i3 = Callq("read_int", 0)
    i4 = Instr("movq", [Reg("rax"), Variable("tmp.1")])
    i5 = Instr("movq", [Variable("tmp.0"), Variable("tmp.2")])
    i6 = Instr("subq", [Variable("tmp.1"), Variable("tmp.2")])
    live_after[i6] = {Variable("tmp.1"), Variable("tmp.2")}
    i7 = Instr("movq", [Variable("tmp.2"), Reg("rdi")])
    live_after[i7] = {Reg("rdi")}
    i8 = Callq("print_int", 1)
    live_after[i8] = {}

    p = X86Program([i1, i2, i3, i4, i5, i6, i7, i8])
    c = Compiler()
    # print(c.uncover_live(p))


def helper_generate_program():
    i1 = Instr("movq", [Immediate(1), Variable("v")])
    i2 = Instr("movq", [Immediate(42), Variable("w")])
    i3 = Instr("movq", [Variable("v"), Variable("x")])
    i4 = Instr("addq", [Immediate(7), Variable("x")])
    i5 = Instr("movq", [Variable("x"), Variable("y")])
    i6 = Instr("movq", [Variable("x"), Variable("z")])
    i7 = Instr("addq", [Variable("w"), Variable("z")])
    i8 = Instr("movq", [Variable("y"), Variable("tmp_0")])
    i9 = Instr("negq", [Variable("tmp_0")])
    i10 = Instr("movq", [Variable("z"), Variable("tmp_1")])
    i11 = Instr("addq", [Variable("tmp_0"), Variable("tmp_1")])
    i12 = Instr("movq", [Variable("tmp_1"), Reg("rdi")])
    i13 = Callq("print_int", 1)
    return X86Program([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13])


def test_build_interference_1():
    i1 = Instr("movq", [Immediate(1), Variable("v")])
    i2 = Instr("movq", [Immediate(42), Variable("w")])
    i3 = Instr("movq", [Variable("v"), Variable("x")])
    i4 = Instr("addq", [Immediate(7), Variable("x")])
    i5 = Instr("movq", [Variable("x"), Variable("y")])
    i6 = Instr("movq", [Variable("x"), Variable("z")])
    i7 = Instr("addq", [Variable("w"), Variable("z")])
    i8 = Instr("movq", [Variable("y"), Variable("tmp_0")])
    i9 = Instr("negq", [Variable("tmp_0")])
    i10 = Instr("movq", [Variable("z"), Variable("tmp_1")])
    i11 = Instr("addq", [Variable("tmp_0"), Variable("tmp_1")])
    i12 = Instr("movq", [Variable("tmp_1"), Reg("rdi")])
    i13 = Callq("print_int", 1)

    p = X86Program([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13])
    c = Compiler()
    live_after = c.uncover_live(p)

    assert c.build_interference(X86Program([i1]), live_after).edges() == set()
    assert c.build_interference(X86Program([i2]), live_after).edges() == {
        UEdge(Variable("w"), Variable("v"))
    }
    assert c.build_interference(X86Program([i3]), live_after).edges() == {
        UEdge(Variable("x"), Variable("w"))
    }
    assert c.build_interference(X86Program([i4]), live_after).edges() == {
        UEdge(Variable("x"), Variable("w"))
    }
    assert c.build_interference(X86Program([i5]), live_after).edges() == {
        UEdge(Variable("y"), Variable("w"))
    }
    assert c.build_interference(X86Program([i6]), live_after).edges() == {
        UEdge(Variable("z"), Variable("w")),
        UEdge(Variable("z"), Variable("y")),
    }
    assert c.build_interference(X86Program([i7]), live_after).edges() == {
        UEdge(Variable("z"), Variable("y"))
    }
    assert c.build_interference(X86Program([i8]), live_after).edges() == {
        UEdge(Variable("tmp_0"), Variable("z"))
    }
    assert c.build_interference(X86Program([i9]), live_after).edges() == {
        UEdge(Variable("tmp_0"), Variable("z"))
    }
    assert c.build_interference(X86Program([i10]), live_after).edges() == {
        UEdge(Variable("tmp_0"), Variable("tmp_1"))
    }
    assert c.build_interference(X86Program([i11]), live_after).edges() == set()
    assert c.build_interference(X86Program([i12]), live_after).edges() == set()
    assert c.build_interference(X86Program([i13]), live_after).edges() == set()


def test_assign_homes_1():
    pseudo_x86 = helper_generate_program()
    c = Compiler()
    p = c.assign_homes(pseudo_x86)
    # print(p)


def test_preclude_and_conclusion_1():
    pseudo_x86 = helper_generate_program()
    c = Compiler()
    p = c.assign_homes(pseudo_x86)
    p = c.prelude_and_conclusion(p)
    # print(p)


if __name__ == "__main__":
    test_uncover_live_1()
    test_uncover_live_2()
    test_build_interference_1()
    test_assign_homes_1()
    test_preclude_and_conclusion_1()
