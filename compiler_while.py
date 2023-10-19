from ast import *
from ast import Assign, Dict, List, Module, Set, arg, expr, stmt, cmpop
import select
from compiler import Temporaries
import compiler_if
from compiler_register_allocator import Dict, List, Set, X86Program, instr, location, stmt
from dataflow_analysis import analyze_dataflow
from graph import DirectedAdjList, Vertex, topological_sort, transpose
from utils import Assign, Dict, List, Module, Set, stmt
from x86_ast import *
from utils import *
from typing import List, Tuple, Set, Dict

from x86_ast import X86Program, arg, instr, location

class Compiler(compiler_if.Compiler):
    ############################################################################
    # Shrink
    ############################################################################

    def shrink_stmt(self, s: stmt) -> stmt:
        match s:
            case While(test, body, []):
                new_body = [self.shrink_stmt(s) for s in body]
                return While(self.shrink_exp(test), new_body, [])
            case _:
                return super().shrink_stmt(s)

    ############################################################################
    # Remove Complex Operands
    ############################################################################

    def rco_stmt(self, s: stmt) -> List[stmt]:
        match s:
            case While(test, body, []):
                new_test, temps = self.rco_exp(test, False)
                new_body = [new_s for s in body for new_s in self.rco_stmt(s)]
                return make_assigns(temps) + [While(new_test, new_body, [])] # type: ignore
            case _:
                return super().rco_stmt(s)

    ############################################################################
    # Explicate Control
    ############################################################################

    def explicate_stmt(
        self, s: stmt, cont: List[stmt], basic_blocks: Dict[str, List[stmt]]
    ) -> List[stmt]:
        match s:
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
                return super().explicate_stmt(s, cont, basic_blocks)
        
    ###########################################################################
    # Uncover Live
    ###########################################################################

    def transfer(self, ss: List[instr], label: str, live_after: Set[location]) -> Set[location]:
        live_before: set[location] = set()
        if label == label_name("conclusion"):
            live_before = set([Reg("rax"), Reg("rsp")])
        else:
            live_before = set()
        
        for s in reversed(ss):
            self.live_after[s] = live_after
            live_before = (live_before - self.write_vars(s)) | self.read_vars(s)
            live_after = live_before
        return live_before

    def uncover_live(self, p: X86Program) -> Dict[instr, Set[location]]:
        assert isinstance(p.body, dict)
        basic_blocks = p.body
        cfg = self.build_cfg(p.body)

        self.live_after: dict[instr, set[location]] = {}

        analyze_dataflow(
            transpose(cfg),
            lambda label, live_after: self.transfer(basic_blocks[label], label, live_after),
            set(),
            lambda x, y: x.union(y),
        )

        return self.live_after
