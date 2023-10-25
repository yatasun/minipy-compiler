from collections import deque
from typing import Dict, List
from graph import DirectedAdjList, transpose
from functools import reduce
from utils import trace
from x86_ast import Jump, JumpIf, instr

def analyze_dataflow(G, transfer, bottom, join):
    trans_G = transpose(G)
    # mapping 是 live_before_block
    mapping = {}
    for v in G.vertices():
        mapping[v] = bottom
    worklist = deque()
    for v in G.vertices():
        worklist.append(v)
    while worklist:
        node = worklist.pop()
        input = reduce(join, [mapping[v] for v in trans_G.adjacent(node)], bottom)
        output = transfer(node, input)
        if output != mapping[node]:
            mapping[node] = output
            for v in G.adjacent(node):
                worklist.append(v)


def build_cfg(basic_blocks: Dict[str, List[instr]]) -> DirectedAdjList:
    cfg = DirectedAdjList()

    for bb, stmts in basic_blocks.items():
        for i in stmts:
            match i:
                case Jump(label) | JumpIf(_, label):
                    cfg.add_edge(bb, label)
    return cfg