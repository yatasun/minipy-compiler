from ast import *
from utils import input_int, add64, sub64, neg64
from itertools import zip_longest
from typing import Union


def interp_exp(e):
  match e:
    case BinOp(left, Add(), right):
      l = interp_exp(left)
      r = interp_exp(right)
      return add64(l, r)
    case BinOp(left, Sub(), right):
      l = interp_exp(left)
      r = interp_exp(right)
      return sub64(l, r)
    case UnaryOp(USub(), v):
      return neg64(interp_exp(v))
    case Constant(value):
      return value
    case Call(Name("input_int"), []):
      return input_int()
    case _:
      raise Exception("error in interp_exp, unexpected " + repr(e))


def interp_stmt(s):
  match s:
    case Expr(Call(Name("print"), [arg])):
      print(interp_exp(arg))
    case Expr(value):
      interp_exp(value)
    case _:
      raise Exception("error in interp_stmt, unexpected " + repr(s))


def interp(p):
  match p:
    case Module(body):
      for s in body:
        interp_stmt(s)
    case _:
      raise Exception("error in interp, unexpected " + repr(p))


# This version is for InterpLvar to inherit from
class InterpLint:
  def interp_exp(self, e, env):
    match e:
      case BinOp(left, Add(), right):
        l = self.interp_exp(left, env)
        r = self.interp_exp(right, env)
        return add64(l, r)
      case BinOp(left, Sub(), right):
        l = self.interp_exp(left, env)
        r = self.interp_exp(right, env)
        return sub64(l, r)
      case UnaryOp(USub(), v):
        return neg64(self.interp_exp(v, env))
      case Constant(value):
        return value
      case Call(Name("input_int"), []):
        return input_int()
      case _:
        raise Exception("error in interp_exp, unexpected " + repr(e))

  # The cont parameter is a list of statements that are the
  # continuaton of the current statement s.
  # We use this continuation-passing approach because
  # it enables the handling of Goto in interp_Cif.py.
  def interp_stmt(self, s, env, cont):
    match s:
      case Expr(Call(Name("print"), [arg])):
        val = self.interp_exp(arg, env)
        print(val, end="")
        return self.interp_stmts(cont, env)
      case Expr(value):
        self.interp_exp(value, env)
        return self.interp_stmts(cont, env)
      case _:
        raise Exception("error in interp_stmt, unexpected " + repr(s))

  def interp_stmts(self, ss, env):
    match ss:
      case []:
        return 0
      case [s, *ss]:
        return self.interp_stmt(s, env, ss)

  def interp(self, p):
    match p:
      case Module(body):
        self.interp_stmts(body, {})
      case _:
        raise Exception("error in interp, unexpected " + repr(p))


# Input: Lint
# Output: Lint
# Purpose: A transformer to perform partial evaluation on Lint.
class PELint:
  @staticmethod
  def pe_neg(r):
    match r:
      case Constant(n):
        return Constant(neg64(n))
      case _:
        return UnaryOp(USub(), r)
  
  @staticmethod
  def pe_add(r1, r2):
    match (r1, r2):
      case (Constant(n1), Constant(n2)):
        return Constant(add64(n1, n2))
      case _:
        return BinOp(r1, Add(), r2)

  @staticmethod
  def pe_sub(r1, r2):
    match (r1, r2):
      case (Constant(n1), Constant(n2)):
        return Constant(sub64(n1, n2))
      case _:
        return BinOp(r1, Sub(), r2)

  @staticmethod
  def pe_exp(e):
    match e:
      case BinOp(left, Add(), right):
        return PELint.pe_add(PELint.pe_exp(left), PELint.pe_exp(right))
      case BinOp(left, Sub(), right):
        return PELint.pe_sub(PELint.pe_exp(left), PELint.pe_exp(right))
      case UnaryOp(USub(), v):
        return PELint.pe_neg(PELint.pe_exp(v))
      case Constant(value):
        return e
      case Call(Name("input_int"), []):
        return e

  @staticmethod
  def pe_stmt(s):
    match s:
      case Expr(Call(Name('print'), [arg])):
        return Expr(Call(Name('print'), [PELint.pe_exp(arg)]))
      case Expr(value):
        return Expr(PELint.pe_exp(value))


  @staticmethod
  def pe_P_int(p):
    match p:
      case Module(body):
        new_body = [PELint.pe_stmt(s) for s in body]
        return Module(new_body)

def compare_ast(node1: Union[expr, list[expr]], node2: Union[expr, list[expr]]) -> bool:
  if type(node1) is not type(node2):
    return False

  if isinstance(node1, AST):
    for k, v in vars(node1).items():
      if k in {"lineno", "end_lineno", "col_offset", "end_col_offset", "ctx"}:
        continue
      if not compare_ast(v, getattr(node2, k)):
        return False
    return True
  elif isinstance(node1, list) and isinstance(node2, list):
    return all(compare_ast(n1, n2) for n1, n2 in zip_longest(node1, node2))
  else:
    return node1 == node2

def test_pe_exp():
  assert compare_ast(PELint.pe_exp(BinOp(Constant(2), Add(), Constant(3))), Constant(5)) # type: ignore
  assert compare_ast(PELint.pe_exp(BinOp(Constant(5), Sub(), Constant(3))), Constant(2)) # type: ignore
  assert compare_ast(PELint.pe_exp(UnaryOp(USub(), Constant(5))), Constant(-5)) # type: ignore
  assert compare_ast(PELint.pe_exp(Call(Name("input_int"), [])), Call(Name("input_int"), [])) # type: ignore

def test_pe_stmt():
  assert compare_ast(PELint.pe_stmt(Expr(BinOp(Constant(2), Add(), Constant(3)))), Expr(Constant(5))) # type: ignore
  assert compare_ast(PELint.pe_stmt(Expr(Call(Name('print'), [Constant(5)]))), Expr(Call(Name('print'), [Constant(5)]))) # type: ignore

if __name__ == "__main__":
  eight = Constant(8)
  neg_eight = UnaryOp(USub(), eight)
  read = Call(Name("input_int"), [])
  ast1_1 = BinOp(read, Add(), neg_eight)
  pr = Expr(Call(Name("print"), [ast1_1]))
  p = Module([pr])
  interp(p)

  # Test partial evaluation
  test_pe_exp()
  test_pe_stmt()
