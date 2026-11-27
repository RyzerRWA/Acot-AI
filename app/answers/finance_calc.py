"""Exact arithmetic for the advice answer. The model writes each calculation as an expression; this module
evaluates it, so a payment, loan size or IRR is never the model's mental arithmetic (ACOT QA AI-030 said
AED 1,095,000 for a loan worth AED 909,152; AI-033 said 7.01% for an IRR of 7.12%)."""
from __future__ import annotations

import ast
import json
import math
import operator
import re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def pmt(rate, n, pv):
    """Level payment that repays pv over n periods at rate per period."""
    return pv / n if rate == 0 else pv * rate / (1 - (1 + rate) ** -n)


def pv(rate, n, payment):
    """Loan a level payment over n periods can service."""
    return payment * n if rate == 0 else payment * (1 - (1 + rate) ** -n) / rate


def fv(rate, n, value):
    return value * (1 + rate) ** n


def npv(rate, flows):
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))


def irr(flows):
    """Rate where npv is zero, by bisection between -99% and 1000%; None when there is no sign change."""
    lo, hi = -0.99, 10.0
    if npv(lo, flows) * npv(hi, flows) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo, flows) * npv(mid, flows) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


_FUNCS = {"pmt": pmt, "pv": pv, "fv": fv, "npv": npv, "irr": irr, "min": min, "max": max, "abs": abs, "sum": sum,
          "round": round}


def evaluate(expr: str):
    """Numbers, + - * / ** ( ), lists and the functions above; anything else raises ValueError."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [walk(e) for e in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[walk(a) for a in node.args])
        raise ValueError(f"not allowed: {ast.dump(node)[:60]}")

    if len(expr) > 400:
        raise ValueError("too long")
    return walk(ast.parse(expr.replace(",_", "").replace("^", "**"), mode="eval"))


PLAN_PROMPT = """List the calculations needed to answer the buyer, using ONLY numbers the buyer gave (and earlier
turns below). Return JSON only, no prose: {{"calcs": [{{"name": "monthly payment (AED)", "expr": "pmt(0.045/12, 300, 1200000)"}}]}}
Allowed in expr: numbers (no commas), + - * / ** ( ), lists [..], and pmt(rate, n, pv), pv(rate, n, payment),
fv(rate, n, value), npv(rate, [cf0, cf1, ...]), irr([cf0, cf1, ...]), min, max, sum. Rates are decimals per
period (5% a year paid monthly = 0.05/12). Percentages: write the ratio times 100 and say "%" in the name.
Definitions: NOI = rent collected after vacancy minus operating costs (management, service charges,
maintenance); never subtract financing or acquisition fees from NOI. Cash yield with fees = NOI / (price + fees).
Cash-on-cash = (NOI - annual interest or debt service) / (cash put in). Cash put in = price - loan + fees
(1m price with a 750k loan and 50k fees = 300k), or deposit + closing costs when those are given instead. Break-even
borrowing rate = the interest rate at which the levered cash-on-cash equals the all-cash yield, unless the buyer
defines it differently. IRR cash flows: money paid out is negative, received is positive, one entry per year.
Return {{"calcs": []}} if nothing needs computing.

Earlier turns:
{history}

Buyer: {question}"""


def _format(value) -> str:
    if value is None:
        return "no solution"
    if isinstance(value, list):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    if isinstance(value, float) and not math.isfinite(value):
        return "undefined"
    text = f"{value:,.2f}" if abs(value) >= 100 else f"{value:,.4f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def results_block(plan_text: str | None) -> str | None:
    """Evaluate the model's plan; one 'name = value' line per calculation that worked, or None."""
    match = re.search(r"\{[\s\S]*\}", plan_text or "")
    if not match:
        return None
    try:
        calcs = json.loads(match.group(0)).get("calcs") or []
    except (ValueError, AttributeError):
        return None
    lines = []
    for calc in calcs[:20]:
        if not isinstance(calc, dict):
            continue
        name, expr = str(calc.get("name") or "").strip(), str(calc.get("expr") or "").strip()
        try:
            value = evaluate(expr)
        except Exception:
            continue
        if name.lower().find("irr") >= 0 and isinstance(value, float) and abs(value) < 10:
            value = value * 100  # irr() returns a ratio; the answer quotes a percentage
            name = name if "%" in name else f"{name} (%)"
        lines.append(f"- {name}: {expr} = {_format(value)}")
    return "\n".join(lines) or None
