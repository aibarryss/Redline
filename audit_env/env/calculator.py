"""Safe arithmetic-only Decimal calculator (no eval, no files, no code execution)."""

from __future__ import annotations

import re
from decimal import (
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    Subnormal,
    Underflow,
    localcontext,
)

from audit_env.constants import MAX_CALC_EXPR_LEN

_TOKEN_RE = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([+\-*/()]))")
_MAX_DEPTH = 20
_CALC_PRECISION = 28
_CALC_EMAX = 100
_CALC_EMIN = -100
_MAX_RESULT_LEN = 128


class CalculatorError(ValueError):
    """Raised on invalid or unsafe calculator expressions."""


def _tokenize(expr: str) -> list[tuple[str, str]]:
    if not isinstance(expr, str):
        raise CalculatorError("Expression must be a string.")
    expr = expr.strip()
    if not expr:
        raise CalculatorError("Expression cannot be empty.")
    if len(expr) > MAX_CALC_EXPR_LEN:
        raise CalculatorError(f"Expression exceeds max length ({MAX_CALC_EXPR_LEN}).")

    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        match = _TOKEN_RE.match(expr, pos)
        if not match:
            raise CalculatorError("Expression contains disallowed tokens or characters.")
        num, op = match.groups()
        if num is not None:
            tokens.append(("NUM", num))
        else:
            tokens.append(("OP", op))
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse_expr(self, depth: int = 0) -> Decimal:
        if depth > _MAX_DEPTH:
            raise CalculatorError("Expression nesting depth exceeded.")
        left = self.parse_term(depth)
        while (tok := self.peek()) is not None and tok[0] == "OP" and tok[1] in ("+", "-"):
            self.consume()
            right = self.parse_term(depth)
            left = left + right if tok[1] == "+" else left - right
        return left

    def parse_term(self, depth: int) -> Decimal:
        if depth > _MAX_DEPTH:
            raise CalculatorError("Expression nesting depth exceeded.")
        left = self.parse_factor(depth)
        while (tok := self.peek()) is not None and tok[0] == "OP" and tok[1] in ("*", "/"):
            self.consume()
            right = self.parse_factor(depth)
            if tok[1] == "*":
                left = left * right
            else:
                if right == 0:
                    raise CalculatorError("Division by zero.")
                left = left / right
        return left

    def parse_factor(self, depth: int) -> Decimal:
        if depth > _MAX_DEPTH:
            raise CalculatorError("Expression nesting depth exceeded.")
        tok = self.peek()
        if tok is None:
            raise CalculatorError("Unexpected end of expression.")
        if tok[0] == "OP" and tok[1] in ("+", "-"):
            self.consume()
            val = self.parse_factor(depth + 1)
            return +val if tok[1] == "+" else -val
        if tok[0] == "OP" and tok[1] == "(":
            self.consume()
            val = self.parse_expr(depth + 1)
            closing = self.peek()
            if closing is None or closing != ("OP", ")"):
                raise CalculatorError("Unmatched parenthesis.")
            self.consume()
            return val
        if tok[0] == "NUM":
            self.consume()
            try:
                # Apply unary plus inside active Decimal context so literals exceeding
                # Emax/Emin trigger Overflow/Underflow/Subnormal immediately.
                return +Decimal(tok[1])
            except (InvalidOperation, Overflow, Underflow, Subnormal) as exc:
                raise CalculatorError("Invalid or out-of-range decimal literal.") from exc
        raise CalculatorError("Invalid syntax in expression.")


def evaluate_expression(expression: str) -> str:
    """Evaluate a pure arithmetic expression using bounded Decimal context."""
    tokens = _tokenize(expression)
    parser = _Parser(tokens)
    with localcontext() as ctx:
        ctx.prec = _CALC_PRECISION
        ctx.Emax = _CALC_EMAX
        ctx.Emin = _CALC_EMIN
        ctx.traps[Overflow] = True
        ctx.traps[Underflow] = True
        ctx.traps[Subnormal] = True
        ctx.traps[DivisionByZero] = True
        ctx.traps[InvalidOperation] = True
        try:
            result = parser.parse_expr()
            if parser.peek() is not None:
                raise CalculatorError("Unexpected trailing tokens.")
            if not result.is_finite():
                raise CalculatorError("Non-finite arithmetic result.")
            normalized = result.normalize()
            if normalized == 0:
                return "0"
            formatted = format(normalized, "f")
        except DivisionByZero as exc:
            raise CalculatorError("Division by zero.") from exc
        except (InvalidOperation, Overflow, Underflow, Subnormal) as exc:
            raise CalculatorError("Arithmetic overflow, underflow, or invalid operation.") from exc
        if len(formatted) > _MAX_RESULT_LEN:
            raise CalculatorError("Formatted result exceeds maximum allowed length.")
        return formatted
