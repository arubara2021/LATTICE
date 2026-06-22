import math
from .ast_nodes import *


class Optimizer:
    def optimize(self, node):
        if isinstance(node, BinaryOp):
            return self.optimize_binary_op(node)
        if isinstance(node, UnaryOp):
            return self.optimize_unary_op(node)
        if isinstance(node, FunctionCall):
            return self.optimize_function_call(node)
        if isinstance(node, Variable):
            return node
        if isinstance(node, Constant):
            return node
        raise Exception(f"Unknown node type: {type(node)}")

    def optimize_binary_op(self, node):
        left = self.optimize(node.left)
        right = self.optimize(node.right)

        if self.is_constant_value(left, 0) and node.op in ('+', '-'):
            if node.op == '+':
                return right
            if node.op == '-':
                return UnaryOp("-", right)

        if self.is_constant_value(right, 0) and node.op in ('+', '-'):
            return left

        if self.is_constant_value(left, 1) and node.op == '*':
            return right

        if self.is_constant_value(right, 1) and node.op in ('*', '/'):
            return left

        if self.is_constant_value(left, 0) and node.op in ('*', '/'):
            if node.op == '*':
                return Constant(0)
            if node.op == '/':
                raise Exception("Division by zero")

        if self.is_constant_value(right, 0) and node.op == '*':
            return Constant(0)

        if self.is_constant_value(right, 0) and node.op == '/':
            raise Exception("Division by zero")

        if self.is_constant_value(right, 0) and node.op == '^':
            return Constant(1)

        if self.is_constant_value(right, 1) and node.op == '^':
            return left

        if self.is_constant_value(left, 0) and node.op == '^':
            return Constant(0)

        if self.is_constant_value(left, 1) and node.op == '^':
            return Constant(1)

        if self.is_constant_value(left, 1) and node.op == '*':
            return right

        if isinstance(left, Constant) and isinstance(right, Constant):
            return self.fold_constants(left, node.op, right)

        return BinaryOp(left, node.op, right)

    def optimize_unary_op(self, node):
        operand = self.optimize(node.operand)

        if node.op == '+':
            return operand

        if isinstance(operand, UnaryOp) and operand.op == '-':
            return operand.operand

        if isinstance(operand, Constant):
            if node.op == '-':
                return Constant(-operand.value)

        return UnaryOp(node.op, operand)

    def optimize_function_call(self, node):
        args = [self.optimize(arg) for arg in node.args]

        if all(isinstance(arg, Constant) for arg in args):
            return self.fold_function(node.name, args)

        return FunctionCall(node.name, args)

    def is_constant_value(self, node, value):
        if isinstance(node, Constant):
            return node.value == value
        return False

    def fold_constants(self, left, op, right):
        a = left.value
        b = right.value

        if op == '+': result = a + b
        elif op == '-': result = a - b
        elif op == '*': result = a * b
        elif op == '/':
            if b == 0:
                raise Exception("Division by zero")
            result = a / b
        elif op == '^': result = a ** b
        elif op == '%':
            if b == 0:
                raise Exception("Modulo by zero")
            result = a % b
        elif op == '<': result = float(a < b)
        elif op == '>': result = float(a > b)
        elif op == '<=': result = float(a <= b)
        elif op == '>=': result = float(a >= b)
        elif op == '==': result = float(a == b)
        elif op == '!=': result = float(a != b)
        else:
            return BinaryOp(left, op, right)

        return Constant(result)

    def fold_function(self, name, args):
        values = [arg.value for arg in args]

        try:
            if name == 'sin': result = math.sin(values[0])
            elif name == 'cos': result = math.cos(values[0])
            elif name == 'tan': result = math.tan(values[0])
            elif name == 'asin': result = math.asin(values[0])
            elif name == 'acos': result = math.acos(values[0])
            elif name == 'atan': result = math.atan(values[0])
            elif name == 'atan2': result = math.atan2(values[0], values[1])
            elif name == 'sinh': result = math.sinh(values[0])
            elif name == 'cosh': result = math.cosh(values[0])
            elif name == 'tanh': result = math.tanh(values[0])
            elif name == 'asinh': result = math.asinh(values[0])
            elif name == 'acosh': result = math.acosh(values[0])
            elif name == 'atanh': result = math.atanh(values[0])
            elif name == 'exp': result = math.exp(values[0])
            elif name == 'log': result = math.log(values[0])
            elif name == 'ln': result = math.log(values[0])
            elif name == 'log2': result = math.log2(values[0])
            elif name == 'log10': result = math.log10(values[0])
            elif name == 'sqrt': result = math.sqrt(values[0])
            elif name == 'cbrt': result = values[0] ** (1.0/3.0)
            elif name == 'abs': result = abs(values[0])
            elif name == 'floor': result = float(math.floor(values[0]))
            elif name == 'ceil': result = float(math.ceil(values[0]))
            elif name == 'round': result = float(round(values[0]))
            elif name == 'sign':
                if values[0] > 0: result = 1.0
                elif values[0] < 0: result = -1.0
                else: result = 0.0
            elif name == 'pow': result = values[0] ** values[1]
            elif name == 'max': result = max(values)
            elif name == 'min': result = min(values)
            elif name == 'relu': result = max(0.0, values[0])
            elif name == 'sigmoid': result = 1.0 / (1.0 + math.exp(-values[0]))
            elif name == 'erf': result = math.erf(values[0])
            elif name == 'reciprocal': result = 1.0 / values[0]
            elif name == 'factorial': result = float(math.factorial(int(values[0])))
            elif name == 'deg': result = math.degrees(values[0])
            elif name == 'rad': result = math.radians(values[0])
            elif name == 'neg': result = -values[0]
            else:
                return FunctionCall(name, args)
            return Constant(result)
        except (ValueError, OverflowError, ZeroDivisionError):
            return FunctionCall(name, args)
