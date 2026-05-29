class BinaryOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOp({self.left}, {self.op}, {self.right})"


class UnaryOp:
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp({self.op}, {self.operand})"


class FunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"FunctionCall({self.name}, {self.args})"


class Variable:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Variable({self.name})"


class Constant:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Constant({self.value})"


