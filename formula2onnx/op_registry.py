BINARY_OPS = {
    '+': 'Add',
    '-': 'Sub',
    '*': 'Mul',
    '/': 'Div',
    '^': 'Pow',
    '%': 'Mod',
}

UNARY_OPS = {
    '-': 'Neg',
    '+': None,
}

COMPARISON_OPS = {
    '<': 'Less',
    '>': 'Greater',
    '<=': 'LessOrEqual',
    '>=': 'GreaterOrEqual',
    '==': 'Equal',
    '!=': 'NotEqual',
}

FUNCTIONS = {
    'sin': 'Sin',
    'cos': 'Cos',
    'tan': 'Tan',
    'asin': 'Asin',
    'acos': 'Acos',
    'atan': 'Atan',
    'sinh': 'Sinh',
    'cosh': 'Cosh',
    'tanh': 'Tanh',
    'asinh': 'Asinh',
    'acosh': 'Acosh',
    'atanh': 'Atanh',
    'exp': 'Exp',
    'log': 'Log',
    'ln': 'Log',
    'sqrt': 'Sqrt',
    'abs': 'Abs',
    'floor': 'Floor',
    'ceil': 'Ceil',
    'round': 'Round',
    'sign': 'Sign',
    'pow': 'Pow',
    'max': 'Max',
    'min': 'Min',
    'relu': 'Relu',
    'sigmoid': 'Sigmoid',
    'erf': 'Erf',
    'clip': 'Clip',
    'neg': 'Neg',
    'reciprocal': 'Reciprocal',
}

SPECIAL_FUNCTIONS = {
    'log2': 'log_base',
    'log10': 'log_base',
    'cbrt': 'cbrt',
    'factorial': 'factorial',
    'deg': 'deg',
    'rad': 'rad',
}

CONSTANTS = {
    'pi': 3.141592653589793,
    'π': 3.141592653589793,
    'euler': 2.718281828459045,
    'phi': 1.618033988749895,
    'φ': 1.618033988749895,
    'speed_of_light': 299792458.0,
    'planck': 6.62607015e-34,
    'planck_reduced': 1.054571817e-34,
    'gravitational': 6.67430e-11,
    'boltzmann': 1.380649e-23,
    'avogadro': 6.02214076e+23,
    'electron_charge': 1.602176634e-19,
    'vacuum_permittivity': 8.8541878128e-12,
    'vacuum_permeability': 1.25663706212e-6,
    'inf': float('inf'),
    'infinity': float('inf'),
    '∞': float('inf'),
    'nan': float('nan'),
}


def get_opset_version():
    return 18


def get_binary_op(op):
    if op in BINARY_OPS:
        return BINARY_OPS[op]
    raise Exception(f"Unknown binary operator: {op}")


def get_unary_op(op):
    if op in UNARY_OPS:
        return UNARY_OPS[op]
    raise Exception(f"Unknown unary operator: {op}")


def get_comparison_op(op):
    if op in COMPARISON_OPS:
        return COMPARISON_OPS[op]
    raise Exception(f"Unknown comparison operator: {op}")


def get_function(name):
    if name in FUNCTIONS:
        return FUNCTIONS[name]
    if name in SPECIAL_FUNCTIONS:
        return SPECIAL_FUNCTIONS[name]
    raise Exception(f"Unknown function: {name}")


def get_constant(name):
    if name in CONSTANTS:
        return CONSTANTS[name]
    return None


def is_special_function(name):
    return name in SPECIAL_FUNCTIONS


def is_constant(name):
    return name in CONSTANTS
