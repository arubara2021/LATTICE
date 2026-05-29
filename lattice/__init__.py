import numpy as np
from .lexer import Lexer
from .parser import Parser
from .optimizer import Optimizer
from .graph_builder import GraphBuilder
from .exporter import Exporter
from .executor import Executor
from .utils import pretty_print


def from_formula(formula, shapes=None, dynamic=False):
    tokens = Lexer(formula).tokenize()
    ast = Parser(tokens).parse()
    opt = Optimizer()
    ast = opt.optimize(ast)
    builder = GraphBuilder(shapes=shapes, dynamic=dynamic)
    model = builder.build(ast)
    exporter = Exporter()
    exporter.validate(model)
    return model


def save(formula, filepath="formula.onnx", shapes=None, dynamic=False):
    model = from_formula(formula, shapes, dynamic)
    exporter = Exporter()
    exporter.export_to_file(model, filepath)
    return filepath


def run(formula, inputs, model_path="formula.onnx", shapes=None, dynamic=True):
    if shapes is None and not dynamic:
        shapes = {}
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                shapes[name] = list(value.shape)
            elif isinstance(value, (list, tuple)):
                shapes[name] = [len(value)]

    save(formula, model_path, shapes, dynamic)
    executor = Executor()
    executor.create_session(model_path)
    return executor.run(inputs)


def benchmark(formula, inputs, runs=1000, model_path="formula.onnx",
              shapes=None, dynamic=True):
    if shapes is None and not dynamic:
        shapes = {}
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                shapes[name] = list(value.shape)
            elif isinstance(value, (list, tuple)):
                shapes[name] = [len(value)]

    save(formula, model_path, shapes, dynamic)
    executor = Executor()
    executor.create_session(model_path)
    return executor.benchmark(inputs, runs)


def describe(formula, shapes=None, dynamic=False):
    model = from_formula(formula, shapes, dynamic)
    pretty_print(model)
    return model
