import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort
from formula2onnx.lexer import Lexer
from formula2onnx.parser import Parser
from formula2onnx.optimizer import Optimizer
from formula2onnx.graph_builder import GraphBuilder
from helpers import PASSED, FAILED, ERRORS


def verify(formula, inputs, expected, name, tolerance=1e-5):
    global PASSED, FAILED, ERRORS
    try:
        tokens = Lexer(formula).tokenize()
        ast = Parser(tokens).parse()
        opt = Optimizer()
        ast = opt.optimize(ast)
        builder = GraphBuilder()
        model = builder.build(ast)

        input_feed = {}
        for k, v in inputs.items():
            input_feed[k] = np.array([v], dtype=np.float32)

        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, input_feed)[0][0]

        if abs(float(result) - float(expected)) < tolerance:
            PASSED += 1
            print(f"  PASS  {name:50s}  got={result}  expected={expected}")
        else:
            FAILED += 1
            err = f"  FAIL  {name:50s}  got={result}  expected={expected}"
            print(err)
            ERRORS.append(err)
    except Exception as e:
        FAILED += 1
        err = f"  ERROR {name:50s}  {e}"
        print(err)
        ERRORS.append(err)


def test_node_generation():
    print("\n--- GRAPH BUILDER: NODE GENERATION ---")

    tokens = Lexer("a + b").tokenize()
    ast = Parser(tokens).parse()
    builder = GraphBuilder()
    model = builder.build(ast)
    nodes = model.graph.node

    global PASSED, FAILED, ERRORS

    if len(nodes) == 1 and nodes[0].op_type == "Add":
        PASSED += 1
        print(f"  PASS  single add node generated")
    else:
        FAILED += 1
        err = f"  FAIL  expected 1 Add node, got {len(nodes)} nodes: {[n.op_type for n in nodes]}"
        print(err)
        ERRORS.append(err)


def test_input_collection():
    print("\n--- GRAPH BUILDER: INPUT COLLECTION ---")

    global PASSED, FAILED, ERRORS

    tokens = Lexer("a + b * c").tokenize()
    ast = Parser(tokens).parse()
    builder = GraphBuilder()
    model = builder.build(ast)
    input_names = [i.name for i in model.graph.input]

    if set(input_names) == {"a", "b", "c"}:
        PASSED += 1
        print(f"  PASS  correct inputs collected: {input_names}")
    else:
        FAILED += 1
        err = f"  FAIL  expected a,b,c got {input_names}"
        print(err)
        ERRORS.append(err)


def test_duplicate_variables():
    print("\n--- GRAPH BUILDER: DUPLICATE VARIABLES ---")

    global PASSED, FAILED, ERRORS

    tokens = Lexer("a + a * a").tokenize()
    ast = Parser(tokens).parse()
    builder = GraphBuilder()
    model = builder.build(ast)
    input_names = [i.name for i in model.graph.input]

    if input_names == ["a"]:
        PASSED += 1
        print(f"  PASS  variable 'a' registered only once")
    else:
        FAILED += 1
        err = f"  FAIL  expected ['a'] got {input_names}"
        print(err)
        ERRORS.append(err)


def test_constant_deduplication():
    print("\n--- GRAPH BUILDER: CONSTANT DEDUPLICATION ---")

    global PASSED, FAILED, ERRORS

    tokens = Lexer("x ^ 2 + y ^ 2 + z ^ 2").tokenize()
    ast = Parser(tokens).parse()
    opt = Optimizer()
    ast = opt.optimize(ast)
    builder = GraphBuilder()
    model = builder.build(ast)
    inits = model.graph.initializer

    if len(inits) == 1 and inits[0].name:
        PASSED += 1
        print(f"  PASS  3 uses of '2' share 1 initializer: {inits[0].name}")
    else:
        FAILED += 1
        names = [i.name for i in inits]
        err = f"  FAIL  expected 1 initializer, got {len(inits)}: {names}"
        print(err)
        ERRORS.append(err)


def test_constant_conversion():
    print("\n--- GRAPH BUILDER: BUILTIN CONSTANTS ---")

    global PASSED, FAILED, ERRORS

    tokens = Lexer("pi").tokenize()
    ast = Parser(tokens).parse()
    builder = GraphBuilder()
    model = builder.build(ast)

    has_initializer = len(model.graph.initializer) > 0
    has_no_inputs = len(model.graph.input) == 0

    if has_initializer and has_no_inputs:
        PASSED += 1
        val = list(model.graph.initializer[0].float_data)
        print(f"  PASS  pi converted to constant initializer: {val}")
    else:
        FAILED += 1
        err = f"  FAIL  pi should be initializer not input"
        print(err)
        ERRORS.append(err)


def test_comparison_output_type():
    print("\n--- GRAPH BUILDER: COMPARISON OUTPUT TYPE ---")

    global PASSED, FAILED, ERRORS

    tokens = Lexer("a < b").tokenize()
    ast = Parser(tokens).parse()
    builder = GraphBuilder()
    model = builder.build(ast)

    out = model.graph.output[0]
    elem_type = out.type.tensor_type.elem_type

    if elem_type == 9:
        PASSED += 1
        print(f"  PASS  comparison output type is BOOL (9)")
    else:
        FAILED += 1
        err = f"  FAIL  expected BOOL(9) got {elem_type}"
        print(err)
        ERRORS.append(err)


def test_numerical_correctness():
    print("\n--- GRAPH BUILDER: NUMERICAL CORRECTNESS ---")

    verify("a + b", {"a": 3, "b": 5}, 8.0, "a + b")
    verify("a * b", {"a": 3, "b": 5}, 15.0, "a * b")
    verify("a - b", {"a": 10, "b": 3}, 7.0, "a - b")
    verify("a / b", {"a": 10, "b": 2}, 5.0, "a / b")
    verify("a ^ b", {"a": 2, "b": 10}, 1024.0, "a ^ b")
    verify("a % b", {"a": 17, "b": 5}, 2.0, "a % b")
    verify("-a", {"a": 5}, -5.0, "unary minus")
    verify("sin(x)", {"x": 0}, 0.0, "sin(0)")
    verify("cos(x)", {"x": 0}, 1.0, "cos(0)")
    verify("sqrt(x)", {"x": 16}, 4.0, "sqrt(16)")
    verify("abs(x)", {"x": -7}, 7.0, "abs(-7)")
    verify("exp(x)", {"x": 0}, 1.0, "exp(0)")
    verify("log(x)", {"x": 1}, 0.0, "log(1)")
    verify("log2(x)", {"x": 64}, 6.0, "log2(64)")
    verify("log10(x)", {"x": 100000}, 5.0, "log10(100000)")
    verify("a < b", {"a": 3, "b": 5}, True, "3 < 5")
    verify("a > b", {"a": 5, "b": 3}, True, "5 > 3")
    verify("a != b", {"a": 3, "b": 5}, True, "3 != 5")
    verify("pi * r ^ 2", {"r": 5}, 78.53981633974483, "pi * r^2")


if __name__ == "__main__":
    test_node_generation()
    test_input_collection()
    test_duplicate_variables()
    test_constant_deduplication()
    test_constant_conversion()
    test_comparison_output_type()
    test_numerical_correctness()

    total = PASSED + FAILED
    print("\n" + "=" * 60)
    print(f"  test_graph_builder.py")
    print(f"  Total: {total}  Passed: {PASSED}  Failed: {FAILED}")
    if FAILED == 0:
        print(f"  STATUS: ALL PASS")
    print("=" * 60)
