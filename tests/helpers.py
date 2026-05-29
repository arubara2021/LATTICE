import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import formula2onnx
from formula2onnx.lexer import Lexer
from formula2onnx.parser import Parser
from formula2onnx.optimizer import Optimizer
from formula2onnx.graph_builder import GraphBuilder
from formula2onnx.exporter import Exporter

PASSED = 0
FAILED = 0
ERRORS = []


def run_test(name, formula, inputs, expected, tolerance=1e-5):
    global PASSED, FAILED, ERRORS
    try:
        result = formula2onnx.run(formula, inputs, model_path="test_temp.onnx")

        if isinstance(expected, bool):
            match = (result == expected)
        elif isinstance(expected, str):
            match = (str(result) == expected)
        else:
            match = abs(float(result) - float(expected)) < tolerance

        if match:
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


def run_lexer_test(name, text, expected_tokens):
    global PASSED, FAILED, ERRORS
    try:
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        token_strs = []
        for t in tokens:
            if t.value is not None:
                token_strs.append(f"{t.type}:{t.value}")
            else:
                token_strs.append(t.type)

        if token_strs == expected_tokens:
            PASSED += 1
            print(f"  PASS  {name}")
        else:
            FAILED += 1
            err = f"  FAIL  {name}\n    got:      {token_strs}\n    expected: {expected_tokens}"
            print(err)
            ERRORS.append(err)
    except Exception as e:
        FAILED += 1
        err = f"  ERROR {name}  {e}"
        print(err)
        ERRORS.append(err)


def run_parser_test(name, text, expected_repr):
    global PASSED, FAILED, ERRORS
    try:
        tokens = Lexer(text).tokenize()
        ast = Parser(tokens).parse()
        got = repr(ast)

        if got == expected_repr:
            PASSED += 1
            print(f"  PASS  {name}")
        else:
            FAILED += 1
            err = f"  FAIL  {name}\n    got:      {got}\n    expected: {expected_repr}"
            print(err)
            ERRORS.append(err)
    except Exception as e:
        FAILED += 1
        err = f"  ERROR {name}  {e}"
        print(err)
        ERRORS.append(err)


def expected_onnx(text, inputs):
    result = 0.0
    if text == "sin(a) + cos(b)":
        result = math.sin(inputs["a"]) + math.cos(inputs["b"])
    elif text == "a + b":
        result = inputs["a"] + inputs["b"]
    elif text == "a * b":
        result = inputs["a"] * inputs["b"]
    elif text == "a - b":
        result = inputs["a"] - inputs["b"]
    elif text == "a / b":
        result = inputs["a"] / inputs["b"]
    return result


def print_summary(test_file_name):
    global PASSED, FAILED, ERRORS
    total = PASSED + FAILED
    print("\n" + "=" * 60)
    print(f"  {test_file_name}")
    print(f"  Total: {total}  Passed: {PASSED}  Failed: {FAILED}")
    if FAILED == 0:
        print(f"  STATUS: ALL PASS")
    else:
        print(f"  STATUS: {FAILED} FAILURES")
        for err in ERRORS:
            print(f"    {err}")
    print("=" * 60)
    PASSED = 0
    FAILED = 0
    ERRORS = []
