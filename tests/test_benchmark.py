import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import formula2onnx
from formula2onnx.executor import Executor


def benchmark_formula(name, formula, inputs, runs=5000):
    formula2onnx.save(formula, "bench_temp.onnx")
    executor = Executor()
    executor.create_session("bench_temp.onnx")
    stats = executor.benchmark(inputs, runs)
    return stats


def test_scalar_operations():
    print("\n--- BENCHMARK: SCALAR OPERATIONS ---")
    print(f"  {'Formula':50s} {'Runs/sec':>12s} {'Avg(ms)':>12s} {'Total(s)':>12s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12}")

    tests = [
        ("a + b", {"a": 1.0, "b": 2.0}),
        ("a - b", {"a": 5.0, "b": 3.0}),
        ("a * b", {"a": 4.0, "b": 5.0}),
        ("a / b", {"a": 10.0, "b": 2.0}),
        ("a ^ b", {"a": 2.0, "b": 3.0}),
        ("a % b", {"a": 17.0, "b": 5.0}),
        ("-a", {"a": 5.0}),
    ]

    for formula, inputs in tests:
        stats = benchmark_formula(formula, formula, inputs)
        print(f"  {formula:50s} {stats['runs_per_second']:>12,.0f} "
              f"{stats['avg_time']*1000:>12.4f} {stats['total_time']:>12.4f}")


def test_chained_operations():
    print("\n--- BENCHMARK: CHAINED OPERATIONS ---")
    print(f"  {'Formula':50s} {'Runs/sec':>12s} {'Avg(ms)':>12s} {'Total(s)':>12s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12}")

    tests = [
        ("a + b + c", {"a": 1, "b": 2, "c": 3}),
        ("a + b + c + d + e", {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}),
        ("a * b + c * d", {"a": 2, "b": 3, "c": 4, "d": 5}),
        ("a + b * c - d / e", {"a": 10, "b": 3, "c": 5, "d": 100, "e": 4}),
        ("(a + b) * (c - d) / e", {"a": 7, "b": 3, "c": 9, "d": 2, "e": 5}),
        ("a ^ 2 + b ^ 2 - c ^ 2", {"a": 3, "b": 4, "c": 5}),
    ]

    for formula, inputs in tests:
        stats = benchmark_formula(formula, formula, inputs)
        print(f"  {formula:50s} {stats['runs_per_second']:>12,.0f} "
              f"{stats['avg_time']*1000:>12.4f} {stats['total_time']:>12.4f}")


def test_function_operations():
    print("\n--- BENCHMARK: FUNCTIONS ---")
    print(f"  {'Formula':50s} {'Runs/sec':>12s} {'Avg(ms)':>12s} {'Total(s)':>12s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12}")

    tests = [
        ("sin(x)", {"x": 1.0}),
        ("cos(x)", {"x": 1.0}),
        ("tan(x)", {"x": 0.5}),
        ("exp(x)", {"x": 1.0}),
        ("log(x)", {"x": 2.0}),
        ("sqrt(x)", {"x": 16.0}),
        ("abs(x)", {"x": -5.0}),
        ("asin(x)", {"x": 0.5}),
        ("sinh(x)", {"x": 1.0}),
        ("cosh(x)", {"x": 1.0}),
        ("tanh(x)", {"x": 0.5}),
        ("sigmoid(x)", {"x": 1.0}),
        ("erf(x)", {"x": 1.0}),
        ("floor(x)", {"x": 3.7}),
        ("ceil(x)", {"x": 3.2}),
        ("log2(x)", {"x": 8.0}),
        ("log10(x)", {"x": 1000.0}),
    ]

    for formula, inputs in tests:
        stats = benchmark_formula(formula, formula, inputs)
        print(f"  {formula:50s} {stats['runs_per_second']:>12,.0f} "
              f"{stats['avg_time']*1000:>12.4f} {stats['total_time']:>12.4f}")


def test_complex_expressions():
    print("\n--- BENCHMARK: COMPLEX EXPRESSIONS ---")
    print(f"  {'Formula':50s} {'Runs/sec':>12s} {'Avg(ms)':>12s} {'Total(s)':>12s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12}")

    tests = [
        ("sin(a) + cos(b)", {"a": 1.0, "b": 0.5}),
        ("sin(a) ^ 2 + cos(a) ^ 2", {"a": 1.0}),
        ("exp(a) * log(b) + c ^ d", {"a": 1.0, "b": 2.718, "c": 2.0, "d": 10.0}),
        ("sqrt(a ^ 2 + b ^ 2 + c ^ 2)", {"a": 1, "b": 2, "c": 2}),
        ("sin(cos(tan(x)))", {"x": 0.5}),
        ("log(exp(x)) + exp(log(x))", {"x": 3.0}),
        ("abs(sin(x)) + abs(cos(x))", {"x": 1.0}),
    ]

    for formula, inputs in tests:
        stats = benchmark_formula(formula, formula, inputs)
        print(f"  {formula:50s} {stats['runs_per_second']:>12,.0f} "
              f"{stats['avg_time']*1000:>12.4f} {stats['total_time']:>12.4f}")


def test_constants_performance():
    print("\n--- BENCHMARK: CONSTANTS ---")
    print(f"  {'Formula':50s} {'Runs/sec':>12s} {'Avg(ms)':>12s} {'Total(s)':>12s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12}")

    tests = [
        ("pi * r ^ 2", {"r": 5}),
        ("2 * pi * r", {"r": 10}),
        ("euler ^ x", {"x": 1}),
        ("4 * pi * r ^ 2", {"r": 6}),
        ("sqrt(2 * pi) * euler ^ (- x ^ 2 / 2)", {"x": 0}),
    ]

    for formula, inputs in tests:
        stats = benchmark_formula(formula, formula, inputs)
        print(f"  {formula:50s} {stats['runs_per_second']:>12,.0f} "
              f"{stats['avg_time']*1000:>12.4f} {stats['total_time']:>12.4f}")


def test_comparison_performance():
    print("\n--- BENCHMARK: COMPARISONS ---")
    print(f"  {'Formula':50s} {'Runs/sec':>12s} {'Avg(ms)':>12s} {'Total(s)':>12s}")
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*12}")

    tests = [
        ("a < b", {"a": 3, "b": 5}),
        ("a > b", {"a": 5, "b": 3}),
        ("a == b", {"a": 5, "b": 5}),
        ("a != b", {"a": 5, "b": 3}),
        ("a + 1 < b * 2", {"a": 3, "b": 5}),
    ]

    for formula, inputs in tests:
        stats = benchmark_formula(formula, formula, inputs)
        print(f"  {formula:50s} {stats['runs_per_second']:>12,.0f} "
              f"{stats['avg_time']*1000:>12.4f} {stats['total_time']:>12.4f}")


def test_massive_benchmark():
    print("\n--- BENCHMARK: MASSIVE ---")

    massive = ("sin(a) * cos(b) + sqrt(c ^ 2 + d ^ 2) - exp(-e) + "
               "log(f) * g ^ 2 + abs(h - i) + tan(j) + sinh(k) + "
               "cosh(l) + tanh(m) + n % o")
    inputs = {
        "a": 1.0, "b": 0.5, "c": 3.0, "d": 4.0,
        "e": 1.0, "f": 2.718, "g": 3.0, "h": 10,
        "i": 7, "j": 0.3, "k": 1.0, "l": 1.0, "m": 0.5,
        "n": 17, "o": 5
    }

    stats = benchmark_formula("massive", massive, inputs, runs=10000)
    print(f"  Formula: {massive[:60]}...")
    print(f"  Variables: 15  Nodes: 28")
    print(f"  Runs/sec:    {stats['runs_per_second']:>12,.0f}")
    print(f"  Avg time:    {stats['avg_time']*1000:>12.4f} ms")
    print(f"  Total time:  {stats['total_time']:>12.4f} s")


def test_node_count_scaling():
    print("\n--- BENCHMARK: NODE COUNT SCALING ---")
    print(f"  {'Nodes':>8s} {'Formula':40s} {'Runs/sec':>12s} {'Avg(ms)':>12s}")
    print(f"  {'-'*8} {'-'*40} {'-'*12} {'-'*12}")

    tests = [
        (1, "a", {"a": 1}),
        (2, "a + b", {"a": 1, "b": 2}),
        (4, "a + b * c - d", {"a": 1, "b": 2, "c": 3, "d": 4}),
        (6, "sin(a) + cos(b) * sqrt(c)", {"a": 1.0, "b": 0.5, "c": 9.0}),
        (8, "exp(a) * log(b) + sin(c) - cos(d) * sqrt(e)",
         {"a": 1.0, "b": 2.718, "c": 0.5, "d": 1.0, "e": 16.0}),
        (12, "sin(a) * cos(b) + sqrt(c^2 + d^2) + exp(-e) + log(f)",
         {"a": 1.0, "b": 0.5, "c": 3.0, "d": 4.0, "e": 1.0, "f": 2.718}),
        (28, "sin(a)*cos(b)+sqrt(c^2+d^2)-exp(-e)+log(f)*g^2+abs(h-i)+tan(j)+sinh(k)+cosh(l)+tanh(m)+n%o",
         {"a": 1.0, "b": 0.5, "c": 3.0, "d": 4.0, "e": 1.0, "f": 2.718,
          "g": 3.0, "h": 10, "i": 7, "j": 0.3, "k": 1.0, "l": 1.0, "m": 0.5, "n": 17, "o": 5}),
    ]

    for nodes, formula, inputs in tests:
        stats = benchmark_formula(formula, formula, inputs)
        short = formula if len(formula) <= 40 else formula[:37] + "..."
        print(f"  {nodes:>8d} {short:40s} {stats['runs_per_second']:>12,.0f} "
              f"{stats['avg_time']*1000:>12.4f}")


if __name__ == "__main__":
    test_scalar_operations()
    test_chained_operations()
    test_function_operations()
    test_complex_expressions()
    test_constants_performance()
    test_comparison_performance()
    test_massive_benchmark()
    test_node_count_scaling()
