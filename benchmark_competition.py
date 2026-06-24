#!/usr/bin/env python3
"""
Benchmark: Formula2ONNX (C Backend) vs NumPy
Compares execution time for various mathematical operations.
"""

import time
import numpy as np
from formula2onnx import from_formula, run
import ctypes
import os

# Load the C backend library
C_LIB = None

# Load the C backend library as a Python module
C_LIB = None

try:
    import lattice.lattice_backend as c_module
    C_LIB = c_module
    print(f"✅ C backend loaded successfully (lattice.lattice_backend)")
except Exception as e:
    print(f"⚠️  Could not load C backend: {e}")
    C_LIB = None


def run_c_kernel(op_name, inputs, size):
    """Run operation using C backend directly"""
    if C_LIB is None:
        return None
    
    try:
        # Call the execute function: execute(op_name, inputs_list, size, constants)
        result = C_LIB.execute(op_name, inputs, size, None)
        return result
    except Exception as e:
        print(f"C kernel error: {e}")
        return None


def benchmark_operation(name, formula, variables, op_type=None, data_size=10_000_000, iterations=10):
    """
    Benchmark a single operation for NumPy, ONNX, and C backend.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"Formula: {formula}")
    print(f"Data size: {data_size:,} elements")
    print(f"{'='*60}")
    
    # Generate random input data
    np.random.seed(42)
    inputs_dict = {}
    inputs_list = []
    for var in variables:
        arr = np.random.rand(data_size).astype(np.float64)
        inputs_dict[var] = arr
        inputs_list.append(arr)
    
    # Benchmark NumPy
    print("\n⏱️  Benchmarking NumPy...")
    numpy_times = []
    result_numpy = None
    for i in range(iterations):
        start = time.perf_counter()
        
        local_vars = inputs_dict.copy()
        local_vars['sin'] = np.sin
        local_vars['cos'] = np.cos
        local_vars['tan'] = np.tan
        local_vars['exp'] = np.exp
        local_vars['log'] = np.log
        local_vars['sqrt'] = np.sqrt
        local_vars['abs'] = np.abs
        
        safe_formula = formula.replace('^', '**')
        try:
            result_numpy = eval(safe_formula, {"__builtins__": {}}, local_vars)
        except Exception as e:
            print(f"NumPy evaluation error: {e}")
            continue
            
        end = time.perf_counter()
        numpy_times.append(end - start)
    
    if not numpy_times:
        print("❌ NumPy benchmark failed")
        return None
        
    avg_numpy = np.mean(numpy_times) * 1000
    
    # Benchmark C Backend (direct)
    lattice_time = None
    result_c = None
    if C_LIB and op_type:
        print("⏱️  Benchmarking C Backend (Direct)...")
        c_times = []
        for i in range(iterations):
            start = time.perf_counter()
            result_c = run_c_kernel(op_type, inputs_list, data_size)
            end = time.perf_counter()
            c_times.append(end - start)
        
        if result_c is not None:
            avg_c = np.mean(c_times) * 1000
            lattice_time = avg_c
    
    # Calculate speedup
    if lattice_time and lattice_time > 0:
        speedup = avg_numpy / lattice_time
    else:
        speedup = None
    
    # Verify correctness
    is_correct = False
    max_diff = None
    if result_numpy is not None and result_c is not None:
        max_diff = np.max(np.abs(result_numpy - result_c))
        is_correct = max_diff < 1e-6
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  NumPy:          {avg_numpy:8.4f} ms")
    if lattice_time:
        print(f"  C Backend:      {lattice_time:8.4f} ms")
        if speedup and speedup >= 1.0:
            print(f"  Speedup:        {speedup:8.2f}x 🚀 (C faster)")
        elif speedup:
            print(f"  Slowdown:       {1/speedup:8.2f}x (NumPy faster)")
    
    if is_correct:
        print(f"  Max Error:      {max_diff:.2e} ✅")
    elif max_diff is not None:
        print(f"  Max Error:      {max_diff:.2e} ❌")
    
    return {
        'name': name,
        'numpy_time': avg_numpy,
        'c_time': lattice_time,
        'speedup': speedup,
        'correct': is_correct
    }

def main():
    print("="*70)
    print(" " * 15 + "FORMULA2ONNX vs NUMPY BENCHMARK")
    print(" " * 15 + "(C Backend Performance Comparison)")
    print("="*70)
    
    benchmarks = [
        # Basic Operations with their C kernel types
        ("Addition", "a + b", ["a", "b"], "add"),
        ("Subtraction", "a - b", ["a", "b"], "sub"),
        ("Multiplication", "a * b", ["a", "b"], "mul"),
        ("Division", "a / b", ["a", "b"], "div"),
        
        # Fused Operations (where C backend should shine)
        ("FMA (Fused Multiply-Add)", "a * b + c", ["a", "b", "c"], "fma"),
        ("Fused Hypot", "sqrt(a**2 + b**2)", ["a", "b"], "fused_hypot"),
        ("Fused Sin2+Cos2", "sin(a)**2 + cos(a)**2", ["a"], "fused_sin2cos2"),
        ("Fused Exp Decay", "a * exp(-b * c)", ["a", "b", "c"], "fused_exp_decay"),
        ("Fused Gaussian", "a * exp(-0.5 * ((d - b) / c)**2)", ["a", "b", "c", "d"], "fused_gaussian"),
        
        # Math Functions
        ("Sine", "sin(a)", ["a"], "sin"),
        ("Cosine", "cos(a)", ["a"], "cos"),
        ("Exponential", "exp(a)", ["a"], "exp"),
        ("Square Root", "sqrt(a)", ["a"], "sqrt"),
    ]
    
    results = []
    
    for bench in benchmarks:
        name, formula, vars_list, op_type = bench
        result = benchmark_operation(name, formula, vars_list, op_type=op_type)
        if result:
            results.append(result)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Operation':<35} {'NumPy(ms)':>10} {'Lattice(ms)':>12} {'Speedup':>10}")
    print("-"*70)
    
    total_numpy = 0
    total_lattice = 0
    
    for r in results:
        if r and r['c_time'] is not None:
            total_numpy += r['numpy_time']
            total_lattice += r['c_time']
            status = "✅" if r['correct'] else "❌"
            print(f"{r['name']:<35} {r['numpy_time']:>10.4f} {r['c_time']:>12.4f} {r['speedup']:>8.2f}x {status}")
    
    print("-"*70)
    overall_speedup = total_numpy / total_lattice if total_lattice > 0 else 0
    print(f"{'TOTAL':<35} {total_numpy:>10.4f} {total_lattice:>12.4f} {overall_speedup:>8.2f}x")
    print("="*70)
    
    if overall_speedup > 1.0:
        print(f"\n🎉 C backend is {overall_speedup:.2f}x faster overall!")
    elif overall_speedup < 1.0 and overall_speedup > 0:
        print(f"\n⚠️  NumPy is {1/overall_speedup:.2f}x faster overall.")
    else:
        print("\n⚠️  No C backend results available.")
    
    print("\nNote: The C backend shows maximum benefit on fused operations")
    print("      where multiple calculations are combined into a single kernel.")

if __name__ == "__main__":
    main()
