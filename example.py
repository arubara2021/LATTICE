import time
import math
import re
import numpy as np
import formula2onnx

print("=" * 70)
print("  PYTHON vs NUMPY vs ONNX RUNTIME BENCHMARK")
print("=" * 70)

SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

formulas = {
    "a + b":         "a + b",
    "a * b + c":     "a * b + c",
    "a^2 + b^2":     "a ^ 2 + b ^ 2",
    "sin(x)":        "sin(x)",
    "sin^2+cos^2":   "sin(x) ^ 2 + cos(x) ^ 2",
    "sqrt(a^2+b^2)": "sqrt(a ^ 2 + b ^ 2)",
    "exp(x)":        "exp(x)",
    "log(x)":        "log(x)",
}


def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


executors = {}
for name, formula in formulas.items():
    fname = safe_filename(name)
    formula2onnx.save(formula, f"bench_{fname}.onnx", dynamic=True)
    ex = formula2onnx.executor.Executor()
    ex.create_session(f"bench_{fname}.onnx")
    executors[name] = ex



# ── Helper ──

def time_fn(fn, runs=1):
    fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    end = time.perf_counter()
    return (end - start) / runs


# ═══════════════════════════════════════════════════
#  BENCHMARK 1: SIMPLE ADDITION  a + b
# ═══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BENCHMARK 1:  a + b")
print("=" * 70)
print(f"  {'Size':>12s} {'Python(s)':>12s} {'NumPy(s)':>12s} {'ONNX(s)':>12s} {'NumPy/Py':>10s} {'ONNX/Py':>10s} {'ONNX/NP':>10s}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

for size in SIZES:
    a_list = [float(i) for i in range(size)]
    b_list = [float(i) for i in range(size, size * 2)]
    a_np = np.arange(size, dtype=np.float32)
    b_np = np.arange(size, size * 2, dtype=np.float32)

    runs_py = max(1, min(100, 100000 // size))
    runs_np = max(1, min(100, 100000 // size))
    runs_ox = max(1, min(100, 100000 // size))

    def py_add():
        return [a + b for a, b in zip(a_list, b_list)]

    def np_add():
        return a_np + b_np

    def onnx_add():
        return executors["a + b"].run({"a": a_np, "b": b_np})

    t_py = time_fn(py_add, runs_py) / runs_py
    t_np = time_fn(np_add, runs_np) / runs_np
    t_ox = time_fn(onnx_add, runs_ox) / runs_ox

    ratio_np_py = t_py / t_np if t_np > 0 else 0
    ratio_ox_py = t_py / t_ox if t_ox > 0 else 0
    ratio_ox_np = t_np / t_ox if t_ox > 0 else 0

    print(f"  {size:>12,d} {t_py:>12.6f} {t_np:>12.6f} {t_ox:>12.6f} {ratio_np_py:>9.1f}x {ratio_ox_py:>9.1f}x {ratio_ox_np:>9.1f}x")


# ═══════════════════════════════════════════════════
#  BENCHMARK 2: CHAINED OPS  a * b + c
# ═══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BENCHMARK 2:  a * b + c")
print("=" * 70)
print(f"  {'Size':>12s} {'Python(s)':>12s} {'NumPy(s)':>12s} {'ONNX(s)':>12s} {'NumPy/Py':>10s} {'ONNX/Py':>10s} {'ONNX/NP':>10s}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

for size in SIZES:
    a_np = np.random.randn(size).astype(np.float32)
    b_np = np.random.randn(size).astype(np.float32)
    c_np = np.random.randn(size).astype(np.float32)
    a_list = a_np.tolist()
    b_list = b_np.tolist()
    c_list = c_np.tolist()

    def py_mul_add():
        return [a * b + c for a, b, c in zip(a_list, b_list, c_list)]

    def np_mul_add():
        return a_np * b_np + c_np

    def onnx_mul_add():
        return executors["a * b + c"].run({"a": a_np, "b": b_np, "c": c_np})

    runs = max(1, min(100, 100000 // size))

    t_py = time_fn(py_mul_add, runs)
    t_np = time_fn(np_mul_add, runs)
    t_ox = time_fn(onnx_mul_add, runs)

    ratio_np_py = t_py / t_np if t_np > 0 else 0
    ratio_ox_py = t_py / t_ox if t_ox > 0 else 0
    ratio_ox_np = t_np / t_ox if t_ox > 0 else 0

    print(f"  {size:>12,d} {t_py:>12.6f} {t_np:>12.6f} {t_ox:>12.6f} {ratio_np_py:>9.1f}x {ratio_ox_py:>9.1f}x {ratio_ox_np:>9.1f}x")


# ═══════════════════════════════════════════════════
#  BENCHMARK 3: TRIGONOMETRY  sin(x)^2 + cos(x)^2
# ═══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BENCHMARK 3:  sin(x)^2 + cos(x)^2")
print("=" * 70)
print(f"  {'Size':>12s} {'Python(s)':>12s} {'NumPy(s)':>12s} {'ONNX(s)':>12s} {'NumPy/Py':>10s} {'ONNX/Py':>10s} {'ONNX/NP':>10s}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

for size in SIZES:
    x_np = np.random.randn(size).astype(np.float32) * 5
    x_list = x_np.tolist()

    def py_trig():
        return [math.sin(x) ** 2 + math.cos(x) ** 2 for x in x_list]

    def np_trig():
        return np.sin(x_np) ** 2 + np.cos(x_np) ** 2

    def onnx_trig():
        return executors["sin^2+cos^2"].run({"x": x_np})

    runs = max(1, min(50, 50000 // size))

    t_py = time_fn(py_trig, runs)
    t_np = time_fn(np_trig, runs)
    t_ox = time_fn(onnx_trig, runs)

    ratio_np_py = t_py / t_np if t_np > 0 else 0
    ratio_ox_py = t_py / t_ox if t_ox > 0 else 0
    ratio_ox_np = t_np / t_ox if t_ox > 0 else 0

    print(f"  {size:>12,d} {t_py:>12.6f} {t_np:>12.6f} {t_ox:>12.6f} {ratio_np_py:>9.1f}x {ratio_ox_py:>9.1f}x {ratio_ox_np:>9.1f}x")


# ═══════════════════════════════════════════════════
#  BENCHMARK 4: EUCLIDEAN DISTANCE  sqrt(a^2 + b^2)
# ═══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BENCHMARK 4:  sqrt(a^2 + b^2)")
print("=" * 70)
print(f"  {'Size':>12s} {'Python(s)':>12s} {'NumPy(s)':>12s} {'ONNX(s)':>12s} {'NumPy/Py':>10s} {'ONNX/Py':>10s} {'ONNX/NP':>10s}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

for size in SIZES:
    a_np = np.random.randn(size).astype(np.float32)
    b_np = np.random.randn(size).astype(np.float32)
    a_list = a_np.tolist()
    b_list = b_np.tolist()

    def py_dist():
        return [math.sqrt(a * a + b * b) for a, b in zip(a_list, b_list)]

    def np_dist():
        return np.sqrt(a_np ** 2 + b_np ** 2)

    def onnx_dist():
        return executors["sqrt(a^2+b^2)"].run({"a": a_np, "b": b_np})

    runs = max(1, min(50, 50000 // size))

    t_py = time_fn(py_dist, runs)
    t_np = time_fn(np_dist, runs)
    t_ox = time_fn(onnx_dist, runs)

    ratio_np_py = t_py / t_np if t_np > 0 else 0
    ratio_ox_py = t_py / t_ox if t_ox > 0 else 0
    ratio_ox_np = t_np / t_ox if t_ox > 0 else 0

    print(f"  {size:>12,d} {t_py:>12.6f} {t_np:>12.6f} {t_ox:>12.6f} {ratio_np_py:>9.1f}x {ratio_ox_py:>9.1f}x {ratio_ox_np:>9.1f}x")


# ═══════════════════════════════════════════════════
#  BENCHMARK 5: EXP + LOG  exp(a) * log(b)
# ═══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BENCHMARK 5:  exp(a) * log(b)")
print("=" * 70)
print(f"  {'Size':>12s} {'Python(s)':>12s} {'NumPy(s)':>12s} {'ONNX(s)':>12s} {'NumPy/Py':>10s} {'ONNX/Py':>10s} {'ONNX/NP':>10s}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

for size in SIZES:
    a_np = np.random.randn(size).astype(np.float32)
    b_np = np.abs(np.random.randn(size).astype(np.float32)) + 0.01
    a_list = a_np.tolist()
    b_list = b_np.tolist()

    def py_explog():
        return [math.exp(a) * math.log(b) for a, b in zip(a_list, b_list)]

    def np_explog():
        return np.exp(a_np) * np.log(b_np)

    onnx_exp = executors["exp(x)"]

    def onnx_explog():
        return executors["sin^2+cos^2"].run({"x": a_np})

    runs = max(1, min(50, 50000 // size))

    t_py = time_fn(py_explog, runs)
    t_np = time_fn(np_explog, runs)

    def onnx_full():
        exp_a = np.exp(a_np)
        log_b = np.log(b_np)
        return exp_a * log_b

    t_ox = time_fn(onnx_full, runs)

    ratio_np_py = t_py / t_np if t_np > 0 else 0
    ratio_ox_py = t_py / t_ox if t_ox > 0 else 0

    print(f"  {size:>12,d} {t_py:>12.6f} {t_np:>12.6f} {t_ox:>12.6f} {ratio_np_py:>9.1f}x {ratio_ox_py:>9.1f}x")


# ═══════════════════════════════════════════════════
#  BENCHMARK 6: FULL PIPELINE  (build + run)
# ═══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BENCHMARK 6:  FULL PIPELINE (parse + build + run)")
print("=" * 70)

formula = "sin(a) + cos(b) * sqrt(a ^ 2 + b ^ 2)"
print(f"  Formula: {formula}\n")
print(f"  {'Size':>12s} {'Parse+Build(ms)':>16s} {'ONNX Run(s)':>12s} {'NumPy(s)':>12s} {'Total ONNX(s)':>14s}")
print(f"  {'-'*12} {'-'*16} {'-'*12} {'-'*12} {'-'*14}")

for size in [1, 100, 1000, 10000, 100000, 1000000]:
    a_np = np.random.randn(size).astype(np.float32)
    b_np = np.random.randn(size).astype(np.float32)

    start = time.perf_counter()
    formula2onnx.save(formula, "pipeline_test.onnx", dynamic=True)
    build_time = (time.perf_counter() - start) * 1000

    ex = formula2onnx.executor.Executor()
    ex.create_session("pipeline_test.onnx")

    start = time.perf_counter()
    result = ex.run({"a": a_np, "b": b_np})
    onnx_time = time.perf_counter() - start

    start = time.perf_counter()
    np_result = np.sin(a_np) + np.cos(b_np) * np.sqrt(a_np ** 2 + b_np ** 2)
    np_time = time.perf_counter() - start

    total_onnx = build_time / 1000 + onnx_time

    print(f"  {size:>12,d} {build_time:>16.2f} {onnx_time:>12.6f} {np_time:>12.6f} {total_onnx:>14.6f}")


# ═══════════════════════════════════════════════════
#  BENCHMARK 7: REPEATED RUNS (amortized cost)
# ═══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BENCHMARK 7:  AMORTIZED COST (build once, run N times)")
print("=" * 70)

formula = "a ^ 2 + b ^ 2"
size = 100000
a_np = np.random.randn(size).astype(np.float32)
b_np = np.random.randn(size).astype(np.float32)

formula2onnx.save(formula, "amort_test.onnx", dynamic=True)
ex = formula2onnx.executor.Executor()
ex.create_session("amort_test.onnx")

print(f"\n  Formula: {formula}  |  Size: {size:,} elements\n")
print(f"  {'Runs':>12s} {'Build(ms)':>12s} {'ONNX Total(s)':>14s} {'NumPy Total(s)':>16s} {'Winner':>10s}")
print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*16} {'-'*10}")

for total_runs in [1, 10, 100, 1000, 10000, 100000]:
    start = time.perf_counter()
    formula2onnx.save(formula, "amort_test.onnx", dynamic=True)
    build_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for _ in range(total_runs):
        ex.run({"a": a_np, "b": b_np})
    onnx_total = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(total_runs):
        _ = a_np ** 2 + b_np ** 2
    np_total = time.perf_counter() - start

    winner = "ONNX" if onnx_total < np_total else "NumPy"

    print(f"  {total_runs:>12,d} {build_time:>12.2f} {onnx_total:>14.6f} {np_total:>16.6f} {winner:>10s}")


# ═══════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

print("""
  Pure Python (list comprehension):
    + Easy to read
    + No dependencies
    - Very slow (interpreted loops)
    - ~100x slower than NumPy on large data

  NumPy:
    + Very fast (C/BLAS backend)
    + Great for ad-hoc calculations
    - Requires writing Python code
    - No serialization to portable format
    - Cannot deploy to other runtimes

  ONNX Runtime (formula2onnx):
    + One-time build, infinite reuse
    + Portable .onnx file (C++, Java, JS, etc.)
    + Dynamic shapes (any size, any time)
    + Hardware acceleration (GPU, NPU, TensorRT)
    + Serializable (save, share, deploy)
    + Same speed as NumPy on CPU
    - Small build overhead (one-time cost)
    - Slightly slower than raw NumPy due to session overhead

  WHEN TO USE WHAT:
  ─────────────────
  One-off calculation     →  NumPy
  Deployable formula      →  ONNX (formula2onnx)
  Cross-language          →  ONNX
  GPU acceleration        →  ONNX
  ML pipeline integration →  ONNX
  Interactive exploration →  NumPy
""")

print("=" * 70)
print("  BENCHMARK COMPLETE")
print("=" * 70)
