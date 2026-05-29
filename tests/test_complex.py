import math
from helpers import run_test, print_summary


def test_nested_arithmetic():
    print("\n--- COMPLEX: NESTED ARITHMETIC ---")

    run_test("((((a+b)+c)+d)+e)", "((((a+b)+c)+d)+e)",
             {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, 15.0)
    run_test("a+(b+(c+(d+(e+f))))", "a+(b+(c+(d+(e+f))))",
             {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}, 21.0)
    run_test("(a+b)*(c+d)*(e+f)", "(a+b)*(c+d)*(e+f)",
             {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}, 189.0)
    run_test("deep parens", "((a + b) * (c + d)) / ((e - f) * (g - h))",
             {"a": 1, "b": 2, "c": 3, "d": 4, "e": 10, "f": 7, "g": 5, "h": 3}, 10.5)


def test_mixed_functions():
    print("\n--- COMPLEX: MIXED FUNCTIONS ---")

    run_test("sin(a)+cos(b)", "sin(a) + cos(b)",
             {"a": 1.0, "b": 0.5}, math.sin(1.0) + math.cos(0.5))
    run_test("exp(a)*log(b)", "exp(a) * log(b)",
             {"a": 1.0, "b": math.e}, math.e)
    run_test("sqrt(a^2+b^2)", "sqrt(a ^ 2 + b ^ 2)",
             {"a": 3, "b": 4}, 5.0)
    run_test("abs(sin(x))", "abs(sin(x))", {"x": -1.0}, abs(math.sin(-1.0)))
    run_test("pow(sqrt(a),2)", "sqrt(a) ^ 2", {"a": 7}, 7.0)
    run_test("log(exp(x))", "log(exp(x))", {"x": 3.0}, 3.0, tolerance=1e-4)
    run_test("exp(log(x))", "exp(log(x))", {"x": 5.0}, 5.0, tolerance=1e-4)
    run_test("sqrt(x^2)", "sqrt(x ^ 2)", {"x": -7}, 7.0)
    run_test("log10(10^x)", "log10(10 ^ x)", {"x": 3}, 3.0)
    run_test("log2(2^x)", "log2(2 ^ x)", {"x": 5}, 5.0)


def test_nested_functions():
    print("\n--- COMPLEX: NESTED FUNCTIONS ---")

    run_test("sin(cos(x))", "sin(cos(x))", {"x": 1.0}, math.sin(math.cos(1.0)))
    run_test("cos(sin(x))", "cos(sin(x))", {"x": 1.0}, math.cos(math.sin(1.0)))
    run_test("exp(sin(x))", "exp(sin(x))", {"x": 0.5}, math.exp(math.sin(0.5)))
    run_test("sin(exp(x))", "sin(exp(x))", {"x": 0.1}, math.sin(math.exp(0.1)))
    run_test("sqrt(abs(x))", "sqrt(abs(x))", {"x": -16}, 4.0)
    run_test("log(abs(x))", "log(abs(x))", {"x": -math.e}, 1.0, tolerance=1e-4)
    run_test("deep nest", "sin(cos(tan(x)))", {"x": 0.5},
             math.sin(math.cos(math.tan(0.5))))


def test_factorial_and_abs():
    print("\n--- COMPLEX: FACTORIAL AND ABS ---")

    run_test("3!", "3!", {}, 6.0)
    run_test("5!", "5!", {}, 120.0)
    run_test("3!+2", "3! + 2", {}, 8.0)
    run_test("3!^2", "3! ^ 2", {}, 36.0)
    run_test("|x-5|", "|x - 5|", {"x": 3}, 2.0)
    run_test("|x-5|+|y-10|", "|x - 5| + |y - 10|", {"x": 2, "y": 15}, 8.0)
    run_test("|sin(x)|", "|sin(x)|", {"x": -1.0}, abs(math.sin(-1.0)))
    run_test("factorial+abs", "3! + |x - 10|", {"x": 7}, 9.0)


def test_physics_formulas():
    print("\n--- COMPLEX: PHYSICS ---")

    run_test("kinematics", "u * t + 0.5 * a * t ^ 2",
             {"u": 10, "a": 9.8, "t": 5}, 172.5)
    run_test("kinetic energy", "0.5 * m * v ^ 2",
             {"m": 70, "v": 10}, 3500.0)
    run_test("potential energy", "m * g * h",
             {"m": 5, "g": 9.8, "h": 20}, 980.0)
    run_test("ohms law", "I * R", {"I": 2.5, "R": 100}, 250.0)
    run_test("power", "I ^ 2 * R", {"I": 3, "R": 50}, 450.0)
    run_test("centripetal", "v ^ 2 / r", {"v": 20, "r": 5}, 80.0)
    run_test("pendulum", "2 * pi * sqrt(L / g)",
             {"L": 1.0, "g": 9.8}, 2 * math.pi * math.sqrt(1.0 / 9.8))
    run_test("lorentz", "1 / sqrt(1 - v ^ 2 / c ^ 2)",
             {"v": 100000, "c": 299792458}, 1.0000001, tolerance=1e-4)
    run_test("schwarzschild", "2 * G * M / (c ^ 2 * r)",
             {"G": 6.674e-11, "M": 5.972e24, "c": 299792458, "r": 6.371e6},
             2 * 6.674e-11 * 5.972e24 / (299792458 ** 2 * 6.371e6),
             tolerance=1e-2)
    run_test("stefan boltzmann", "sigma * A * T ^ 4",
             {"sigma": 5.67e-8, "A": 1.0, "T": 5778},
             5.67e-8 * 1.0 * 5778 ** 4, tolerance=1e6)


def test_engineering_formulas():
    print("\n--- COMPLEX: ENGINEERING ---")

    run_test("dB gain", "20 * log10(V_out / V_in)",
             {"V_out": 10, "V_in": 1}, 20.0)
    run_test("impedance", "sqrt(R ^ 2 + (X_L - X_C) ^ 2)",
             {"R": 3, "X_L": 8, "X_C": 4}, 5.0)
    run_test("BMI", "weight / (height ^ 2)",
             {"weight": 70, "height": 1.75}, 70 / 1.75 ** 2, tolerance=1e-2)
    run_test("compound interest", "P * (1 + r / n) ^ (n * t)",
             {"P": 1000, "r": 0.05, "n": 12, "t": 10},
             1000 * (1 + 0.05 / 12) ** (12 * 10), tolerance=1e-1)
    run_test("wind chill", "13.12 + 0.6215 * T - 11.37 * V ^ 0.16 + 0.3965 * T * V ^ 0.16",
             {"T": -10, "V": 20},
             13.12 + 0.6215 * (-10) - 11.37 * 20 ** 0.16 + 0.3965 * (-10) * 20 ** 0.16,
             tolerance=1e-1)


def test_massive_formula():
    print("\n--- COMPLEX: MASSIVE FORMULA ---")

    massive = ("sin(a) * cos(b) + sqrt(c ^ 2 + d ^ 2) - exp(-e) + "
               "log(f) * g ^ 2 + abs(h - i) + tan(j) + sinh(k) + "
               "cosh(l) + tanh(m) + n % o")
    inputs = {
        "a": 1.0, "b": 0.5, "c": 3.0, "d": 4.0,
        "e": 1.0, "f": 2.718, "g": 3.0, "h": 10,
        "i": 7, "j": 0.3, "k": 1.0, "l": 1.0, "m": 0.5,
        "n": 17, "o": 5
    }
    expected = (
        math.sin(1.0) * math.cos(0.5) +
        math.sqrt(9.0 + 16.0) -
        math.exp(-1.0) +
        math.log(2.718) * 9.0 +
        abs(10 - 7) +
        math.tan(0.3) +
        math.sinh(1.0) +
        math.cosh(1.0) +
        math.tanh(0.5) +
        17 % 5
    )
    run_test("massive 15 vars 14 ops", massive, inputs, expected, tolerance=1e-2)

    massive2 = ("(sin(x) + cos(x)) ^ 2 + (sin(x) - cos(x)) ^ 2 + "
                "sqrt(a ^ 2 + b ^ 2 + c ^ 2) + log(euler ^ x) + "
                "abs(-pi) + factorial(4)")
    inputs2 = {"x": 1.0, "a": 1.0, "b": 2.0, "c": 2.0}
    expected2 = (
        (math.sin(1.0) + math.cos(1.0)) ** 2 +
        (math.sin(1.0) - math.cos(1.0)) ** 2 +
        math.sqrt(1 + 4 + 4) +
        1.0 +
        math.pi +
        24.0
    )
    run_test("massive trig + algebra", massive2, inputs2, expected2, tolerance=1e-1)


def test_extreme_values():
    print("\n--- COMPLEX: EXTREME VALUES ---")

    run_test("very large", "a + b", {"a": 1e15, "b": 1e15}, 2e15, tolerance=1e10)
    run_test("very small", "a * b", {"a": 1e-10, "b": 1e-10}, 1e-20, tolerance=1e-22)
    run_test("many operations", "a + b - c * d / e ^ f",
             {"a": 100, "b": 200, "c": 50, "d": 2, "e": 2, "f": 3}, 287.5)
    run_test("zero result", "a - a", {"a": 42}, 0.0)
    run_test("identity chain", "exp(log(a))", {"a": 7}, 7.0, tolerance=1e-4)
    run_test("cancel out", "sin(x) ^ 2 + cos(x) ^ 2 - 1", {"x": 100.0}, 0.0,
             tolerance=1e-5)


if __name__ == "__main__":
    test_nested_arithmetic()
    test_mixed_functions()
    test_nested_functions()
    test_factorial_and_abs()
    test_physics_formulas()
    test_engineering_formulas()
    test_massive_formula()
    test_extreme_values()
    print_summary("test_complex.py")
