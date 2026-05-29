import math
from helpers import run_test, print_summary


def test_sin():
    print("\n--- TRIG: SIN ---")

    run_test("sin(0)", "sin(x)", {"x": 0}, 0.0)
    run_test("sin(pi/6)", "sin(x)", {"x": math.pi / 6}, 0.5, tolerance=1e-4)
    run_test("sin(pi/4)", "sin(x)", {"x": math.pi / 4}, 0.707107, tolerance=1e-4)
    run_test("sin(pi/3)", "sin(x)", {"x": math.pi / 3}, 0.866025, tolerance=1e-4)
    run_test("sin(pi/2)", "sin(x)", {"x": math.pi / 2}, 1.0, tolerance=1e-4)
    run_test("sin(pi)", "sin(x)", {"x": math.pi}, 0.0, tolerance=1e-4)
    run_test("sin(-x)", "sin(-x)", {"x": 1.0}, -math.sin(1.0))
    run_test("sin(1)", "sin(x)", {"x": 1.0}, 0.841471, tolerance=1e-4)


def test_cos():
    print("\n--- TRIG: COS ---")

    run_test("cos(0)", "cos(x)", {"x": 0}, 1.0)
    run_test("cos(pi/3)", "cos(x)", {"x": math.pi / 3}, 0.5, tolerance=1e-4)
    run_test("cos(pi/2)", "cos(x)", {"x": math.pi / 2}, 0.0, tolerance=1e-4)
    run_test("cos(pi)", "cos(x)", {"x": math.pi}, -1.0, tolerance=1e-4)
    run_test("cos(1)", "cos(x)", {"x": 1.0}, 0.540302, tolerance=1e-4)


def test_tan():
    print("\n--- TRIG: TAN ---")

    run_test("tan(0)", "tan(x)", {"x": 0}, 0.0)
    run_test("tan(pi/4)", "tan(x)", {"x": math.pi / 4}, 1.0, tolerance=1e-4)
    run_test("tan(1)", "tan(x)", {"x": 1.0}, 1.557408, tolerance=1e-4)


def test_inverse_trig():
    print("\n--- TRIG: INVERSE ---")

    run_test("asin(0)", "asin(x)", {"x": 0}, 0.0)
    run_test("asin(1)", "asin(x)", {"x": 1}, math.pi / 2, tolerance=1e-4)
    run_test("asin(0.5)", "asin(x)", {"x": 0.5}, math.pi / 6, tolerance=1e-4)
    run_test("acos(0)", "acos(x)", {"x": 0}, math.pi / 2, tolerance=1e-4)
    run_test("acos(1)", "acos(x)", {"x": 1}, 0.0, tolerance=1e-4)
    run_test("atan(0)", "atan(x)", {"x": 0}, 0.0)
    run_test("atan(1)", "atan(x)", {"x": 1}, math.pi / 4, tolerance=1e-4)
    run_test("asin(x)+acos(x)=pi/2", "asin(x) + acos(x)", {"x": 0.5},
             math.pi / 2, tolerance=1e-4)


def test_hyperbolic():
    print("\n--- TRIG: HYPERBOLIC ---")

    run_test("sinh(0)", "sinh(x)", {"x": 0}, 0.0)
    run_test("cosh(0)", "cosh(x)", {"x": 0}, 1.0)
    run_test("tanh(0)", "tanh(x)", {"x": 0}, 0.0)
    run_test("sinh(1)+cosh(1)=e", "sinh(x) + cosh(x)", {"x": 1.0},
             math.e, tolerance=1e-4)
    run_test("tanh(x)", "tanh(x)", {"x": 1.0}, 0.761594, tolerance=1e-4)


def test_trig_identities():
    print("\n--- TRIG: IDENTITIES ---")

    run_test("sin^2+cos^2=1", "sin(x) ^ 2 + cos(x) ^ 2", {"x": 0.5}, 1.0)
    run_test("sin^2+cos^2=1", "sin(x) ^ 2 + cos(x) ^ 2", {"x": 1.0}, 1.0)
    run_test("sin^2+cos^2=1", "sin(x) ^ 2 + cos(x) ^ 2", {"x": 2.0}, 1.0)
    run_test("sin^2+cos^2=1", "sin(x) ^ 2 + cos(x) ^ 2", {"x": 100.0}, 1.0)
    run_test("sin(-x)=-sin(x)", "sin(-x) + sin(x)", {"x": 1.5}, 0.0)
    run_test("cos(-x)=cos(x)", "cos(-x) - cos(x)", {"x": 1.5}, 0.0)
    run_test("tan=sin/cos", "tan(x) * cos(x) - sin(x)", {"x": 0.7}, 0.0,
             tolerance=1e-4)
    run_test("sin(a+b)=sacos+cosa*sinb",
             "sin(a) * cos(b) + cos(a) * sin(b) - sin(a + b)",
             {"a": 0.3, "b": 0.4}, 0.0, tolerance=1e-4)


def test_trig_with_constants():
    print("\n--- TRIG: WITH CONSTANTS ---")

    run_test("sin(pi)", "sin(pi)", {}, 0.0, tolerance=1e-4)
    run_test("cos(pi)", "cos(pi)", {}, -1.0, tolerance=1e-4)
    run_test("sin(pi/2)", "sin(pi / 2)", {}, 1.0, tolerance=1e-4)


def test_trig_in_physics():
    print("\n--- TRIG: PHYSICS ---")

    run_test("projectile range", "v ^ 2 * sin(2 * theta) / g",
             {"v": 20, "theta": math.pi / 4, "g": 9.8}, 40.816, tolerance=1e-1)
    run_test("wave displacement", "A * sin(2 * pi * f * t)",
             {"A": 5, "pi": 3.14159, "f": 440, "t": 0.001}, 5 * math.sin(2 * math.pi * 440 * 0.001),
             tolerance=1e-2)
    run_test("pendulum angle", "theta * cos(sqrt(g / L) * t)",
             {"theta": 0.1, "g": 9.8, "L": 1.0, "t": 1.0},
             0.1 * math.cos(math.sqrt(9.8) * 1.0), tolerance=1e-2)


if __name__ == "__main__":
    test_sin()
    test_cos()
    test_tan()
    test_inverse_trig()
    test_hyperbolic()
    test_trig_identities()
    test_trig_with_constants()
    test_trig_in_physics()
    print_summary("test_trig.py")
