from helpers import run_test, print_summary


def test_addition():
    print("\n--- BASIC OPS: ADDITION ---")

    run_test("a + b", "a + b", {"a": 3, "b": 5}, 8.0)
    run_test("a + b + c", "a + b + c", {"a": 1, "b": 2, "c": 3}, 6.0)
    run_test("a + b + c + d + e", "a + b + c + d + e",
             {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, 15.0)
    run_test("a + (-b)", "a + (-b)", {"a": 10, "b": 3}, 7.0)
    run_test("neg + neg", "-a + -b", {"a": 3, "b": 5}, -8.0)


def test_subtraction():
    print("\n--- BASIC OPS: SUBTRACTION ---")

    run_test("a - b", "a - b", {"a": 10, "b": 3}, 7.0)
    run_test("a - b - c", "a - b - c", {"a": 10, "b": 3, "c": 2}, 5.0)
    run_test("b - a", "b - a", {"a": 10, "b": 3}, -7.0)
    run_test("a - a", "a - a", {"a": 999}, 0.0)


def test_multiplication():
    print("\n--- BASIC OPS: MULTIPLICATION ---")

    run_test("a * b", "a * b", {"a": 3, "b": 5}, 15.0)
    run_test("a * b * c", "a * b * c", {"a": 2, "b": 3, "c": 4}, 24.0)
    run_test("a * 0", "a * 0", {"a": 999}, 0.0)
    run_test("a * 1", "a * 1", {"a": 999}, 999.0)
    run_test("a * (-b)", "a * (-b)", {"a": 3, "b": 5}, -15.0)
    run_test("implicit mult", "2a", {"a": 5}, 10.0)
    run_test("implicit mult paren", "3(a + b)", {"a": 2, "b": 4}, 18.0)


def test_division():
    print("\n--- BASIC OPS: DIVISION ---")

    run_test("a / b", "a / b", {"a": 10, "b": 2}, 5.0)
    run_test("a / b / c", "a / b / c", {"a": 120, "b": 2, "c": 3}, 20.0)
    run_test("a / 1", "a / 1", {"a": 999}, 999.0)
    run_test("0 / a", "0 / a", {"a": 999}, 0.0)
    run_test("fraction", "a / b", {"a": 1, "b": 3}, 0.333333, tolerance=1e-4)


def test_power():
    print("\n--- BASIC OPS: POWER ---")

    run_test("a ^ 2", "a ^ 2", {"a": 5}, 25.0)
    run_test("a ^ 3", "a ^ 3", {"a": 3}, 27.0)
    run_test("2 ^ 10", "2 ^ 10", {}, 1024.0)
    run_test("a ^ 0", "a ^ 0", {"a": 999}, 1.0)
    run_test("a ^ 1", "a ^ 1", {"a": 999}, 999.0)
    run_test("right assoc", "2 ^ 3 ^ 2", {}, 512.0)
    run_test("a ^ 0.5", "a ^ 0.5", {"a": 16}, 4.0)


def test_modulo():
    print("\n--- BASIC OPS: MODULO ---")

    run_test("a % b", "a % b", {"a": 17, "b": 5}, 2.0)
    run_test("a % b", "a % b", {"a": 10, "b": 3}, 1.0)
    run_test("a % b", "a % b", {"a": 10, "b": 10}, 0.0)
    run_test("a % b", "a % b", {"a": 7, "b": 3}, 1.0)


def test_mixed_operations():
    print("\n--- BASIC OPS: MIXED ---")

    run_test("a + b * c", "a + b * c", {"a": 2, "b": 3, "c": 4}, 14.0)
    run_test("(a + b) * c", "(a + b) * c", {"a": 2, "b": 3, "c": 4}, 20.0)
    run_test("a * b - c * d", "a * b - c * d", {"a": 3, "b": 4, "c": 2, "d": 5}, 2.0)
    run_test("a + b * c - d / e", "a + b * c - d / e",
             {"a": 10, "b": 3, "c": 5, "d": 100, "e": 4}, 0.0)
    run_test("a ^ 2 + b ^ 2", "a ^ 2 + b ^ 2", {"a": 3, "b": 4}, 25.0)
    run_test("pythagoras", "sqrt(a ^ 2 + b ^ 2)", {"a": 3, "b": 4}, 5.0)
    run_test("quadratic", "(-b + sqrt(b ^ 2 - 4 * a * c)) / (2 * a)",
             {"a": 1, "b": -5, "c": 6}, 3.0)
    run_test("complex chain", "a * b + c / d - e ^ f % g",
             {"a": 2, "b": 3, "c": 20, "d": 4, "e": 2, "f": 3, "g": 5}, 7.0)


if __name__ == "__main__":
    test_addition()
    test_subtraction()
    test_multiplication()
    test_division()
    test_power()
    test_modulo()
    test_mixed_operations()
    print_summary("test_basic_ops.py")
