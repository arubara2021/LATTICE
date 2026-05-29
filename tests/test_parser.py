from helpers import run_parser_test, print_summary


def test_numbers():
    print("\n--- PARSER: NUMBERS ---")

    run_parser_test(
        "integer",
        "42",
        "Constant(42)"
    )

    run_parser_test(
        "float",
        "3.14",
        "Constant(3.14)"
    )

    run_parser_test(
        "scientific",
        "1.5e10",
        "Constant(15000000000.0)"
    )


def test_variables():
    print("\n--- PARSER: VARIABLES ---")

    run_parser_test(
        "simple var",
        "x",
        "Variable(x)"
    )

    run_parser_test(
        "underscore var",
        "F_gravity",
        "Variable(F_gravity)"
    )


def test_binary_ops():
    print("\n--- PARSER: BINARY OPS ---")

    run_parser_test(
        "addition",
        "a + b",
        "BinaryOp(Variable(a), +, Variable(b))"
    )

    run_parser_test(
        "subtraction",
        "a - b",
        "BinaryOp(Variable(a), -, Variable(b))"
    )

    run_parser_test(
        "multiplication",
        "a * b",
        "BinaryOp(Variable(a), *, Variable(b))"
    )

    run_parser_test(
        "division",
        "a / b",
        "BinaryOp(Variable(a), /, Variable(b))"
    )

    run_parser_test(
        "power",
        "a ^ b",
        "BinaryOp(Variable(a), ^, Variable(b))"
    )

    run_parser_test(
        "modulo",
        "a % b",
        "BinaryOp(Variable(a), %, Variable(b))"
    )


def test_precedence():
    print("\n--- PARSER: PRECEDENCE ---")

    run_parser_test(
        "mul before add",
        "a + b * c",
        "BinaryOp(Variable(a), +, BinaryOp(Variable(b), *, Variable(c)))"
    )

    run_parser_test(
        "add parens override",
        "(a + b) * c",
        "BinaryOp(BinaryOp(Variable(a), +, Variable(b)), *, Variable(c))"
    )

    run_parser_test(
        "power before mul",
        "a * b ^ c",
        "BinaryOp(Variable(a), *, BinaryOp(Variable(b), ^, Variable(c)))"
    )

    run_parser_test(
        "power right assoc",
        "a ^ b ^ c",
        "BinaryOp(Variable(a), ^, BinaryOp(Variable(b), ^, Variable(c)))"
    )

    run_parser_test(
        "comparison below add",
        "a < b + c",
        "BinaryOp(Variable(a), <, BinaryOp(Variable(b), +, Variable(c)))"
    )

    run_parser_test(
        "deep nesting",
        "a + b * c - d / e",
        "BinaryOp(BinaryOp(Variable(a), +, BinaryOp(Variable(b), *, Variable(c))), -, BinaryOp(Variable(d), /, Variable(e)))"
    )


def test_unary():
    print("\n--- PARSER: UNARY ---")

    run_parser_test(
        "unary minus",
        "-a",
        "UnaryOp(-, Variable(a))"
    )

    run_parser_test(
        "unary plus",
        "+a",
        "UnaryOp(+, Variable(a))"
    )

    run_parser_test(
        "double negation",
        "--a",
        "UnaryOp(-, UnaryOp(-, Variable(a)))"
    )

    run_parser_test(
        "unary before power",
        "-a ^ 2",
        "UnaryOp(-, BinaryOp(Variable(a), ^, Constant(2)))"
    )

    run_parser_test(
        "unary in expression",
        "a * -b",
        "BinaryOp(Variable(a), *, UnaryOp(-, Variable(b)))"
    )

    run_parser_test(
        "unary in parens",
        "(-a + b)",
        "BinaryOp(UnaryOp(-, Variable(a)), +, Variable(b))"
    )


def test_functions():
    print("\n--- PARSER: FUNCTIONS ---")

    run_parser_test(
        "sin",
        "sin(x)",
        "FunctionCall(sin, [Variable(x)])"
    )

    run_parser_test(
        "cos",
        "cos(x)",
        "FunctionCall(cos, [Variable(x)])"
    )

    run_parser_test(
        "two args",
        "pow(a, b)",
        "FunctionCall(pow, [Variable(a), Variable(b)])"
    )

    run_parser_test(
        "function with expression arg",
        "sin(a + b)",
        "FunctionCall(sin, [BinaryOp(Variable(a), +, Variable(b))])"
    )

    run_parser_test(
        "nested functions",
        "sin(cos(x))",
        "FunctionCall(sin, [FunctionCall(cos, [Variable(x)])])"
    )

    run_parser_test(
        "three args",
        "clamp(x, 0, 1)",
        "FunctionCall(clamp, [Variable(x), Constant(0), Constant(1)])"
    )

    run_parser_test(
        "no parens is variable",
        "sin + 5",
        "BinaryOp(Variable(sin), +, Constant(5))"
    )


def test_factorial():
    print("\n--- PARSER: FACTORIAL ---")

    run_parser_test(
        "simple factorial",
        "5!",
        "FunctionCall(factorial, [Constant(5)])"
    )

    run_parser_test(
        "variable factorial",
        "n!",
        "FunctionCall(factorial, [Variable(n)])"
    )

    run_parser_test(
        "double factorial",
        "3!!",
        "FunctionCall(factorial, [FunctionCall(factorial, [Constant(3)])])"
    )

    run_parser_test(
        "factorial before power",
        "3! ^ 2",
        "BinaryOp(FunctionCall(factorial, [Constant(3)]), ^, Constant(2))"
    )


def test_absolute_value():
    print("\n--- PARSER: ABSOLUTE VALUE ---")

    run_parser_test(
        "simple abs",
        "|x|",
        "FunctionCall(abs, [Variable(x)])"
    )

    run_parser_test(
        "abs of expression",
        "|x + 3|",
        "FunctionCall(abs, [BinaryOp(Variable(x), +, Constant(3))])"
    )

    run_parser_test(
        "abs in expression",
        "|a| + |b|",
        "BinaryOp(FunctionCall(abs, [Variable(a)]), +, FunctionCall(abs, [Variable(b)]))"
    )


def test_comparisons():
    print("\n--- PARSER: COMPARISONS ---")

    run_parser_test(
        "less than",
        "a < b",
        "BinaryOp(Variable(a), <, Variable(b))"
    )

    run_parser_test(
        "greater than",
        "a > b",
        "BinaryOp(Variable(a), >, Variable(b))"
    )

    run_parser_test(
        "equal",
        "a == b",
        "BinaryOp(Variable(a), ==, Variable(b))"
    )

    run_parser_test(
        "not equal",
        "a != b",
        "BinaryOp(Variable(a), !=, Variable(b))"
    )

    run_parser_test(
        "less or equal",
        "a <= b",
        "BinaryOp(Variable(a), <=, Variable(b))"
    )

    run_parser_test(
        "greater or equal",
        "a >= b",
        "BinaryOp(Variable(a), >=, Variable(b))"
    )

    run_parser_test(
        "comparison with math",
        "a + 1 < b * 2",
        "BinaryOp(BinaryOp(Variable(a), +, Constant(1)), <, BinaryOp(Variable(b), *, Constant(2)))"
    )


if __name__ == "__main__":
    test_numbers()
    test_variables()
    test_binary_ops()
    test_precedence()
    test_unary()
    test_functions()
    test_factorial()
    test_absolute_value()
    test_comparisons()
    print_summary("test_parser.py")
