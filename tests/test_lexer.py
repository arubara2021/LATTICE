from helpers import run_lexer_test, print_summary


def test_basic_tokens():
    print("\n--- LEXER: BASIC TOKENS ---")

    run_lexer_test(
        "single number",
        "42",
        ['NUMBER:42', 'EOF']
    )

    run_lexer_test(
        "decimal number",
        "3.14",
        ['NUMBER:3.14', 'EOF']
    )

    run_lexer_test(
        "leading dot",
        ".5",
        ['NUMBER:0.5', 'EOF']
    )

    run_lexer_test(
        "scientific notation",
        "1.5e10",
        ['NUMBER:15000000000.0', 'EOF']
    )

    run_lexer_test(
        "negative scientific",
        "1.6e-19",
        ['NUMBER:1.6e-19', 'EOF']
    )

    run_lexer_test(
        "positive scientific",
        "2.5E+3",
        ['NUMBER:2500.0', 'EOF']
    )


def test_operators():
    print("\n--- LEXER: OPERATORS ---")

    run_lexer_test(
        "all arithmetic",
        "+ - * / ^ % !",
        ['PLUS:+', 'MINUS:-', 'STAR:*', 'SLASH:/', 'CARET:^', 'MOD:%', 'FACT:!', 'EOF']
    )

    run_lexer_test(
        "all comparisons",
        "< > <= >= == !=",
        ['LT:<', 'GT:>', 'LTE:<=', 'GTE:>=', 'EQ:==', 'NEQ:!=', 'EOF']
    )

    run_lexer_test(
        "double star power",
        "a ** b",
        ['IDENT:a', 'CARET:^', 'IDENT:b', 'EOF']
    )

    run_lexer_test(
        "brackets and pipe",
        "[ a | b ]",
        ['LBRACKET:[', 'IDENT:a', 'PIPE:|', 'IDENT:b', 'RBRACKET:]', 'EOF']
    )

    run_lexer_test(
        "comma and colon",
        "a, b: c",
        ['IDENT:a', 'COMMA:,', 'IDENT:b', 'COLON::', 'IDENT:c', 'EOF']
    )

    run_lexer_test(
        "assign",
        "a = b",
        ['IDENT:a', 'ASSIGN:=', 'IDENT:b', 'EOF']
    )


def test_identifiers():
    print("\n--- LEXER: IDENTIFIERS ---")

    run_lexer_test(
        "simple variable",
        "x",
        ['IDENT:x', 'EOF']
    )

    run_lexer_test(
        "multi char name",
        "theta",
        ['IDENT:theta', 'EOF']
    )

    run_lexer_test(
        "underscore name",
        "F_gravity",
        ['IDENT:F_gravity', 'EOF']
    )

    run_lexer_test(
        "name with digit",
        "x1",
        ['IDENT:x1', 'EOF']
    )

    run_lexer_test(
        "name with digit and underscore",
        "v_initial_2",
        ['IDENT:v_initial_2', 'EOF']
    )

    run_lexer_test(
        "function name",
        "sin",
        ['IDENT:sin', 'EOF']
    )


def test_implicit_multiplication():
    print("\n--- LEXER: IMPLICIT MULTIPLICATION ---")

    run_lexer_test(
        "number before paren",
        "2(3)",
        ['NUMBER:2', 'STAR:*', 'LPAREN:(', 'NUMBER:3', 'RPAREN:)', 'EOF']
    )

    run_lexer_test(
        "number before ident",
        "5x",
        ['NUMBER:5', 'STAR:*', 'IDENT:x', 'EOF']
    )

    run_lexer_test(
        "paren before paren",
        "(2)(3)",
        ['LPAREN:(', 'NUMBER:2', 'RPAREN:)', 'STAR:*', 'LPAREN:(', 'NUMBER:3', 'RPAREN:)', 'EOF']
    )

    run_lexer_test(
        "paren before number",
        "(2)3",
        ['LPAREN:(', 'NUMBER:2', 'RPAREN:)', 'STAR:*', 'NUMBER:3', 'EOF']
    )

    run_lexer_test(
        "paren before ident",
        "(2)x",
        ['LPAREN:(', 'NUMBER:2', 'RPAREN:)', 'STAR:*', 'IDENT:x', 'EOF']
    )

    run_lexer_test(
        "number before function",
        "5sin(x)",
        ['NUMBER:5', 'STAR:*', 'IDENT:sin', 'LPAREN:(', 'IDENT:x', 'RPAREN:)', 'EOF']
    )


def test_complex_expressions():
    print("\n--- LEXER: COMPLEX EXPRESSIONS ---")

    run_lexer_test(
        "full expression",
        "sin(a) + b * c ^ 2",
        ['IDENT:sin', 'LPAREN:(', 'IDENT:a', 'RPAREN:)',
         'PLUS:+', 'IDENT:b', 'STAR:*', 'IDENT:c', 'CARET:^', 'NUMBER:2', 'EOF']
    )

    run_lexer_test(
        "comparison with math",
        "a + 1 < b * 2",
        ['IDENT:a', 'PLUS:+', 'NUMBER:1', 'LT:<',
         'IDENT:b', 'STAR:*', 'NUMBER:2', 'EOF']
    )

    run_lexer_test(
        "absolute value",
        "|x + 3|",
        ['PIPE:|', 'IDENT:x', 'PLUS:+', 'NUMBER:3', 'PIPE:|', 'EOF']
    )

    run_lexer_test(
        "factorial",
        "5!",
        ['NUMBER:5', 'FACT:!', 'EOF']
    )

    run_lexer_test(
        "spaces ignored",
        "  a  +  b  ",
        ['IDENT:a', 'PLUS:+', 'IDENT:b', 'EOF']
    )


if __name__ == "__main__":
    test_basic_tokens()
    test_operators()
    test_identifiers()
    test_implicit_multiplication()
    test_complex_expressions()
    print_summary("test_lexer.py")
