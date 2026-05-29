from .ast_nodes import *
from .tokens import *


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None

    def expect(self, token_type):
        if self.current_token.type == token_type:
            self.advance()
        else:
            raise Exception(
                f"Expected {token_type}, got {self.current_token}"
            )

    def parse(self):
        result = self.parse_comparison()
        if self.current_token.type != TT_EOF:
            raise Exception(
                f"Unexpected token: {self.current_token}"
            )
        return result

    def parse_comparison(self):
        left = self.parse_expression()

        while self.current_token.type in (
            TT_LT, TT_GT, TT_LTE, TT_GTE, TT_EQ, TT_NEQ
        ):
            op = self.current_token.value
            self.advance()
            right = self.parse_expression()
            left = BinaryOp(left, op, right)

        return left

    def parse_expression(self):
        left = self.parse_term()

        while self.current_token.type in (TT_PLUS, TT_MINUS):
            op = self.current_token.value
            self.advance()
            right = self.parse_term()
            left = BinaryOp(left, op, right)

        return left

    def parse_term(self):
        left = self.parse_factor()

        while self.current_token.type in (TT_STAR, TT_SLASH, TT_MOD):
            op = self.current_token.value
            self.advance()
            right = self.parse_factor()
            left = BinaryOp(left, op, right)

        return left

    def parse_factor(self):
        if self.current_token.type == TT_MINUS:
            self.advance()
            operand = self.parse_factor()
            return UnaryOp("-", operand)

        if self.current_token.type == TT_PLUS:
            self.advance()
            operand = self.parse_factor()
            return UnaryOp("+", operand)

        left = self.parse_function()

        while self.current_token.type == TT_FACT:
            self.advance()
            left = FunctionCall("factorial", [left])

        if self.current_token.type == TT_CARET:
            self.advance()
            right = self.parse_factor()
            return BinaryOp(left, "^", right)

        return left

    def parse_function(self):
        if (self.current_token.type == TT_IDENT
                and self.peek() is not None
                and self.peek().type == TT_LPAREN):

            name = self.current_token.value
            self.advance()
            self.advance()

            args = []
            if self.current_token.type != TT_RPAREN:
                args.append(self.parse_expression())
                while self.current_token.type == TT_COMMA:
                    self.advance()
                    args.append(self.parse_expression())

            self.expect(TT_RPAREN)
            return FunctionCall(name, args)

        return self.parse_primary()

    def parse_primary(self):
        token = self.current_token

        if token.type == TT_NUMBER:
            self.advance()
            return Constant(token.value)

        if token.type == TT_IDENT:
            self.advance()
            return Variable(token.value)

        if token.type == TT_LPAREN:
            self.advance()
            node = self.parse_expression()
            self.expect(TT_RPAREN)
            return node

        if token.type == TT_PIPE:
            self.advance()
            node = self.parse_expression()
            self.expect(TT_PIPE)
            return FunctionCall("abs", [node])

        raise Exception(f"Unexpected token: {token}")
