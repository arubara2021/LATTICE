from .tokens import *


class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if len(self.text) > 0 else None

    def advance(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def peek(self):
        peek_pos = self.pos + 1
        return self.text[peek_pos] if peek_pos < len(self.text) else None

    def is_ident_start_char(self, c):
        return c.isalpha() or c == '_' or c in '√∂∫'

    def is_ident_char(self, c):
        return (c.isalpha() or c.isdigit() or c == '_'
                or c in '√∂∫₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹')

    def tokenize(self):
        tokens = []
        while self.current_char is not None:
            if self.current_char.isspace():
                self.advance()

            elif self.current_char.isdigit() or (
                    self.current_char == '.'
                    and self.peek() is not None
                    and self.peek().isdigit()):
                tokens.append(self.handle_number())

            elif self.is_ident_start_char(self.current_char):
                tokens.append(self.handle_ident())

            elif self.current_char == '∞':
                tokens.append(Token(TT_NUMBER, float('inf')))
                self.advance()

            else:
                token = self.handle_operator()
                if token is None:
                    raise Exception(f"Invalid character: {self.current_char}")
                tokens.append(token)

            if tokens and self.current_char is not None:
                last_type = tokens[-1].type
                if last_type in (TT_NUMBER, TT_RPAREN):
                    if (self.current_char == '('
                            or self.is_ident_start_char(self.current_char)
                            or self.current_char.isdigit()
                            or self.current_char == '.'):
                        tokens.append(Token(TT_STAR, '*'))

        tokens.append(Token(TT_EOF))
        return tokens

    def handle_number(self):
        number_str = ''
        dot_count = 0

        while (self.current_char is not None
               and (self.current_char.isdigit() or self.current_char == '.')):
            if self.current_char == '.':
                dot_count += 1
                if dot_count > 1:
                    raise Exception("Invalid number format")
            number_str += self.current_char
            self.advance()

        if self.current_char is not None and self.current_char in 'eE':
            save_pos = self.pos
            exp_str = self.current_char
            self.advance()

            if self.current_char is not None and self.current_char in '+-':
                exp_str += self.current_char
                self.advance()

            has_digits = False
            while self.current_char is not None and self.current_char.isdigit():
                has_digits = True
                exp_str += self.current_char
                self.advance()

            if has_digits:
                number_str += exp_str
            else:
                self.pos = save_pos
                self.current_char = self.text[save_pos]

        if dot_count == 0 and 'e' not in number_str.lower():
            return Token(TT_NUMBER, int(number_str))
        return Token(TT_NUMBER, float(number_str))

    def handle_ident(self):
        name = ''
        while self.current_char is not None and self.is_ident_char(self.current_char):
            name += self.current_char
            self.advance()
        return Token(TT_IDENT, name)

    def handle_operator(self):
        char = self.current_char
        self.advance()

        if char == '+': return Token(TT_PLUS, '+')
        if char == '-': return Token(TT_MINUS, '-')
        if char == '/': return Token(TT_SLASH, '/')
        if char == '%': return Token(TT_MOD, '%')
        if char == ',': return Token(TT_COMMA, ',')
        if char == ':': return Token(TT_COLON, ':')
        if char == '(': return Token(TT_LPAREN, '(')
        if char == ')': return Token(TT_RPAREN, ')')
        if char == '[': return Token(TT_LBRACKET, '[')
        if char == ']': return Token(TT_RBRACKET, ']')
        if char == '^': return Token(TT_CARET, '^')
        if char == '|': return Token(TT_PIPE, '|')

        if char == '*':
            if self.current_char == '*':
                self.advance()
                return Token(TT_CARET, '^')
            return Token(TT_STAR, '*')

        if char == '=':
            if self.current_char == '=':
                self.advance()
                return Token(TT_EQ, '==')
            return Token(TT_ASSIGN, '=')

        if char == '!':
            if self.current_char == '=':
                self.advance()
                return Token(TT_NEQ, '!=')
            return Token(TT_FACT, '!')

        if char == '<':
            if self.current_char == '=':
                self.advance()
                return Token(TT_LTE, '<=')
            return Token(TT_LT, '<')

        if char == '>':
            if self.current_char == '=':
                self.advance()
                return Token(TT_GTE, '>=')
            return Token(TT_GT, '>')

        return None
