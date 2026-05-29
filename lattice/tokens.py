TT_PLUS     = 'PLUS'
TT_MINUS    = 'MINUS'
TT_STAR     = 'STAR'
TT_SLASH    = 'SLASH'
TT_CARET    = 'CARET'
TT_MOD      = 'MOD'
TT_FACT     = 'FACT'
TT_COMMA    = 'COMMA'
TT_COLON    = 'COLON'
TT_LPAREN   = 'LPAREN'
TT_RPAREN   = 'RPAREN'
TT_LBRACKET = 'LBRACKET'
TT_RBRACKET = 'RBRACKET'
TT_PIPE     = 'PIPE'
TT_NUMBER   = 'NUMBER'
TT_IDENT    = 'IDENT'
TT_EOF      = 'EOF'
TT_EQ       = 'EQ'
TT_NEQ      = 'NEQ'
TT_LT       = 'LT'
TT_GT       = 'GT'
TT_LTE      = 'LTE'
TT_GTE      = 'GTE'
TT_ASSIGN   = 'ASSIGN'


class Token:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        if self.value is not None:
            return f'{self.type}:{self.value}'
        return f'{self.type}'
