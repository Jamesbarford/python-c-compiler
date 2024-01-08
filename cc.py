#!/usr/bin/env python3
from typing import cast, NoReturn
import sys

def panic(argv) -> NoReturn:
    print(f"ERROR: {argv}")
    exit(1)

OP_PLUS = ord("+")
OP_SUB = ord("-")
OP_MUL = ord("*")
OP_DIV = ord("/")
OP_SHL = 0x20
OP_SHR = 0x21

op_to_alu = { OP_SHR: "SHR",OP_SHL: "SHL",OP_MUL: "MUL", OP_PLUS: "ADD", OP_SUB: "SUB", OP_DIV: "DIV" }
op_to_str = { OP_SHR: ">>",OP_SHL: "<<",OP_MUL: "*", OP_PLUS: "+", OP_SUB: "-", OP_DIV: "/" }
str_to_op = {">>": OP_SHR, "<<": OP_SHL, "+": OP_PLUS, "-": OP_SUB, "*": OP_MUL, "/": OP_DIV}

TK_IDENT = 0
TK_PUNCT = 1
TK_KW = 2
TK_IDENT = 3
TK_I64 = 4
TK_F64 = 5
keywords = {"void","int","long","while","break","return","continue","if","else"}
mult_tk = {"<<",">>","++","--","->","=="}

class Token:
    def __init__(self, kind: int, lineno: int) -> None:
        self.kind = kind
        self.lineno = lineno

class TokenIdent(Token):
    def __init__(self, ident: str, lineno: int) -> None:
        super().__init__(TK_IDENT,lineno)
        self.ident = ident
    def __str__(self) -> str: return self.ident

class TokenPunct(Token):
    def __init__(self, punct: str, lineno: int) -> None:
        super().__init__(TK_PUNCT,lineno)
        self.punct = punct
    def __str__(self) -> str: return self.punct

class TokenI64(Token):
    def __init__(self, i64: int, lineno: int) -> None:
        super().__init__(TK_I64, lineno)
        self.i64 = i64
    def __str__(self) -> str: return str(self.i64)

class TokenF64(Token):
    def __init__(self, f64: float, lineno: int) -> None:
        super().__init__(TK_F64, lineno)
        self.f64 = f64
    def __str__(self) -> str: return str(self.f64)

class TokenKeyWord(Token):
    def __init__(self, ident: str, lineno: int) -> None:
        super().__init__(TK_KW, lineno)
        self.ident = ident
    def __str__(self) -> str: return self.ident

class Lexer:
    def __init__(self, code: str):
        self.code = code
        self.idx = 0
        self.code_len = len(code)
        self.lineno = 1

    def get_next(self) -> str:
        if self.idx == self.code_len:
            return '\0'
        ch = self.code[self.idx]
        if ch == '\n': self.lineno += 1
        self.idx+=1
        return ch
    def peek(self) -> str: return self.code[self.idx]

    def rewind(self): self.idx -= 1

def lexident(lexer: Lexer, ch) -> str:
    ident = ch
    while ch := lexer.get_next():
        if not str.isalpha(ch) and not str.isdigit(ch) and ch != '\0':
            break
        ident += ch
    lexer.rewind()
    return ident

def lexnum(lexer: Lexer, ch) -> tuple[bool, int | float]:
    strnum = ch
    while str.isdigit((ch := lexer.get_next())):
        strnum += ch
    if ch == '.':
        strnum += '.'
        while str.isdigit((ch := lexer.get_next())):
            strnum += ch
        lexer.rewind()
        return True, float(strnum)
    lexer.rewind()
    return False, int(strnum)

def lexc(code: str) -> list[Token]:
    tokens = []
    lexer = Lexer(code)
    while ch := lexer.get_next():
        if ch == '\0': break
        elif str.isalpha(ch):
            ident = lexident(lexer,ch)
            if ident in keywords: tokens.append(TokenKeyWord(ident,lexer.lineno))
            else: tokens.append(TokenIdent(ident,lexer.lineno))
        elif str.isdigit(ch):
            is_float,num = lexnum(lexer,ch)
            if is_float: tokens.append(TokenF64(float(num),lexer.lineno))
            else: tokens.append(TokenI64(int(num),lexer.lineno))
        elif ch in {"+","-","*","/",";","{","}","(",")"}: tokens.append(TokenPunct(ch,lexer.lineno))
        elif ch == '>':
            if lexer.peek() == '>':
                lexer.get_next()
                tokens.append(TokenPunct(">>",lexer.lineno))
            else:
                tokens.append(TokenPunct(ch,lexer.lineno))
        elif ch == '<':
            if lexer.peek() == '<':
                lexer.get_next()
                tokens.append(TokenPunct("<<",lexer.lineno))
            else:
                tokens.append(TokenPunct(ch,lexer.lineno))
        elif ch == '=':
            if lexer.peek() == '=':
                lexer.get_next()
                tokens.append(TokenPunct("==",lexer.lineno))
            else:
                tokens.append(TokenPunct(ch,lexer.lineno))
    return tokens

# AST ===========
# This limited implementation can do arithmetic operations given an Ast
AST_INT = 0
AST_FLOAT = 1
AST_COMPOUND = 2
AST_LITERAL = 3
AST_LVAR = 4
AST_DECL = 5
AST_FUN = 5
AST_FUN_CALL = 6
AST_PTR = 7
AST_RETURN = 8
AST_BREAK = 9
AST_CONTINUE = 10
AST_WHILE = 11
AST_IF = 12

class AstType:
    def __init__(self, size: int, kind: int = 0, ptr = None) -> None:
        self.kind: int = kind
        self.issigned: bool = False
        self.size: int = size
        self.ptr: AstType | None = ptr
    def __str__(self) -> str:
        if self.kind == AST_INT: return "int"
        elif self.kind == AST_FLOAT: return "float"
        elif self.kind == AST_COMPOUND: return "compound"
        elif self.kind == AST_LITERAL: return "literal"
        elif self.kind == AST_LVAR: return "lvar"
        elif self.kind == AST_DECL: return "decl"
        elif self.kind == AST_FUN: return "function"
        elif self.kind == AST_FUN_CALL: return "function_call"
        elif self.kind == AST_PTR: return "pointer"
        elif self.kind == AST_RETURN: return "return"
        elif self.kind == AST_BREAK: return "break"
        elif self.kind == AST_CONTINUE: return "continue"
        elif self.kind == AST_WHILE: return "while"
        elif self.kind == AST_IF: return "if"
        else: return "unknown"

ast_type_i32 = AstType(size=4, kind=AST_INT)
ast_type_i64 = AstType(size=8, kind=AST_INT)
ast_type_f64 = AstType(size=8, kind=AST_FLOAT)

class AstTypePtr(AstType):
    def __init__(self, base: AstType) -> None:
        super().__init__(8,AST_PTR,base)

class Ast:
    kind: int
    def __init__(self, ast_type: AstType | None = None) -> None:
        self.type = ast_type
        self.offset = 0

class AstI32(Ast):
    def __init__(self, i32: int) -> None:
        super().__init__(ast_type_i32)
        self.kind = AST_LITERAL
        self.i64 = i32
    def __str__(self) -> str: return str(self.i64)

class AstI64(Ast):
    def __init__(self, i64: int) -> None:
        super().__init__(ast_type_i64)
        self.kind = AST_LITERAL
        self.i64 = i64
    def __str__(self) -> str: return str(self.i64)

class AstF64(Ast):
    def __init__(self, f64: float) -> None:
        super().__init__(ast_type_f64)
        self.kind = AST_LITERAL
        self.f64 = f64
    def __str__(self) -> str: return str(self.f64)

class AstBinaryOp(Ast):
    def __init__(self, ast_type: AstType, left: Ast, op: int, right: Ast) -> None:
        super().__init__(ast_type)
        self.left = left
        self.right = right
        self.kind = op
    def __str__(self) -> str: return f"<binop>\n\t\t({self.left} {op_to_str[self.kind]} {self.right})"

class AstCompound(Ast):
    # list of Ast's
    def __init__(self, argv: list[Ast]) -> None:
        super().__init__(None)
        self.kind = AST_COMPOUND
        self.stmts = argv
    def __str__(self) -> str: return "<compound>\n" + "\n".join(f"\t{node}" for node in cast(list[Ast], self.stmts))

class AstFunction(Ast):
    def __init__(self, ret_ast_type: AstType, fname: str, params: list[Ast], body: AstCompound, local_defs: list[Ast]) -> None:
        super().__init__(ret_ast_type)
        self.kind = AST_FUN
        self.fname = fname
        self.params = params
        self.body = body
        self.locals = local_defs

    def __str__(self) -> str:
        params = ", ".join(f"{node.type} {node}" for node in cast(list[Ast], self.params))
        body = str(self.body) #  "\n".join(f"\t{node}" for node in cast(list[Ast], self.body.stmts))
        return f"<fun> {self.type} {self.fname}\n<params>({params})\n{body}"

class AstFunctionCall(Ast):
    def __init__(self, ast_type: AstType, argv: list[Ast], fname: str)-> None:
        super().__init__(ast_type)
        self.kind = AST_FUN_CALL
        self.argv = argv
        self.fname = fname

class AstDecl(Ast):
    def __init__(self, var: Ast, init: Ast | None)-> None:
        super().__init__(None)
        self.kind = AST_DECL
        self.var = var
        self.init = init
    def __str__(self) -> str: return f"<decl> {self.var.type} \n\t\t{self.var} = {self.init}"

class AstLVar(Ast):
    def __init__(self, ast_type: AstType, name: str) -> None:
        super().__init__(ast_type)
        self.kind = AST_LVAR
        self.name = name
    def __str__(self) -> str: return f"<lvar> {self.name}"

class AstReturn(Ast):
    def __init__(self, ast_type: AstType, retval: Ast | None) -> None:
        super().__init__(ast_type)
        self.kind = AST_RETURN
        self.retval = retval
    def __str__(self) -> str: return f"<return> {self.retval}"

class AstWhile(Ast):
    def __init__(self, cond: Ast|None, body: Ast, begin_label: str, end_label: str) -> None:
        super().__init__(None)
        self.kind = AST_WHILE
        self.cond = cond
        self.body = body
        self.begin_label = begin_label
        self.end_label = end_label

class AstBreak(Ast):
    def __init__(self,label: str) -> None:
        super().__init__(None)
        self.label = label
        self.kind = AST_BREAK

class AstContinue(Ast):
    def __init__(self,label: str) -> None:
        super().__init__(None)
        self.label = label
        self.kind = AST_CONTINUE

class AstIf(Ast):
    def __init__(self, cond: Ast,then: Ast,els: Ast | None) -> None:
        super().__init__(None)
        self.kind = AST_IF
        self.cond = cond
        self.then = then
        self.els = els

# I've just made this up
def get_priority(tok: TokenPunct) -> int:
    if tok.punct in {'[','.','->'}: return 1
    elif tok.punct == '/': return 2
    elif tok.punct == '*': return 3
    elif tok.punct == '+': return 4
    elif tok.punct == '-': return 5
    elif tok.punct in {'&','|','>>','<<'}: return 6
    elif tok.punct == '==': return 7
    else: return -1

label_count = 1
def create_label():
    global label_count
    label_count += 1
    return f".L{label_count}"

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.tokens_len = len(tokens)
        self.ptr = 0
        self.env: dict = {}
        self.types = {"int": ast_type_i64, "long": ast_type_i32, "float": ast_type_f64}
        self.tmp_env: dict = {}
        self.tmp_func: AstFunction
        self.tmp_locals: list[Ast]
        self.tmp_ret_type: AstType
        self.tmp_loop_end: str | None
        self.tmp_loop_begin: str | None

    def get_type(self,name:str) -> AstType:
        if (val := self.types.get(name)) is not None: return val
        panic(f"Invalid type name: {name}")

    def is_type(self,tok:Token) -> bool: return (isinstance(tok,TokenIdent) or isinstance(tok,TokenKeyWord)) and tok.ident in self.types

    def is_punct_match(self,tok: Token | None, punct: str) -> bool: return isinstance(tok,TokenPunct) and tok.punct == punct

    def rewind(self) -> None: self.ptr -= 1

    def peek(self) -> Token | None: return None if self.ptr == self.tokens_len else self.tokens[self.ptr]

    def get_next(self) -> Token | None:
        if self.ptr == self.tokens_len:
            return None
        tok = self.tokens[self.ptr]
        self.ptr += 1
        return tok

    def expect_tok_next(self, expected: str) -> bool:
        tok = self.get_next()
        if isinstance(tok, TokenPunct) and tok.punct == expected: return True
        if tok: panic(f"Parser error: Missmatched characters: {tok} != {expected} at line: {tok.lineno}")
        else: panic(f"Parser error: expected {expected} ran out of input")

    def env_get(self, name: str) -> Ast | None:
        cur_env = self.tmp_env
        while cur_env:
            if isinstance((ast := cur_env.get(name)), Ast): return ast
            cur_env = cur_env.get("parent")
        return None

    def func_get(self,name:str) -> AstFunction | None:
        func = self.env_get(name)
        return func if isinstance(func,AstFunction) else None

    def parse_function_arguments(self, name: str) -> Ast:
        func: AstFunction | None = self.func_get(name)
        argv = []
        tok = self.peek()
        while tok and not self.is_punct_match(tok,')'):
            ast = self.parse_expr()
            tok = self.get_next()
            argv.append(ast)
            if self.is_punct_match(tok,')'): break
            elif not self.is_punct_match(tok,','): panic(f"Expected ',' got: {tok}")
            tok = self.peek()
        if len(argv) == 0: self.get_next() # move passed '(' as we've not parsed anything :(
        if func: return AstFunctionCall(cast(AstType,func.type),argv,name)
        return AstFunctionCall(ast_type_i64,argv,name)

    def parse_function_call_or_identifier(self, name: TokenIdent) -> None | Ast:
        tok = self.get_next()
        if self.is_punct_match(tok,'('):
            return self.parse_function_arguments(name.ident)
        self.rewind()
        if (ast := self.env_get(name.ident)) is None: panic(f"Identifier: {name.ident} is undefined at line: {name.lineno}")
        return ast

    def parse_primary(self) -> Ast | None:
        tok = self.get_next()
        if tok is None: panic("Ran out of input while parsing primary expression")
        if isinstance(tok,TokenIdent): return self.parse_function_call_or_identifier(tok)
        elif isinstance(tok,TokenI64): return AstI64(tok.i64)
        elif isinstance(tok,TokenF64): return AstF64(tok.f64)
        elif isinstance(tok,TokenPunct):
            self.rewind()
            return None

    def parse_expr(self, prec: int = 16) -> Ast | None:
        lhs = self.parse_primary()
        if lhs is None: return None
        while 1:
            if (tok := self.get_next()) is None: return lhs
            if not isinstance(tok,TokenPunct):
                self.rewind()
                return lhs
            prec2 = get_priority(tok)

            if prec2 < 0 or prec <= prec2:
                self.rewind()
                return lhs

            if self.is_punct_match(tok,'='):
                if not lhs.kind in {AST_LVAR}: panic(f"{lhs} is not an lvalue")

            next_prec = prec2
            if self.is_punct_match(tok,'='):
                next_prec += 1
            rhs = self.parse_expr(next_prec)
            if rhs is None: panic(f"lefthand lvar missing right hand value at line: {tok.lineno}")

            lhs = AstBinaryOp(ast_type_i64,lhs,str_to_op[tok.punct],rhs)

    def parse_declaration_initialiser(self, var: Ast, terminators: set[str]) -> None | Ast:
        tok = self.get_next()
        if self.is_punct_match(tok,'='):
            init = self.parse_expr()
            tok = self.get_next()
            assert isinstance(tok,TokenPunct) and tok.punct in terminators
            return AstDecl(var,init)
        self.rewind()
        tok = self.get_next()
        if isinstance(tok,TokenPunct) and tok.punct in terminators:
            return AstDecl(var,None)
        panic(f"Invalid variable initaliser: {tok}")

    def parse_statement(self):
        tok = self.get_next()
        if isinstance(tok,TokenKeyWord):
            if tok.ident == "if":
                self.expect_tok_next('(')
                cond = self.parse_expr()
                if cond is None: panic("if <cond> cannot be None")
                self.expect_tok_next(')')
                then = self.parse_statement()
                tok = self.get_next()
                if isinstance(tok,TokenKeyWord) and tok.ident == "else":
                    els = self.parse_statement()
                    return AstIf(cond,then,els)
                self.rewind()
                return AstIf(cond,then,None)
            elif tok.ident == "return":
                print("here")
                retval = self.parse_expr()
                self.expect_tok_next(';')
                return AstReturn(self.tmp_ret_type,retval)
            elif tok.ident == "while":
                while_begin = create_label()
                while_end = create_label()
                self.tmp_loop_end = while_end
                self.tmp_loop_begin = while_begin
                self.tmp_env = {"parent": self.tmp_env}
                while_cond = self.parse_expr(16)
                self.expect_tok_next(')')
                while_body = self.parse_statement()
                self.tmp_env = self.tmp_env["parent"]
                self.tmp_loop_begin = None
                self.tmp_loop_end = None
                return AstWhile(while_cond,while_body,while_begin,while_end)
            elif tok.ident == "break":
                if self.tmp_loop_end is None: panic(f"Floating 'break' statement at line: {tok.lineno}")
                return AstBreak(self.tmp_loop_end)
            elif tok.ident == "continue":
                if self.tmp_loop_begin is None: panic(f"Floating 'continue' statement at line: {tok.lineno}")
                return AstContinue(self.tmp_loop_begin)

    def parse_compound(self) -> Ast:
        statements = []
        self.tmp_env = {"parent": self.env}
        tok = self.peek()
        while tok and not self.is_punct_match(tok,'}'):
            if self.is_type(tok):
                base_type = self.parse_base_type()
                while True:
                    next_type = self.parse_ptr(base_type)
                    varname = self.get_next()
                    if varname is None or not isinstance(varname,TokenIdent): break
                    var = AstLVar(next_type,varname.ident)
                    self.tmp_env[var.name] = var
                    self.tmp_locals.append(var)
                    statement = self.parse_declaration_initialiser(var,{',',';'})
                    if statement is not None: statements.append(statement)
                    self.rewind()
                    tok = self.get_next()
                    if self.is_punct_match(tok,';'): break
                    elif self.is_punct_match(tok,','): continue
                    else: panic(f"Unexpected token: {tok}")
            else:
                stmt = self.parse_statement()
                if stmt: statements.append(stmt)
                else: break
            tok = self.peek()
        self.tmp_env = self.tmp_env["parent"]
        self.expect_tok_next('}')
        return AstCompound(statements)

    def parse_base_type(self) -> AstType:
        tok = self.get_next()
        if tok is None: panic("Ran out of tokens while parsing base_type")
        if isinstance(tok,TokenIdent) or isinstance(tok,TokenKeyWord):
            return self.get_type(tok.ident)
        panic(f"undefined type {tok}")

    def parse_ptr(self, base_type: AstType) -> AstType:
        ptr_type = base_type
        while True:
            tok = self.get_next()
            if not self.is_punct_match(tok,'*'):
                self.rewind()
                return ptr_type
            ptr_type = AstTypePtr(ptr_type)

    def parse_type(self) -> AstType:
        base_type = self.parse_base_type()
        return self.parse_ptr(base_type)

    def parse_params(self) -> list[Ast]:
        params = []
        self.expect_tok_next('(')
        while tok := self.peek():
            if isinstance(tok, TokenPunct) and tok.punct == ')':
                self.get_next()
                break
            param_type = self.parse_type()
            name = self.get_next()
            if name is None: panic(f"Expected a named variable while parsing function parameters of {self.tmp_func.fname}")
            elif not isinstance(name,TokenIdent): panic("Expected Identifier got {name} at line {name.lineno}")
            params.append(AstLVar(param_type,name.ident))
        return params

    def parse_function(self, ret_type: AstType, tok_ident: TokenIdent) -> Ast:
        self.tmp_env = {
            "parent": self.env
        }
        self.tmp_locals = []
        self.tmp_ret_type = ret_type
        tmp_compound = AstCompound([])
        params = self.parse_params()
        self.expect_tok_next('{')
        func = AstFunction(ret_type, tok_ident.ident, params, tmp_compound, [])
        body = self.parse_compound()
        func.body = cast(AstCompound, body)
        func.locals = self.tmp_locals
        self.tmp_locals = []
        self.tmp_func = func
        return func

    def parse_top(self) -> Ast | None:
        tok = self.peek()
        if tok is None: return None
        if isinstance(tok,TokenKeyWord) or isinstance(tok,TokenIdent):
            ret_type = self.parse_type()
            name = self.peek()
            if isinstance(name,TokenIdent):
                self.get_next()
                ast = self.parse_function(ret_type,name)
                return ast
        else:
            panic(f"Error expected function definition at top level def got {tok} at line {tok.lineno}")

    def parse(self) -> list[Ast]: return list(iter(self.parse_top, None))

## TAC =======================
# Three Adress Code IR START
TAC_NULL = -1
TAC_REG = 0
TAC_ALU = 1
TAC_INT = 2
TAC_FLOAT = 3
TAC_BINOP = 4
TAC_LIST = 5
TAC_FUNC = 6
TAC_SAVE = 7
TAC_LOAD = 8
TAC_RETURN = 9

class TACNode:
    def __init__(self, kind: int) -> None:
        self.kind = kind

class TAClist(TACNode):
    def __init__(self, tac_list: list[TACNode]) -> None:
        super().__init__(TAC_LIST)
        self.tac_list = tac_list
    def __str__(self) -> str:
        buf = ""
        for tac in self.tac_list:
            buf += f"\t{tac}\n"
        return buf

class TACNull(TACNode):
    def __init__(self) -> None:
        super().__init__(TAC_NULL)
    def __str__(self) -> str: return "NULL"

class TACInt(TACNode):
    def __init__(self, num: int, size: int) -> None:
        super().__init__(TAC_INT)
        self.num = num
        self.size = size
    def __str__(self) -> str: return str(f"@i{self.size*8}::{self.num}")

class TACFloat(TACNode):
    def __init__(self, num: float, size: int) -> None:
        super().__init__(TAC_FLOAT)
        self.num = num
        self.size = size
    def __str__(self) -> str: return str(f"@f{self.size*8}::{self.num}")

class TACReg(TACNode):
    def __init__(self, reg: int) -> None:
        super().__init__(TAC_REG)
        self.reg = reg
    def __str__(self) -> str: return str(f"R{self.reg}")

class TACAlu(TACNode):
    def __init__(self, alu: int) -> None:
        super().__init__(TAC_ALU)
        self.alu = alu
    def __str__(self) -> str: return f"{op_to_alu[self.alu]}"

class TACBinOp(TACNode):
    def __init__(self, alu: TACNode, op1: TACNode, op2: TACNode, result: TACNode) -> None:
        super().__init__(TAC_BINOP)
        self.alu = alu
        self.op1 = op1
        self.op2 = op2
        self.result = result
    def __str__(self) -> str:
        # weird; kinda TAC, kinda ASM o7
        return f"{self.result} = {self.alu} {self.op1}, {self.op2}"

class TACFunc(TACNode):
    def __init__(self, name: str, body: TACNode, local_vars: list[Ast], params: list[Ast]):
        super().__init__(TAC_FUNC)
        self.name = name
        self.locals = local_vars
        self.body = body
        self.params = params
    def __str__(self) -> str: return f"{self.name}::\n{self.body}"

class TACSave(TACNode):
    def __init__(self, variable: Ast, init: TACNode | None, reg: TACNode) -> None:
        super().__init__(TAC_SAVE)
        self.var = variable
        self.init = init
        self.reg = reg
        self.offset = variable.offset
    def __str__(self) -> str: return f"{self.reg} = {self.init}"

class TACLoad(TACNode):
    def __init__(self, variable: Ast, reg: TACNode) -> None:
        super().__init__(TAC_LOAD)
        self.var = variable
        self.reg = reg
    def __str__(self) -> str: return f"{self.reg}"

class TACReturn(TACNode):
    def __init__(self, reg: TACReg, expr: Ast|None) -> None:
        super().__init__(TAC_RETURN)
        self.reg = reg
        self.expr = expr
    def __str__(self) -> str: return f"RET {self.reg}"

class Register:
    __reg: int = 0

    @staticmethod
    def get_next() -> TACNode:
        reg = Register.__reg
        Register.__reg += 1
        return TACReg(reg)

    @staticmethod
    def reset() -> None: Register.__reg = 0

class IR:
    def __init__(self) -> None:
        self.var_to_reg = {}
        self.ops: list[TACNode] = []

def ir_literal(ast: AstF64|AstI32|AstI64) -> TACNode:
    if ast.type:
        if isinstance(ast,AstI32) or isinstance(ast,AstI64): return TACInt(ast.i64,ast.type.size)
        elif isinstance(ast,AstF64): return TACFloat(ast.f64,ast.type.size)
        else: panic(f"unknown kind: {ast.type.kind}")
    else: panic(f"kind: {ast.type} is NULL")

def ir_compound(ir: IR, stmts: list[Ast] = []) -> TACNode:
    return TAClist([ir_expr(ir,stmt) for stmt in stmts])

def ir_save(ir: IR, ast: AstDecl) -> TACNode:
    if ast.init:
        init = ir_expr(ir,ast.init)
        save = TACSave(ast.var,init,Register.get_next())
    else:
        save = TACSave(ast.var,None,Register.get_next())
    lvar = cast(AstLVar,ast.var)
    ir.var_to_reg[lvar.name] = save.reg
    ir.ops.append(save)
    return save

def ir_load(ir:IR, ast: AstLVar) -> TACNode: return TACLoad(ast,ir.var_to_reg[ast.name])

def ir_return(ir: IR, ast: AstReturn) -> TACNode:
    expr = ir_expr(ir,ast.retval)
    reg = cast(TACReg, expr)
    if ast.retval: ret = TACReturn(reg,ast.retval)
    else: ret = TACReturn(reg,None)
    ir.ops.append(ret)
    return ret

def ir_expr(ir: IR, ast: Ast | None) -> TACNode:
    if ast is None: return TACNull()
    if isinstance(ast,AstI32) or \
            isinstance (ast,AstI64) or \
            isinstance(ast,AstF64):   return ir_literal(ast)
    elif isinstance(ast,AstCompound): return ir_compound(ir,ast.stmts)
    elif isinstance(ast,AstDecl): return ir_save(ir,ast) # This will be a local
    elif isinstance(ast,AstLVar):
            load = ir_load(ir,ast)
            return load
    elif isinstance(ast,AstReturn): return ir_return(ir,ast)
    elif isinstance(ast,AstFunction):
        body = ir_compound(ir,ast.body.stmts)
        return TACFunc(ast.fname,body,ast.locals,ast.params)
    elif ast.kind in op_to_alu:
        quad = cast(TACBinOp, ir_binop(ir,ast))
        ir.ops.append(quad)
        return quad.result
    else: panic(f"kind: {ast} not handled")

def ir_binop(ir: IR, ast: Ast) -> TACNode:
    binop = cast(AstBinaryOp,ast)
    return TACBinOp(
        TACAlu(ast.kind),
        ir_expr(ir,binop.left),
        ir_expr(ir,binop.right),
        Register.get_next())

def ir_gen(funcs: list[Ast]) -> list[TACNode]:
    ir_funcs = []
    for func in funcs:
        if not isinstance(func,AstFunction): panic("Can only generate ir from function definitions")
        ir = IR()
        ir_fn = ir_expr(ir,func)
        cast(TACFunc,ir_fn).body = TAClist(ir.ops)
        #print(ir.ops)
        ir_funcs.append(ir_fn)
    return ir_funcs

## CODE GEN
x86_registers = ["RDI","RSI","RDX","RCX","R8","R9","R10","R11","R12","R13","R14","R15"]
def x86(ops: list[TACNode], stack_space: int) -> str:
    x86_code = []
    for op in ops:
        if isinstance(op,TACReturn):
            if stack_space > 0:
                x86_code.append(f"ADD\tRSP, {stack_space}\n\t")
            if op.expr:
                x86_code.append(f"MOVQ\tRAX, {op.expr.offset}[RBP]\n\t")
            x86_code.append(f"LEAVE\n\tRET\n\n")
        if isinstance(op,TACSave):
            if isinstance(op.init,TACInt):
                x86_code.append(f"MOVQ\t{op.var.offset}[RBP], {op.init.num}\n\t")
            else:
                x86_code.append(f"MOVQ\t{op.var.offset}[RBP], RAX\n\t")
        if isinstance(op,TACBinOp):
            binop = cast(TACBinOp,op)
            alu = cast(TACAlu, binop.alu)
            mnemonic = op_to_alu[alu.alu]
            # XXX: optimiser would nuke this as it is a constant expression
            if isinstance(binop.op1,TACInt) and isinstance(binop.op2,TACInt):
                x86_code.append(f"MOVQ\tRAX, {binop.op1.num}\n\t")
                x86_code.append(f"MOVQ\tRCX, {binop.op2.num}\n\t")
                x86_code.append(f"{mnemonic}\tRAX, RCX\n\t")
            elif isinstance(binop.op1,TACLoad) and isinstance(binop.op2,TACInt):
                if alu.alu == OP_SHL or alu.alu == OP_SHR:
                    x86_code.append(f"{mnemonic}\tAL, {binop.op2.num}\n\t")
                else:
                    x86_code.append(f"MOVQ\tRCX, {binop.op2.num}\n\t")
                    x86_code.append(f"MOVQ\tRAX, {binop.op1.var.offset}[RBP]\n\t")
                    x86_code.append(f"{mnemonic}\tRAX, RCX\n\t")
                    x86_code.append(f"MOVQ\t{binop.op1.var.offset}[RBP], RAX\n\t")
            elif isinstance(binop.op1,TACLoad) and isinstance(binop.op2, TACLoad):
                x86_code.append(f"MOVQ\tRAX, {binop.op1.var.offset}[RBP]\n\t")
                x86_code.append(f"MOVQ\tRCX, {binop.op2.var.offset}[RBP]\n\t")
                x86_code.append(f"{mnemonic}\tRAX, RCX\n\t")
                x86_code.append(f"MOVQ\t{binop.op1.var.offset}[RBP], RAX\n\t")
    return ''.join(x86_code)

def align(n: int, m: int) -> int:
    r = n % m
    return n if r == 0 else n-r+m

def x86_func(func: TACFunc) -> str:
    asm_func = f"_{func.name}::\n\tPUSH\tRBP\n\tMOVQ\tRBP, RSP\n\t"
    body = cast(TAClist,func.body)
    local_size = 0
    stack_space = 0
    new_offset = 0
    offset = 0
    for locl in func.locals:
        if locl.type:
            local_size += align(locl.type.size,8)
            new_offset -= align(locl.type.size,8)
            locl.offset = new_offset
    for param in func.params:
        if param.type:
            local_size += align(param.type.size,8)
    if local_size > 0:
        stack_space = align(local_size,16)
        asm_func += f"SUB\tRSP,{stack_space}\n\t"
    offset = stack_space
    arg = 2
    ireg = 0
    for _ in func.params:
        if ireg == 6:
            off_ = arg * 8
            asm_func += f"MOVQ\tRAX, {off_}[RBP]\n\tMOVQ\t{-offset}[RBP], RAX\n\t"
            arg += 1
        else:
            asm_func += f"MOVQ\t{-offset}[RBP], {x86_registers[ireg]}\n\t"
            ireg += 1
    return f"{asm_func}{x86(body.tac_list,stack_space)}"

def make_add_i64(a: Ast, b: Ast) -> Ast: return AstBinaryOp(ast_type_i64,a,OP_PLUS,b)
def make_shl_i64(a: Ast, b: Ast) -> Ast: return AstBinaryOp(ast_type_i64,a,OP_SHL,b)

def main():
    if len(sys.argv) < 2:
        panic(f"Usage: {sys.argv[0]} <file>.c")
    with open(sys.argv[1]) as f:
        code = f.read()
    print("C ======")
    print(code)

    print("LEX =====")
    tokens = lexc(code)
    for tok in tokens: print(f"'{tok}' ", end='')
    print('\n')

    print("AST =====")
    parser = Parser(tokens)
    ast_list = parser.parse()
    for ast in ast_list:
        print(ast)

    print("TAC =====")
    ir_list = ir_gen(ast_list)
    for it in ir_list:
        print(it)

    print("x86 =====")
    asm_funcs = []
    for it in ir_list:
        if not isinstance(it,TACFunc):
            panic("Only ir functions are supported")
        asm_funcs.append(x86_func(it))
    for asm in asm_funcs:
        print(asm)

if __name__ == "__main__":
    main()
