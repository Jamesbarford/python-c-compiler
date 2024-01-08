# c compiler in python


This is a toy compiler attempting to compile a subset of c in less than 1000 
lines of code. Not much is supported as a result. I wrote this over a *weekend 
as a challenge, or more realistically procrastinating from what I should be doing.

As a result this can only handle basic integer operations! Although the
parser has support for a `while` with break and continue, an `if` and
function calls. I've not built these out as I ran out of time. (when I next 
procrastinate I might add them :D).

I use `sys` to be able to access the command line arguments, `cast` to 
hack around pythons type system and `NoReturn` for my implementation of `panic`.
Other than that there are no python libraries used.

# Usage
```sh
./cc.py <file>
```

## Output:
This shows all of the stages used to generate the x86 code.
```
C ======
int main() {
    int x = 4 + 5;
    int y = x + 9;
    return y;
}

LEX =====
'int' 'main' '(' ')' '{' 'int' 'x' '=' '4' '+' '5' ';' 'int' 'y' '=' 'x' '+' '9' ';' 'return' 'y' ';' '}' 

AST =====
here
<fun> int main
<params>()
<compound>
	<decl> int 
		<lvar> x = <binop>
		(4 + 5)
	<decl> int 
		<lvar> y = <binop>
		(<lvar> x + 9)
	<return> <lvar> y
TAC =====
main::
	R0 = ADD @i64::4, @i64::5
	R1 = R0
	R2 = ADD R1, @i64::9
	R3 = R2
	RET R3

x86 =====
_main::
	PUSH	RBP
	MOVQ	RBP, RSP
	SUB	RSP,16
	MOVQ	RAX, 4
	MOVQ	RCX, 5
	ADD	RAX, RCX
	MOVQ	-8[RBP], RAX
	MOVQ	RCX, 9
	MOVQ	RAX, -8[RBP]
	ADD	RAX, RCX
	MOVQ	-8[RBP], RAX
	MOVQ	-16[RBP], RAX
	ADD	RSP, 16
	MOVQ	RAX, -16[RBP]
	LEAVE
	RET
```

# Components

## Lexer
- This is classical lexer splitting the code into a list of tokens without
  using pythons `re` library.

## Parser 
- Essentially an LL recursive decent parser, precidence is essentially operator
  climbing.

## IR - TAC
- A three Address Code intermediate representation is used to flatten out the
  ast into something that is easier to convert to assembly

## x86_64
- A semi-realistic intel style assembly, I'm not too worried that the x86_64
  may not run, it's out side of the scope of this project.

# Inspirations & Resources
- [Compiler design in c](https://holub.com/compiler/)
- [8cc](https://github.com/rui314/8cc)
- [parsing expressions by precidence climbing](https://eli.thegreenplace.net/2012/08/02/parsing-expressions-by-precedence-climbing)
- [Crafting interpreters](https://craftinginterpreters.com/)
- [Engineering a compiler - 3rd edition](https://www.amazon.com/Engineering-Compiler-Keith-D-Cooper/dp/0128154128)
- [tcc](http://bellard.org/tcc/)
- [cc65](https://cc65.github.io/)
- [TempleOS](https://templeos.org/)
- [JS c compiler](https://github.com/Captainarash/CaptCC)

## Twitch
I semi-frequently stream on twitch at: https://twitch.tv/Jamesbarford mostly
c and mostly compiler related.

*Taking a weekend -
I'm building a more _real_ compiler in `c` for my masters. There's months of 
learning crammed into these rather shoddy ~900 of code.
