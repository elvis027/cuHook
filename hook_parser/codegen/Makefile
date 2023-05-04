FLEX = flex
YACC = byacc
CC = gcc

DEBUGFLAGS = -fsanitize=address -g -Wall
LDFLAGS = -lfl

TARGETS = lex.yy.c y.tab.c codegen

.PHONY: all
all: $(TARGETS)

.PHONY: v
v: y.tab.c

lex.yy.c: scanner.l
	$(FLEX) $^

y.tab.c: parser.y
	$(YACC) -vd $^

codegen: lex.yy.c y.tab.c codegen.c
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGETS) y.tab.h y.output
