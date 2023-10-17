
CC = gcc
CFLAGS = -arch x86_64

# %: runtime.o %.c
# 	gcc -g -std=c99 -arch x86_64 runtime.o

runtime.o: runtime.c
	gcc -c -g -std=c99 -arch x86_64 runtime.c

lldb:
	lldb a.out
debug:
	lldb a.out -o "br s -n print_int" -o "b "
dump:
	objdump	-d a.out