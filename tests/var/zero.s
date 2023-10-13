	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    movq $0, %rdi
    callq _print_int
    popq %rbp
    retq 

