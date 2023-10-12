	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    subq $0, %rsp
    movq $0, %rdi
    callq _print_int
    addq $0, %rsp
    popq %rbp
    retq 

