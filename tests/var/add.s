	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    subq $8, %rsp
    movq $40, -8(%rbp)
    addq $2, -8(%rbp)
    movq -8(%rbp), %rdi
    callq _print_int
    addq $8, %rsp
    popq %rbp
    retq 

