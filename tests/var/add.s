	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq $40, -8(%rbp)
    addq $2, -8(%rbp)
    movq -8(%rbp), %rdi
    callq _print_int
    addq $16, %rsp
    popq %rbp
    retq 

