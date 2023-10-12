	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    subq $24, %rsp
    movq $10, -8(%rbp)
    movq $20, -16(%rbp)
    negq -16(%rbp)
    movq -8(%rbp), %rax
    movq %rax, -24(%rbp)
    movq -16(%rbp), %rax
    addq %rax, -24(%rbp)
    movq -24(%rbp), %rdi
    callq _print_int
    addq $24, %rsp
    popq %rbp
    retq 

