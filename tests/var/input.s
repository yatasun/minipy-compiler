	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    callq _read_int
    movq %rax, -8(%rbp)
    movq -8(%rbp), %rdi
    callq _print_int
    addq $16, %rsp
    popq %rbp
    retq 

