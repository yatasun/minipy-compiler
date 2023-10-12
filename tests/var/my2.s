	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    subq $56, %rsp
    callq _read_int
    movq %rax, -8(%rbp)
    callq _read_int
    movq %rax, -16(%rbp)
    movq $1000, -24(%rbp)
    movq $2000, -32(%rbp)
    negq -32(%rbp)
    movq -8(%rbp), %rax
    movq %rax, -40(%rbp)
    movq -16(%rbp), %rax
    addq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    movq %rax, -48(%rbp)
    movq -24(%rbp), %rax
    subq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    movq %rax, -56(%rbp)
    movq -32(%rbp), %rax
    addq %rax, -56(%rbp)
    movq -56(%rbp), %rdi
    callq _print_int
    addq $56, %rsp
    popq %rbp
    retq 

