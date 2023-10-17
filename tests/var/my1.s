	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %r12
    pushq %rbx
    movq $10, %r12
    movq $20, %rbx
    negq %rbx
    addq %rbx, %r12
    movq %r12, %rdi
    callq _print_int
    popq %rbx
    popq %r12
    popq %rbp
    retq 

