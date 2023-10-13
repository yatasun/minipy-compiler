	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r12
    movq $10, %r12
    movq $20, %rbx
    negq %rbx
    addq %rbx, %r12
    movq %r12, %rdi
    callq _print_int
    popq %r12
    popq %rbx
    popq %rbp
    retq 

