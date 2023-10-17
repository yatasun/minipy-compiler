	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %rbx
    callq _read_int
    movq %rax, %rbx
    callq _read_int
    movq %rax, %r12
    movq $1000, %r13
    movq $2000, %r14
    negq %r14
    addq %r12, %rbx
    subq %r13, %rbx
    addq %r14, %rbx
    movq %rbx, %rdi
    callq _print_int
    popq %rbx
    popq %r14
    popq %r13
    popq %r12
    popq %rbp
    retq 

