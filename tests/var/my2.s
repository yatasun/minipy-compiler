	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r13
    pushq %r14
    pushq %r12
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
    popq %r12
    popq %r14
    popq %r13
    popq %rbx
    popq %rbp
    retq 

