	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %r12
    pushq %rbx
    callq _read_int
    movq %rax, %rbx
    callq _read_int
    movq %rax, %r12
    subq %r12, %rbx
    movq %rbx, %rdi
    callq _print_int
    popq %rbx
    popq %r12
    popq %rbp
    retq 

