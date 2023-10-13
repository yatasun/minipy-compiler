	.globl _main
_main:
    pushq %rbp
    movq %rsp, %rbp
    callq _read_int
    movq %rax, %rdx
    callq _read_int
    movq %rax, %rcx
    subq %rcx, %rdx
    movq %rdx, %rdi
    callq _print_int
    popq %rbp
    retq 