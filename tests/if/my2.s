	.align 16
_block.15:
    movq %r13, %rdi
    callq _print_int
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.16:
    movq %rbx, %r13
    addq $2, %r13
    jmp _block.15

	.align 16
_block.17:
    movq %rbx, %r13
    addq $10, %r13
    jmp _block.15

	.align 16
_block.18:
    cmpq $0, %r12
    je _block.16
    jmp _block.17

	.align 16
_block.19:
    cmpq $2, %r12
    je _block.16
    jmp _block.17

	.align 16
_start:
    callq _read_int
    movq %rax, %r12
    callq _read_int
    movq %rax, %rbx
    cmpq $1, %r12
    jl _block.18
    jmp _block.19

	.align 16
_conclusion:
    addq $8, %rsp
    popq %r13
    popq %rbx
    popq %r12
    popq %rbp
    retq 

	.globl _main
	.align 16
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %r12
    pushq %rbx
    pushq %r13
    subq $8, %rsp
    jmp _start


