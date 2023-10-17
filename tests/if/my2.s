	.align 16
_block.15:
    movq %rbx, %rdi
    callq _print_int
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.16:
    movq %rbx, %rbx
    addq $2, %rbx
    jmp _block.15

	.align 16
_block.17:
    movq %rbx, %rbx
    addq $10, %rbx
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
    popq %r12
    popq %rbx
    popq %rbp
    retq 

	.globl _main
	.align 16
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r12
    jmp _start


