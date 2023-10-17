	.align 16
_block.9:
    movq %rbx, %rdi
    callq _print_int
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.10:
    movq %r13, %rbx
    addq $2, %rbx
    jmp _block.9

	.align 16
_block.11:
    movq %r13, %rbx
    addq $10, %rbx
    jmp _block.9

	.align 16
_block.12:
    cmpq $0, %r12
    je _block.10
    jmp _block.11

	.align 16
_block.13:
    cmpq $2, %r12
    je _block.10
    jmp _block.11

	.align 16
_start:
    callq _read_int
    movq %rax, %r12
    callq _read_int
    movq %rax, %r13
    cmpq $1, %r12
    jl _block.12
    jmp _block.13

	.align 16
_conclusion:
    addq $8, %rsp
    popq %r12
    popq %rbx
    popq %r13
    popq %rbp
    retq 

	.globl _main
	.align 16
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %r13
    pushq %rbx
    pushq %r12
    subq $8, %rsp
    jmp _start


