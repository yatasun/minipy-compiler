	.align 16
_block.6:
    movq %rbx, %rdi
    callq _print_int
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.7:
    movq %r13, %rbx
    addq $2, %rbx
    jmp _block.6

	.align 16
_block.8:
    movq %r13, %rbx
    addq $10, %rbx
    jmp _block.6

	.align 16
_block.9:
    cmpq $0, %r12
    je _block.7
    jmp _block.8

	.align 16
_block.10:
    cmpq $2, %r12
    je _block.7
    jmp _block.8

	.align 16
_start:
    callq _read_int
    movq %rax, %r12
    callq _read_int
    movq %rax, %r13
    cmpq $1, %r12
    jl _block.9
    jmp _block.10

	.align 16
_conclusion:
    addq $8, %rsp
    popq %rbx
    popq %r13
    popq %r12
    popq %rbp
    retq 

	.globl _main
	.align 16
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %r12
    pushq %r13
    pushq %rbx
    subq $8, %rsp
    jmp _start


