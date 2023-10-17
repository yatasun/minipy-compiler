	.align 16
_block.0:
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.1:
    movq $1, %rdi
    callq _print_int
    jmp _block.0

	.align 16
_block.2:
    movq $2, %rdi
    callq _print_int
    jmp _block.0

	.align 16
_block.3:
    cmpq %rbx, %r13
    jl _block.1
    jmp _block.2

	.align 16
_block.4:
    movq $3, %rdi
    callq _print_int
    jmp _block.0

	.align 16
_start:
    callq _read_int
    movq %rax, %r12
    callq _read_int
    movq %rax, %r13
    callq _read_int
    movq %rax, %rbx
    cmpq %r13, %r12
    jl _block.3
    jmp _block.4

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


