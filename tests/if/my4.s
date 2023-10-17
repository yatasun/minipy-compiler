	.align 16
_block.0:
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.1:
    movq $2, %rdi
    callq _print_int
    jmp _block.0

	.align 16
_block.2:
    movq $3, %rdi
    callq _print_int
    jmp _block.0

	.align 16
_start:
    callq _read_int
    movq %rax, %r12
    callq _read_int
    movq %rax, %rbx
    cmpq %rbx, %r12
    jl _block.1
    jmp _block.2

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


