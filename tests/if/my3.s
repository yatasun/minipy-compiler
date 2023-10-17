	.align 16
_block.22:
    movq %rbx, %rdi
    callq _print_int
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.23:
    movq $42, %rbx
    jmp _block.22

	.align 16
_block.24:
    movq $0, %rbx
    jmp _block.22

	.align 16
_start:
    callq _read_int
    movq %rax, %rbx
    cmpq $1, %rbx
    je _block.23
    jmp _block.24

	.align 16
_conclusion:
    addq $8, %rsp
    popq %rbx
    popq %rbp
    retq 

	.globl _main
	.align 16
_main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    subq $8, %rsp
    jmp _start


