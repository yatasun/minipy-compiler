	.align 16
_block.10:
    movq %rbx, %rdi
    callq _print_int
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.11:
    movq $2, %rbx
    jmp _block.10

	.align 16
_block.12:
    movq $3, %rbx
    jmp _block.10

	.align 16
_block.13:
    movq $1, %rax
    cmpq $1, %rax
    je _block.11
    jmp _block.12

	.align 16
_start:
    movq $1, %rbx
    cmpq $1, %rbx
    je _block.13
    jmp _block.12

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


