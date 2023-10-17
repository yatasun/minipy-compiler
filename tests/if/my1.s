	.align 16
_block.6:
    movq %rbx, %rdi
    callq _print_int
    movq $0, %rax
    jmp _conclusion

	.align 16
_block.7:
    movq $2, %rbx
    jmp _block.6

	.align 16
_block.8:
    movq $3, %rbx
    jmp _block.6

	.align 16
_block.9:
    movq $1, %rax
    cmpq $1, %rax
    je _block.7
    jmp _block.8

	.align 16
_start:
    movq $1, %rbx
    cmpq $1, %rbx
    je _block.9
    jmp _block.8

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


