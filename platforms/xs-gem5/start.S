/* Two-hart raw-image entry for configs/example/smt_idealkmhv3.py. */
.section .text.init
.globl _start
_start:
    csrr a0, mhartid
    li t0, 1
    bgtu a0, t0, park
    la t0, trap
    csrw mtvec, t0
    bnez a0, wait_boot
    la t0, __bss_start
    la t1, __bss_end
clear_bss:
    bgeu t0, t1, boot_done
    sd zero, 0(t0)
    addi t0, t0, 8
    j clear_bss
boot_done:
    fence rw, rw
    la t0, boot_ready
    li t1, 1
    sd t1, 0(t0)
    j enter
wait_boot:
    la t0, boot_ready
1:
    ld t1, 0(t0)
    beqz t1, 1b
    fence rw, rw
enter:
    la sp, stacks
    addi t0, a0, 1
    slli t0, t0, 12
    add sp, sp, t0
    call main
park:
    j park
trap:
    li t0, 0x40600004
    li t1, 33
    sb t1, 0(t0)
    li a0, 2
    .word 0x0000006b
    j park

.section .data
.balign 64
boot_ready: .dword 0
.section .bss
.balign 16
stacks: .skip 8192
.section .note.GNU-stack,"",@progbits
