ENTRY(_start)
SECTIONS
{
    . = 0x80000000;
    .text : { *(.text.init) *(.text*) }
    .rodata : { *(.rodata*) }
    . = ALIGN(64);
    .data : { *(.data*) *(.sdata*) }
    . = ALIGN(64);
    __bss_start = .;
    .bss : { *(.bss*) *(.sbss*) *(COMMON) }
    . = ALIGN(8);
    __bss_end = .;
}
