/* Minimal two-hart execution layer for litmus7-generated thread functions.
 * Synchronization is outside the tested bodies. Only hart 0 initializes,
 * collects, and prints; each epoch completes before the next initialization.
 */
#include <stdint.h>

#include "test.h"

#ifndef TRIALS
#define TRIALS 128
#endif
#ifndef SWAP_HARTS
#define SWAP_HARTS 0
#endif

static volatile unsigned go __attribute__((aligned(64)));
static volatile unsigned done __attribute__((aligned(64)));

static inline void
fence(void)
{
    __asm__ volatile("fence rw,rw" ::: "memory");
}

/* Leave bank access opportunities for the other hart while polling. */
static inline void
spin_pause(void)
{
    __asm__ volatile(".rept 16\n\tnop\n\t.endr" ::: "memory");
}

static void
puts_uart(const char *s)
{
    while (*s)
        *(volatile char *)0x40600004 = *s++;
}

static void
hex(uint64_t v)
{
    const char *digits = "0123456789abcdef";
    for (int shift = 60; shift >= 0; shift -= 4)
        *(volatile char *)0x40600004 = digits[(v >> shift) & 15];
}

static void __attribute__((noreturn))
finish(unsigned status)
{
    fence();
    register unsigned long code __asm__("a0") = status;
    __asm__ volatile(".word 0x0000006b" :: "r"(code) : "memory");
    for (;;) {}
}

static struct {
    int64_t values[NOUT];
    unsigned count;
} histogram[NSTATES + 1];
static unsigned used;

static void
record_outcome(const int64_t *values)
{
    unsigned i;
    for (i = 0; i < used; ++i) {
        unsigned j = 0;
        while (j < NOUT && histogram[i].values[j] == values[j])
            ++j;
        if (j == NOUT)
            break;
    }
    /* At most NSTATES legal tuples plus the first forbidden tuple. */
    if (i == used) {
        for (unsigned j = 0; j < NOUT; ++j)
            histogram[i].values[j] = values[j];
        ++used;
    }
    ++histogram[i].count;
}

static void
print_outcomes(void)
{
    for (unsigned i = 0; i < used; ++i) {
        puts_uart("LITMUS_OUTCOME ");
        hex(histogram[i].count);
        for (unsigned j = 0; j < NOUT; ++j) {
            puts_uart(" ");
            hex((uint64_t)(int64_t)histogram[i].values[j]);
        }
        puts_uart("\n");
    }
}

int
main(unsigned hart)
{
    const unsigned role = hart ^ SWAP_HARTS;
    for (unsigned epoch = 1; epoch <= TRIALS; ++epoch) {
        if (hart == 0) {
            litmus_init();
            fence();
            go = epoch;
        } else {
            while (go != epoch)
                spin_pause();
        }
        fence();
        /* Deterministic start skew explores different interleavings. */
        unsigned delay = ((epoch * 13 + role * 7) & 31);
        for (unsigned i = 0; i < delay; ++i)
            __asm__ volatile("nop");
        litmus_run(role);
        fence();
        if (hart == 1) {
            done = epoch;
        } else {
            while (done != epoch)
                spin_pause();
            fence();
            int64_t values[NOUT];
            for (unsigned i = 0; i < NOUT; ++i)
                values[i] = litmus_read(i);
            record_outcome(values);
            if (!litmus_allowed(values)) {
                print_outcomes();
                puts_uart("LITMUS_FORBIDDEN ");
                hex(epoch);
                puts_uart("\n");
                finish(1);
            }
        }
    }
    if (hart == 0) {
        print_outcomes();
        puts_uart("LITMUS_DONE ");
        hex(TRIALS);
        puts_uart("\n");
        finish(0);
    }
    for (;;) {}
}
