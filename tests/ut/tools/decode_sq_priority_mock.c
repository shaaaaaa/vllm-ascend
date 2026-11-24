// SPDX-License-Identifier: Apache-2.0
// Exercise the actual test fixture with a CPU HAL backend and fresh state per invocation.
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int (*Allocate)(uint32_t, void *, void *);
int sq_test_arm(Allocate, uint32_t, uint32_t);
int halSqCqAllocate(uint32_t, void *, void *);
int sq_test_lookup(uint32_t, uint32_t *);
unsigned sq_test_attempts(void);

static uint32_t expected[15], expected_device;
static int call_num, fail_first;

static int driver(uint32_t device, void *input, void *output) {
    assert(device == expected_device);
    assert(memcmp(input, expected, sizeof(expected)) == 0);
    *(uint32_t *)output = 50u + (uint32_t)++call_num;
    return fail_first && call_num == 6 ? 17 : 0;
}

int main(int argc, char **argv) {
    assert(argc == 3);
    uint32_t first = (uint32_t)atoi(argv[1]);
    fail_first = atoi(argv[2]);
    assert(sq_test_arm(driver, 3, 5) != 0);
    assert(sq_test_arm(driver, 3, first) == 0);
    assert(sq_test_arm(driver, 3, first) != 0);
    for (int i = 0; i < 8; ++i) {
        uint32_t input[15] = {0, 0, 64, 16, 2048, 1024, 0, 1u << 16, 0, 0, UINT32_MAX, 0, 8, 0, 0};
        expected_device = i == 0 ? 4 : 3;
        if (i == 1) input[7] = 0;  // ordinary SQ
        if (i == 2) input[10] = 3; // named stream
        if (i == 3) input[12] = 0; // hardware SQ
        if (i == 4) input[11] = 5; // caller has its own priority
        uint32_t original[15], id, priority;
        memcpy(original, input, sizeof(input));
        memcpy(expected, input, sizeof(input));
        if (i == 5 || i == 6) expected[11] = i == 5 ? first : 7 - first;
        int rc = halSqCqAllocate(expected_device, input, &id);
        assert(memcmp(original, input, sizeof(input)) == 0);
        assert(rc == (fail_first && i == 5 ? 17 : 0));
        int found = sq_test_lookup(id, &priority) == 0;
        assert(found == (rc == 0 && (i == 5 || i == 6)));
        if (found) assert(priority == expected[11]);
    }
    assert(sq_test_attempts() == 3);
    return 0;
}
