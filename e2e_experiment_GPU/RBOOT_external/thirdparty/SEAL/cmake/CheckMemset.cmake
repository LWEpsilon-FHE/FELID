# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

check_cxx_source_runs("
    int main(void)
    {
        char str[] = \"ghghghghghghghghghghgh\";
        int r = memset_s(str, sizeof(str), 'a', 5);
        return r;
    }"
    SEAL_MEMSET_S_FOUND)

check_symbol_exists(explicit_bzero "string.h" SEAL_EXPLICIT_BZERO_FOUND)

check_symbol_exists(explicit_memset "string.h" SEAL_EXPLICIT_MEMSET_FOUND)
