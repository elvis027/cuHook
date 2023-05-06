#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "codegen.h"

#define MAX_NUM_OF_SYMBOLS 6000
#define CUDA_DEPRECATED_STR "__CUDA_DEPRECATED"
#define CUDNN_DEPRECATED_STR "CUDNN_DEPRECATED"

int curr_idx;
Symbol symbol_table[MAX_NUM_OF_SYMBOLS];
DataType curr_type;
char *curr_type_str;
int curr_scope;
Deprecated_t curr_deprecated;
int curr_is_array;

/* Init and Fini Functions */
void init_codegen()
{
    curr_idx = 0;
    curr_scope = 0;
    clear_curr();
}

void fini_codegen()
{
    for(int i = 0; i < curr_idx; i++) {
        if(symbol_table[i].type_str != NULL)
            free(symbol_table[i].type_str);
        if(symbol_table[i].name != NULL)
            free(symbol_table[i].name);
    }

    init_codegen();
}

/* Symbol Table Functions */
void set_curr_symbol_deprecated(Deprecated_t deprecated)
{
    curr_deprecated = deprecated;
}

void set_curr_symbol_type(DataType type)
{
    curr_type = type;
}

void set_curr_symbol_func_type(DataType type)
{
    curr_type = type;
}

void set_curr_symbol_type_str(char *type_str)
{
    curr_type_str = type_str;
}

void set_curr_symbol_array()
{
    curr_is_array = 1;
}

void add_symbol(char *name)
{
    if(curr_idx >= MAX_NUM_OF_SYMBOLS) {
        yyerror("Symbol table is full");
    }
    else {
        symbol_table[curr_idx].type = curr_type;
        symbol_table[curr_idx].type_str = curr_type_str;
        symbol_table[curr_idx].name = name;
        symbol_table[curr_idx].scope = curr_scope;
        symbol_table[curr_idx].deprecated = curr_deprecated;
        symbol_table[curr_idx].is_array = curr_is_array;
        curr_idx++;

        curr_type_str = NULL;
        clear_curr();
    }
}

void set_curr_symbol_scope_inc()
{
    curr_scope++;
}

void set_curr_symbol_scope_dec()
{
    curr_scope--;
}

void set_func_params_scope(char *func_name)
{
    int func_idx = find_symbol(func_name);
    symbol_table[func_idx].num_args = curr_idx - func_idx - 1;
    for(int i = curr_idx - 1; i > func_idx; i--) {
        symbol_table[i].scope = curr_scope;
    }
}

/* Codegen Function */
void codegen()
{
    if(symbol_table[0].type == T_CUDA_FUNC)
        codegen_cuda_hook();
    else if(symbol_table[0].type == T_CUDNN_FUNC)
        codegen_cudnn_hook();
}

void codegen_cuda_hook()
{
    printf("// hook.cpp\n");
    printf("void *dlsym(void *handle, const char *symbol)\n{\n");
    printf("    /* Hook functions for cuda version < 11.3 */\n");
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDA_FUNC)
            continue;

        printf("    ");
        if(idx != 0) printf("else ");
        printf("if(strcmp(symbol, SYMBOL_STRING(%s)) == 0) {\n", symbol_table[idx].name);
        printf("        return reinterpret_cast<void *>(%s);\n", symbol_table[idx].name);
        printf("    }\n");
    }
    printf("}\n\n");

    printf("// cuda_hook.hpp\n");
    printf("enum cuda_hook_symbols\n{\n");
    int symbols_count = 0;
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDA_FUNC)
            continue;
        printf("    ");
        print_symbol_enum(symbol_table[idx].name);
        printf(",\n");
        ++symbols_count;
    }
    printf("    NUM_CUDA_HOOK_SYMBOLS = %d\n};\n\n", symbols_count);

    printf("// cuda_hook.cpp\n");
    printf("/* ****************************** replace posthook of cuGetProcAddress() ****************************** */\n");
    printf("/* cuGetProcAddress() is the entry of cuda api functions for cuda version >= 11.3 */\n");
    printf("CUresult cuGetProcAddress_posthook(\n");
    printf("    const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)\n");
    printf("{\n");
    printf("    hook_log.debug(\"cuGetProcAddress: symbol \"s + string(symbol) + \", cudaVersion \"s + std::to_string(cudaVersion));\n\n");
    printf("    /* Hook functions for cuda version >= 11.3 */\n");
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDA_FUNC)
            continue;

        printf("    ");
        if(idx != 0) printf("else ");
        printf("if(strcmp(symbol, \"%s\") == 0) {\n", symbol_table[idx].name);

        printf("        cuda_hook_info.func_actual[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] = *pfn;\n");

        printf("        *pfn = reinterpret_cast<void *>(%s);\n", symbol_table[idx].name);
        printf("    }\n");
    }
    printf("    \\\\trace_dump.dump(\"cuGetProcAddress\");\n");
    printf("    return CUDA_SUCCESS;\n");
    printf("}\n");
    printf("/* ****************************** replace posthook of cuGetProcAddress() ****************************** */\n\n");

    printf("/* prehook, proxy, posthook functions start */\n");
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDA_FUNC)
            continue;

        printf("CUresult %s_prehook(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if((i - 1) % 4 == 0) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    return CUDA_SUCCESS;\n");
        printf("}\n\n");

        printf("CUresult %s_proxy(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if((i - 1) % 4 == 0) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    typedef decltype(&%s) func_type;\n", symbol_table[idx].name);
        printf("    void *actual_func;\n");

        printf("    if(!(actual_func = cuda_hook_info.func_actual[");
        print_symbol_enum(symbol_table[idx].name);
        printf("])) {\n");

        printf("        actual_func = actual_dlsym(libcuda_handle, SYMBOL_STRING(%s));\n",
            symbol_table[idx].name);

        printf("        cuda_hook_info.func_actual[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] = actual_func;\n");
        printf("    }\n");

        printf("    return ((func_type)actual_func)(");
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            printf("%s", symbol_table[idx + i].name);
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n        ");
        }
        printf(");\n");
        printf("}\n\n");

        printf("CUresult %s_posthook(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if((i - 1) % 4 == 0) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    trace_dump.dump(\"%s\");\n", symbol_table[idx].name);
        printf("    return CUDA_SUCCESS;\n");
        printf("}\n\n");
    }
    printf("/* prehook, proxy, posthook functions end */\n\n");

    printf("static void cuda_hook_init()\n{\n");
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDA_FUNC)
            continue;

        printf("    cuda_hook_info.func_prehook[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] =\n");
        printf("        reinterpret_cast<void *>(%s_prehook);\n", symbol_table[idx].name);

        printf("    cuda_hook_info.func_proxy[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] =\n");
        printf("        reinterpret_cast<void *>(%s_proxy);\n", symbol_table[idx].name);

        printf("    cuda_hook_info.func_posthook[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] =\n");
        printf("        reinterpret_cast<void *>(%s_posthook);\n", symbol_table[idx].name);
    }
    printf("}\n\n");

    printf("/* hook function start */\n");
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDA_FUNC)
            continue;

        printf("CUDA_HOOK_GEN(\n");

        printf("    ");
        print_symbol_enum(symbol_table[idx].name);
        printf(",\n");

        printf("    ");
        if(symbol_table[idx].deprecated == CUDA)
            printf(CUDA_DEPRECATED_STR);
        printf(",\n");

        printf("    %s,\n", symbol_table[idx].name);

        printf("    (");
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n    ");
        }
        printf("),\n");

        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if((i - 1) % 4 == 0) printf("    ");
            printf("%s", symbol_table[idx + i].name);
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n");
        }
        printf(")\n\n");
    }
    printf("/* hook function end */\n");
}

void codegen_cudnn_hook()
{
    printf("// cudnn_hook.hpp\n");
    printf("enum cudnn_hook_symbols\n{\n");
    int symbols_count = 0;
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDNN_FUNC)
            continue;
        printf("    ");
        print_symbol_enum(symbol_table[idx].name);
        printf(",\n");
        ++symbols_count;
    }
    printf("    NUM_CUDNN_HOOK_SYMBOLS = %d\n};\n\n", symbols_count);

    printf("// cudnn_hook.cpp\n");
    printf("/* prehook, proxy, posthook functions start */\n");
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDNN_FUNC)
            continue;
        if(symbol_table[idx + 1].type != T_HANDLE_T &&
           symbol_table[idx + 1].type != T_HANDLE_T_PTR)
            continue;

        printf("cudnnStatus_t %s_prehook(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(i & 1) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR ||
               symbol_table[idx + i].type == T_HANDLE_T_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    return CUDNN_STATUS_SUCCESS;\n");
        printf("}\n\n");

        printf("cudnnStatus_t %s_proxy(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(i & 1) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR ||
               symbol_table[idx + i].type == T_HANDLE_T_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    typedef decltype(&%s) func_type;\n", symbol_table[idx].name);
        printf("    void *actual_func;\n");

        printf("    if(!(actual_func = cudnn_hook_info.func_actual[");
        print_symbol_enum(symbol_table[idx].name);
        printf("])) {\n");

        printf("        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(%s));\n",
            symbol_table[idx].name);

        printf("        cudnn_hook_info.func_actual[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] = actual_func;\n");
        printf("    }\n");

        printf("    return ((func_type)actual_func)(\n");
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if((i - 1) % 4 == 0) printf("        ");
            printf("%s", symbol_table[idx + i].name);
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n");
        }
        printf(");\n");
        printf("}\n\n");

        printf("cudnnStatus_t %s_posthook(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(i & 1) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR ||
               symbol_table[idx + i].type == T_HANDLE_T_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    trace_dump.dump(\"%s\");\n", symbol_table[idx].name);
        printf("    return CUDNN_STATUS_SUCCESS;\n");
        printf("}\n\n");
    }

    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDNN_FUNC)
            continue;
        if(symbol_table[idx + 1].type == T_HANDLE_T ||
           symbol_table[idx + 1].type == T_HANDLE_T_PTR)
            continue;

        printf("cudnnStatus_t %s_prehook(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(i & 1) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR ||
               symbol_table[idx + i].type == T_HANDLE_T_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    return CUDNN_STATUS_SUCCESS;\n");
        printf("}\n\n");

        printf("cudnnStatus_t %s_proxy(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(i & 1) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR ||
               symbol_table[idx + i].type == T_HANDLE_T_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    typedef decltype(&%s) func_type;\n", symbol_table[idx].name);
        printf("    void *actual_func;\n");

        printf("    if(!(actual_func = cudnn_hook_info.func_actual[");
        print_symbol_enum(symbol_table[idx].name);
        printf("])) {\n");

        printf("        actual_func = actual_dlsym(libcudnn_handle, SYMBOL_STRING(%s));\n",
            symbol_table[idx].name);

        printf("        cudnn_hook_info.func_actual[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] = actual_func;\n");
        printf("    }\n");

        printf("    return ((func_type)actual_func)(\n");
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if((i - 1) % 4 == 0) printf("        ");
            printf("%s", symbol_table[idx + i].name);
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n");
        }
        printf(");\n");
        printf("}\n\n");

        printf("cudnnStatus_t %s_posthook(\n", symbol_table[idx].name);
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(i & 1) printf("    ");
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR ||
               symbol_table[idx + i].type == T_HANDLE_T_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n");
        }
        printf(")\n{\n");
        printf("    trace_dump.dump(\"%s\");\n", symbol_table[idx].name);
        printf("    return CUDNN_STATUS_SUCCESS;\n");
        printf("}\n\n");
    }
    printf("/* prehook, proxy, posthook functions end */\n\n");

    printf("static void cudnn_hook_init()\n{\n");
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDNN_FUNC)
            continue;

        printf("    cudnn_hook_info.func_prehook[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] =\n");
        printf("        reinterpret_cast<void *>(%s_prehook);\n", symbol_table[idx].name);

        printf("    cudnn_hook_info.func_proxy[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] =\n");
        printf("        reinterpret_cast<void *>(%s_proxy);\n", symbol_table[idx].name);

        printf("    cudnn_hook_info.func_posthook[");
        print_symbol_enum(symbol_table[idx].name);
        printf("] =\n");
        printf("        reinterpret_cast<void *>(%s_posthook);\n", symbol_table[idx].name);
    }
    printf("}\n\n");

    printf("/* hook function start */\n");
    // CUDNN_HANDLE_HOOK_GEN
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDNN_FUNC)
            continue;
        if(symbol_table[idx + 1].type != T_HANDLE_T &&
           symbol_table[idx + 1].type != T_HANDLE_T_PTR)
            continue;

        printf("CUDNN_HANDLE_HOOK_GEN(\n");

        printf("    ");
        print_symbol_enum(symbol_table[idx].name);
        printf(",\n");

        printf("    ");
        if(symbol_table[idx].deprecated == CUDNN)
            printf(CUDNN_DEPRECATED_STR);
        printf(",\n");

        printf("    %s,\n", symbol_table[idx].name);

        printf("    (");
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR ||
               symbol_table[idx + i].type == T_HANDLE_T_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n    ");
        }
        printf("),\n");

        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if((i - 1) % 4 == 0) printf("    ");
            printf("%s", symbol_table[idx + i].name);
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n");
        }
        printf(")\n\n");
    }

    // CUDNN_HOOK_GEN
    for(int idx = 0; idx < curr_idx; idx++) {
        if(symbol_table[idx].type != T_CUDNN_FUNC)
            continue;
        if(symbol_table[idx + 1].type == T_HANDLE_T ||
           symbol_table[idx + 1].type == T_HANDLE_T_PTR)
            continue;

        printf("CUDNN_HOOK_GEN(\n");

        printf("    ");
        print_symbol_enum(symbol_table[idx].name);
        printf(",\n");

        printf("    ");
        if(symbol_table[idx].deprecated == CUDNN)
            printf(CUDNN_DEPRECATED_STR);
        printf(",\n");

        printf("    %s,\n", symbol_table[idx].name);

        printf("    (");
        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if(symbol_table[idx + i].type == T_OTHERS_PTR ||
               symbol_table[idx + i].type == T_VOID_PTR ||
               symbol_table[idx + i].type == T_HANDLE_T_PTR)
                printf("%s%s", symbol_table[idx + i].type_str,
                               symbol_table[idx + i].name);
            else
                printf("%s %s", symbol_table[idx + i].type_str,
                                symbol_table[idx + i].name);
            if(symbol_table[idx + i].is_array)
                printf("[]");
            if(i != symbol_table[idx].num_args)
                if(i & 1) printf(", ");
                else printf(",\n    ");
        }
        printf("),\n");

        for(int i = 1; i <= symbol_table[idx].num_args; i++) {
            if((i - 1) % 4 == 0) printf("    ");
            printf("%s", symbol_table[idx + i].name);
            if(i != symbol_table[idx].num_args)
                if(i % 4) printf(", ");
                else printf(",\n");
        }
        printf(")\n\n");
    }
    printf("/* hook function end */\n");
}

/* Utility Function */
void list_symbol_table()
{
    printf("=== symbol table start ===\n");
    for(int i = 0; i < curr_idx; i++) {
        printf("index: %d\n", i);
        printf("name: %s\n", symbol_table[i].name);
        printf("type: %d\n", symbol_table[i].type);
        printf("type_str: %s\n", symbol_table[i].type_str);
        printf("scope: %d\n", symbol_table[i].scope);
        printf("num_args: %d\n", symbol_table[i].num_args);
        printf("deprecated: %d\n", symbol_table[i].deprecated);
        printf("is_array: %d\n\n", symbol_table[i].is_array);
    }
    printf("=== symbol table end ===\n");
}

int find_symbol(char *name)
{
    int index = -1;
    for(index = curr_idx - 1; index >= 0; index--) {
        if(symbol_table[index].name && !strcmp(name, symbol_table[index].name))
            return index;
    }
    if(index < 0) {
        list_symbol_table();
        yyerror("Symbol is not found");
    }
    return index;
}

void print_symbol_enum(char *name)
{
    char last = 0, curr = 0, next = 0;
    for(int i = 0; i < strlen(name); i++) {
        curr = name[i];
        last = (i - 1 >= 0) ? name[i - 1] : curr;
        next = (i + 1 < strlen(name)) ? name[i + 1] : curr;

        if(i == strlen(name) - 1 && isdigit(curr))
            printf("%c", toupper(curr));
        else if(isupper(curr) || isdigit(curr))
            if(islower(last) || (isupper(last) || isdigit(last)) && islower(next))
                printf("_%c", toupper(curr));
            else
                printf("%c", toupper(curr));
        else
            printf("%c", toupper(curr));
    }
}

void clear_curr()
{
    curr_type = T_UNKNOWN;
    if(curr_type_str != NULL)
        free(curr_type_str);
    curr_type_str = NULL;
    curr_deprecated = NON;
    curr_is_array = 0;
}
