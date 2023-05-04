#ifndef _CODEGEN_H_
#define _CODEGEN_H_

typedef enum
{
    T_UNKNOWN,
    T_CUDA_FUNC,
    T_CUDNN_FUNC,
    T_HANDLE_T,
    T_HANDLE_T_PTR,
    T_VOID_PTR,
    T_OTHERS,
    T_OTHERS_PTR
} DataType;

typedef enum
{
    NON,
    CUDA,
    CUDNN
} Deprecated_t;

typedef struct
{
    DataType type;              // symbol type
    char *type_str;             // type string
    char *name;                 // symbol name
    int scope;                  // variable scope
    int num_args;               // [For Function] number of arguments
    Deprecated_t deprecated;    // [For Function] NON / CUDA / CUDNN
    int is_array;               // append []
} Symbol;

/* Defined in parser.y */
void yyerror(const char *msg);

/* Init and Fini Functions */
void init_codegen();
void fini_codegen();

/* Symbol Table Functions */
void set_curr_symbol_deprecated(Deprecated_t deprecated);
void set_curr_symbol_type(DataType type);
void set_curr_symbol_func_type(DataType type);
void set_curr_symbol_type_str(char *type_str);
void set_curr_symbol_array();
void add_symbol(char *name);
void set_curr_symbol_scope_inc();
void set_curr_symbol_scope_dec();
void set_func_params_scope(char *func_name);

/* Codegen Functions */
void codegen();
void codegen_cuda_hook();
void codegen_cudnn_hook();

/* Utility Functions */
void list_symbol_table();
int find_symbol(char *name);
void print_symbol_enum(char *name);
void clear_curr();

#endif /* _CODEGEN_H_ */
