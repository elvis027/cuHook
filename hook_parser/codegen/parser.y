%{

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codegen.h"

#define STR_CAT(NUM_STR, STRS...)  \
    str_cat(NUM_STR, (char *[]){STRS})

#define STR_FREE(NUM_STR, STRS...)  \
    str_free(NUM_STR, (char *[]){STRS})

// str
char *str_cat(int num_str, char *str_arr[]);
char *str_free(int num_str, char *str_arr[]);

// scanner
extern int line_number;
int yylex(void);

%}

%start program
%union {
    char *sval;
}

/* terminal */
// type token
%token<sval> CONST SIGNED UNSIGNED VOID
%token<sval> INT CHAR SHORT LONG LONG_LONG FLOAT DOUBLE
%token<sval> SIZE_T INT32_T INT64_T UINT32_T UINT64_T

%token<sval> CUDAAPI_T CUDA_DEPRECATED_T
%token<sval> CURESULT_T CUDA_TYPE_T

%token<sval> CUDNNWINAPI_T CUDNN_DEPRECATED_T
%token<sval> CUDNN_STATUS_T CUDNN_HANDLE_T CUDNN_TYPE_T
// ID CONSTANT
%token<sval> IDENTIFIER
// OP
%token<sval> '*' '(' ')' '[' ']' ';' ','

/* nonterminal */
// program
%type<sval> func_decl_s
// func_decl_s
%type<sval> func_decl
// func_decl
%type<sval> type ident func_def_params
%type<sval> deprecated_empty api_macro
// type
%type<sval> const_empty signed_empty long_short_empty ptr_empty
%type<sval> long_short_char signed_float_void_others
// func_def_params
%type<sval> func_def_param

/* associative and precedence */
%left ';'
%left ','
%left '('

%%

program
    : func_decl_s   { STR_FREE(1, $1); }
    ;

func_decl_s
    : func_decl                 { $$ = STR_FREE(1, $1); }
    | func_decl_s func_decl     { $$ = STR_FREE(2, $1, $2); }
    ;

func_decl
    : deprecated_empty type api_macro ident '(' func_def_params ')' ';'
        {
            set_curr_symbol_scope_inc();
            set_func_params_scope($4);
            set_curr_symbol_scope_dec();
            $$ = STR_FREE(8, $1, $2, $3, $4, $5, $6, $7, $8);
        }
    ;

deprecated_empty
    : /* empty */           { $$ = NULL; }
    | CUDA_DEPRECATED_T
        {
            set_curr_symbol_deprecated(CUDA);
            $$ = STR_FREE(1, $1);
        }
    | CUDNN_DEPRECATED_T
        {
            set_curr_symbol_deprecated(CUDNN);
            $$ = STR_FREE(1, $1);
        }
    ;

type
    : const_empty signed_empty long_short_empty INT ptr_empty
        {
            if($5 != NULL) set_curr_symbol_type(T_OTHERS_PTR);
            else set_curr_symbol_type(T_OTHERS);
            set_curr_symbol_type_str(STR_CAT(5, $1, $2, $3, $4, $5));
            $$ = STR_FREE(5, $1, $2, $3, $4, $5);
        }
    | const_empty signed_empty long_short_char ptr_empty
        {
            if($4 != NULL) set_curr_symbol_type(T_OTHERS_PTR);
            else set_curr_symbol_type(T_OTHERS);
            set_curr_symbol_type_str(STR_CAT(4, $1, $2, $3, $4));
            $$ = STR_FREE(4, $1, $2, $3, $4);
        }
    | const_empty signed_float_void_others ptr_empty
        {
            if((!strcmp($2, "cudnnHandle_t") || !strcmp($2, "const cudnnHandle_t")))
                if($3 == NULL)
                    set_curr_symbol_type(T_HANDLE_T);
                else
                    set_curr_symbol_type(T_HANDLE_T_PTR);
            else if((!strcmp($2, "void") || !strcmp($2, "const void")) &&  $3 != NULL)
                set_curr_symbol_type(T_VOID_PTR);
            else if($3 != NULL)
                set_curr_symbol_type(T_OTHERS_PTR);
            else
                set_curr_symbol_type(T_OTHERS);
            set_curr_symbol_type_str(STR_CAT(3, $1, $2, $3));
            $$ = STR_FREE(3, $1, $2, $3);
        }
    | CONST ptr_empty
        {
            if($2 != NULL) set_curr_symbol_type(T_OTHERS_PTR);
            else set_curr_symbol_type(T_OTHERS);
            set_curr_symbol_type_str(STR_CAT(2, $1, $2));
            $$ = STR_FREE(2, $1, $2);
        }
    ;

api_macro
    : CUDAAPI_T
        {
            set_curr_symbol_func_type(T_CUDA_FUNC);
            $$ = STR_FREE(1, $1);
        }
    | CUDNNWINAPI_T
        {
            set_curr_symbol_func_type(T_CUDNN_FUNC);
            $$ = STR_FREE(1, $1);
        }
    ;

ident
    : /* empty */
        {
            clear_curr();
            $$ = NULL;
        }
    | IDENTIFIER
        {
            add_symbol(strdup($1));
            $$ = $1;
        }
    | IDENTIFIER '[' ']'
        {
            set_curr_symbol_array();
            add_symbol(strdup($1));
            $$ = STR_FREE(3, $1, $2, $3);
        }
    ;

func_def_params
    : /* empty */       { $$ = NULL; }
    | func_def_param    { $$ = STR_FREE(1, $1); }
    ;

const_empty
    : /* empty */   { $$ = NULL; }
    | CONST         { $$ = $1; }
    ;

signed_empty
    : /* empty */   { $$ = NULL; }
    | SIGNED        { $$ = $1; }
    | UNSIGNED      { $$ = $1; }
    ;

long_short_empty
    : /* empty */   { $$ = NULL; }
    | LONG_LONG     { $$ = $1; }
    | LONG          { $$ = $1; }
    | SHORT         { $$ = $1; }
    ;

ptr_empty
    : /* empty */   { $$ = NULL; }
    | '*'           { $$ = $1; }
    | '*' '*'
        {
            $$ = strdup("**");
            STR_FREE(2, $1, $2);
        }
    ;

long_short_char
    : LONG_LONG     { $$ = $1; }
    | LONG          { $$ = $1; }
    | SHORT         { $$ = $1; }
    | CHAR          { $$ = $1; }
    ;

signed_float_void_others
    : SIGNED                    { $$ = $1; }
    | UNSIGNED                  { $$ = $1; }
    | FLOAT                     { $$ = $1; }
    | DOUBLE                    { $$ = $1; }
    | VOID                      { $$ = $1; }
    | SIZE_T                    { $$ = $1; }
    | INT32_T                   { $$ = $1; }
    | INT64_T                   { $$ = $1; }
    | UINT32_T                  { $$ = $1; }
    | UINT64_T                  { $$ = $1; }
    | CURESULT_T                { $$ = $1; }
    | CUDNN_STATUS_T            { $$ = $1; }
    | CUDNN_HANDLE_T            { $$ = $1; }
    | CUDA_TYPE_T const_empty   { $$ = $1; }
    | CUDNN_TYPE_T const_empty  { $$ = $1; }
    ;

func_def_param
    : type ident
        { $$ = STR_FREE(2, $1, $2); }
    | func_def_param ',' type ident
        { $$ = STR_FREE(4, $1, $2, $3, $4); }
    ;

%%

int main(void)
{
    init_codegen();
    yyparse();
    codegen();
    fini_codegen();
    return 0;
}

void yyerror(const char *msg)
{
    fprintf(stderr, "Error at line %d: %s\n", line_number, msg);
    exit(1);
}

char *str_free(int num_str, char *str_arr[])
{
    for(int i = 0; i < num_str; i++)
        if(str_arr[i] != NULL) {
            free(str_arr[i]);
            str_arr[i] = NULL;
        }
    return NULL;
}

char *str_cat(int num_str, char *str_arr[])
{
    size_t buff_len = 1;

    for(int i = 0; i < num_str; i++)
        if(str_arr[i] != NULL)
            buff_len += strlen(str_arr[i]) + 1;

    char *buff = malloc(sizeof(char) * buff_len);
    buff[0] = '\0';

    for(int i = 0; i < num_str; i++)
        if(str_arr[i] != NULL)
            sprintf(buff + strlen(buff), "%s ", str_arr[i]);

    int last = (buff_len - 2 >= 0) ? buff_len - 2 : 0;
    buff[last] = '\0';

    return buff;
}
