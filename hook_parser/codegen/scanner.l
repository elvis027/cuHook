%{

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "codegen.h"
#include "y.tab.h"

#define PUSH_SVAL(str) (yylval.sval = strdup(str))

int line_number = 1;

%}

%x COMMENT_SIN
%x COMMENT_MUL

identifier  [a-zA-Z_][a-zA-Z0-9_]*

%%

"//"                    { BEGIN COMMENT_SIN; }
<COMMENT_SIN>[^\n]+     {}
<COMMENT_SIN>\n         { line_number++; BEGIN INITIAL; }

"/*"                    { BEGIN COMMENT_MUL; }
<COMMENT_MUL>"*/"       { BEGIN INITIAL; }
<COMMENT_MUL>[^*\n]+    {}
<COMMENT_MUL>"*"        {}
<COMMENT_MUL>\n         { line_number++; }

"void"                  { PUSH_SVAL(yytext); return VOID; }
"int"                   { PUSH_SVAL(yytext); return INT; }
"double"                { PUSH_SVAL(yytext); return DOUBLE; }
"float"                 { PUSH_SVAL(yytext); return FLOAT; }
"char"                  { PUSH_SVAL(yytext); return CHAR; }
"const"                 { PUSH_SVAL(yytext); return CONST; }
"signed"                { PUSH_SVAL(yytext); return SIGNED; }
"unsigned"              { PUSH_SVAL(yytext); return UNSIGNED; }
"short"                 { PUSH_SVAL(yytext); return SHORT; }
"long"                  { PUSH_SVAL(yytext); return LONG; }
"long long"             { PUSH_SVAL(yytext); return LONG_LONG; }
"size_t"                { PUSH_SVAL(yytext); return SIZE_T; }
"int32_t"               { PUSH_SVAL(yytext); return INT32_T; }
"int64_t"               { PUSH_SVAL(yytext); return INT64_T; }
"uint32_t"              { PUSH_SVAL(yytext); return UINT32_T; }
"uint64_t"              { PUSH_SVAL(yytext); return UINT64_T; }

"CUDNNWINAPI"           { PUSH_SVAL(yytext); return CUDNNWINAPI_T; }
"CUDNN_DEPRECATED"      { PUSH_SVAL(yytext); return CUDNN_DEPRECATED_T; }
"cudnnStatus_t"         { PUSH_SVAL(yytext); return CUDNN_STATUS_T; }
"cudnnHandle_t"         { PUSH_SVAL(yytext); return CUDNN_HANDLE_T; }
"libraryPropertyType"   { PUSH_SVAL(yytext); return CUDNN_TYPE_T; }
cudnn[a-zA-Z0-9_]*_t    { PUSH_SVAL(yytext); return CUDNN_TYPE_T; }

"CUDAAPI"               { PUSH_SVAL(yytext); return CUDAAPI_T; }
"__CUDA_DEPRECATED"     { PUSH_SVAL(yytext); return CUDA_DEPRECATED_T; }
"CUresult"              { PUSH_SVAL(yytext); return CURESULT_T; }
"CUDA_LAUNCH_PARAMS"    { PUSH_SVAL(yytext); return CUDA_TYPE_T; }
CU[a-zA-Z0-9_]*         { PUSH_SVAL(yytext); return CUDA_TYPE_T; }
cuda[a-zA-Z0-9_]*_t     { PUSH_SVAL(yytext); return CUDA_TYPE_T; }
cu[a-zA-Z0-9_]*_t       { PUSH_SVAL(yytext); return CUDA_TYPE_T; }

"*"                     { PUSH_SVAL(yytext); return '*'; }
"("                     { PUSH_SVAL(yytext); return '('; }
")"                     { PUSH_SVAL(yytext); return ')'; }
"["                     { PUSH_SVAL(yytext); return '['; }
"]"                     { PUSH_SVAL(yytext); return ']'; }
";"                     { PUSH_SVAL(yytext); return ';'; }
","                     { PUSH_SVAL(yytext); return ','; }
#.+                     {}

{identifier}            { PUSH_SVAL(yytext); return IDENTIFIER; }

[ \t]+                  {}
\n                      { line_number++; }

%%
