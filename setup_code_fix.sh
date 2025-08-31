#!/bin/bash

# 配置代码补全
# 1. 首先生成编译数据库
workspace='/home/wangjiyuan'
project_dir="${workspace}/dfnn_my_mlir"
build_dir="${project_dir}/build"

# 进入构建目录
cd "${build_dir}" || exit 1

# 让CMake生成编译数据库
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

# 创建符号链接到项目根目录
ln -sf "${build_dir}/compile_commands.json" "${project_dir}/compile_commands.json"

# 检查编译数据库是否生成
ls -la "${project_dir}/compile_commands.json"
head -5 "${project_dir}/compile_commands.json"

# 2. 配置clangd（代码补全）
# 创建clangd配置目录和文件
mkdir -p ~/.config/clangd
cat > ~/.config/clangd/config.yaml << EOF
CompileFlags:
  Add:
    - -I${workspace}/llvm_install/include
    - -I${workspace}/miniconda3/envs/py312/include/python3.12
    - -std=c++17
  CompilationDatabase: .
Diagnostics:
  Suppress: [unknown-warning-option, unused-argument]
EOF

# 3. 创建VSCode工作区配置
mkdir -p "${project_dir}/.vscode"

cat > "${project_dir}/.vscode/settings.json" << EOF
{
    "C_Cpp.default.compilerPath": "${workspace}/llvm_install/bin/clang++",
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.includePath": [
        "${workspace}/llvm_install/include",
        "${workspace}/miniconda3/envs/py312/include/python3.12",
        "\${workspaceFolder}/include"
    ],
    "clangd.path": "/data/llvm-main/bin/clangd",
    "clangd.arguments": [
        "--compile-commands-dir=\${workspaceFolder}/build",
        "--query-driver=${workspace}/llvm_install/bin/clang++",
        "--header-insertion=never"
    ],
    "cmake.configureArgs": [
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ],
    "editor.formatOnSave": true,
    "C_Cpp.formatting": "clangFormat",
    "C_Cpp.clang_format_path": "${workspace}/llvm_install/bin/clang-format",
}
EOF

# 4. 创建.clang-format文件（代码格式化）
cat > "${project_dir}/.clang-format" << EOF
BasedOnStyle: LLVM
AccessModifierOffset: -2
AlignAfterOpenBracket: Align
AlignConsecutiveAssignments: false
AlignConsecutiveDeclarations: false
AlignEscapedNewlines: Left
AlignOperands: true
AlignTrailingComments: true
AllowAllParametersOfDeclarationOnNextLine: true
AllowShortBlocksOnASingleLine: false
AllowShortCaseLabelsOnASingleLine: false
AllowShortFunctionsOnASingleLine: Empty
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
AlwaysBreakAfterDefinitionReturnType: None
AlwaysBreakAfterReturnType: None
AlwaysBreakBeforeMultilineStrings: false
AlwaysBreakTemplateDeclarations: true
BinPackArguments: true
BinPackParameters: true
BraceWrapping:
  AfterClass: false
  AfterControlStatement: false
  AfterEnum: false
  AfterFunction: false
  AfterNamespace: false
  AfterObjCDeclaration: false
  AfterStruct: false
  AfterUnion: false
  BeforeCatch: false
  BeforeElse: false
  IndentBraces: false
BreakBeforeBinaryOperators: None
BreakBeforeBraces: Attach
BreakBeforeTernaryOperators: true
BreakConstructorInitializers: BeforeComma
ColumnLimit: 80
CommentPragmas: '^ IWYU pragma:'
CompactNamespaces: false
ConstructorInitializerAllOnOneLineOrOnePerLine: true
ConstructorInitializerIndentWidth: 4
ContinuationIndentWidth: 4
Cpp11BracedListStyle: true
DerivePointerAlignment: false
DisableFormat: false
FixNamespaceComments: true
IncludeCategories:
  - Regex: '^<.*\.h>'
    Priority: 1
  - Regex: '^<.*'
    Priority: 2
  - Regex: '.*'
    Priority: 3
IncludeIsMainRegex: '([-_](test|unittest))?$'
IndentCaseLabels: false
IndentWidth: 4
IndentWrappedFunctionNames: false
KeepEmptyLinesAtTheStartOfBlocks: true
MaxEmptyLinesToKeep: 1
NamespaceIndentation: None
ObjCBlockIndentWidth: 4
ObjCSpaceAfterProperty: false
ObjCSpaceBeforeProtocolList: true
PenaltyBreakBeforeFirstCallParameter: 19
PenaltyBreakComment: 300
PenaltyBreakString: 1000
PenaltyBreakTemplateDeclaration: 10
PenaltyExcessCharacter: 1000000
PenaltyReturnTypeOnItsOwnLine: 60
PointerAlignment: Right
ReflowComments: true
SortIncludes: true
SortUsingDeclarations: true
SpaceAfterCStyleCast: false
SpaceAfterTemplateKeyword: false
SpaceBeforeAssignmentOperators: true
SpaceBeforeCpp11BracedList: false
SpaceBeforeCtorInitializerColon: true
SpaceBeforeInheritanceColon: true
SpaceBeforeParens: ControlStatements
SpaceBeforeRangeBasedForLoopColon: true
SpaceInEmptyParentheses: false
SpacesBeforeTrailingComments: 1
SpacesInAngles: false
SpacesInContainerLiterals: true
SpacesInCStyleCastParentheses: false
SpacesInParentheses: false
SpacesInSquareBrackets: false
TabWidth: 4
UseTab: Never
EOF

# 5. 测试配置
# # 测试clangd
# cd "${project_dir}" || exit 1
# "/data/llvm-main/bin/clangd" --check=src/simple_mlir.cpp

# # 测试代码格式化
# "/data/llvm-main/bin/clangd" -i src/simple_mlir.cpp --style=file

echo "配置完成！请重新启动VSCode或者重新加载窗口"
