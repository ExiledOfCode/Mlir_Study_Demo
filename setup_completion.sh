#!/bin/bash

# 配置代码补全 - dfnn-mlir项目专用版

# 0. 设置变量
workspace="$HOME/dfnn_my_mlir"  # 根据你的实际路径调整
llvm_install_dir="/data/llvm-main"  # 替换为你的LLVM安装路径

# 1. 生成编译数据库
cd "${workspace}/build" || exit 1

# 让CMake生成编译数据库
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

# 创建符号链接到项目根目录
ln -sf build/compile_commands.json ../compile_commands.json

# 检查编译数据库是否生成
ls -la ../compile_commands.json
head -5 ../compile_commands.json

# 2. 配置clangd（代码补全）
mkdir -p ~/.config/clangd
cat > ~/.config/clangd/config.yaml << EOF
CompileFlags:
  Add:
    - -I${llvm_install_dir}/include
    - -I${workspace}/include
    - -std=c++17
  CompilationDatabase: .
Diagnostics:
  Suppress: [unknown-warning-option, unused-argument]
EOF

# 3. 创建VSCode工作区配置
mkdir -p "${workspace}/.vscode"

cat > "${workspace}/.vscode/settings.json" << EOF
{
    "C_Cpp.default.compilerPath": "${llvm_install_dir}/bin/clang++",
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.includePath": [
        "${llvm_install_dir}/include",
        "${workspace}/include",
        "\${workspaceFolder}/include"
    ],
    "clangd.path": "${llvm_install_dir}/bin/clangd",
    "clangd.arguments": [
        "--compile-commands-dir=\${workspaceFolder}/build",
        "--query-driver=${llvm_install_dir}/bin/clang++",
        "--header-insertion=never"
    ],
    "cmake.configureArgs": [
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ],
    "editor.formatOnSave": true,
    "C_Cpp.formatting": "clangFormat",
    "C_Cpp.clang_format_path": "${llvm_install_dir}/bin/clang-format"
}
EOF

# 4. 创建.clang-format文件（代码格式化）
cat > "${workspace}/.clang-format" << EOF
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
echo "配置完成！请执行以下步骤："
echo "1. 在VSCode中重新加载窗口或重启VSCode"
echo "2. 确保已安装以下VSCode扩展："
echo "   - C/C++ (Microsoft)"
echo "   - clangd (LLVM)"
echo "   - CMake Tools"
echo "3. 检查clangd是否正常工作"
