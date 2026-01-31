@echo off
REM GTA1 LoRA 训练启动脚本
REM ========================================

echo 正在启动GTA1 LoRA训练...
echo.

REM 检查Python环境
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 显示当前Python版本
echo Python版本:
python --version
echo.

REM 检查CUDA
echo 检查CUDA可用性...
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU数量: {torch.cuda.device_count()}')"
echo.

REM 设置环境变量
set PYTHONPATH=%PYTHONPATH%;%CD%\src

REM 检查配置文件
if not exist "configs\lora_config.yaml" (
    echo 错误: 未找到配置文件 configs\lora_config.yaml
    pause
    exit /b 1
)

REM 检查数据文件
if not exist "preprocessing\inp.json" (
    echo 错误: 未找到训练数据文件 preprocessing\inp.json
    pause
    exit /b 1
)

echo 开始训练...
echo ========================================
echo 模型: Qwen/Qwen2.5-VL-3B-Instruct
echo 配置: configs\lora_config.yaml
echo 输出目录: output\gta1_lora
echo ========================================
echo.

REM 运行训练
python src\grpo_grounding.py --config configs\lora_config.yaml

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 训练完成！
    echo 模型已保存至: output\gta1_lora
    echo ========================================
) else (
    echo.
    echo 训练过程中出现错误，错误代码: %ERRORLEVEL%
)

echo.
pause
