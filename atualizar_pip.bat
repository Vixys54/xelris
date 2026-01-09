@echo off
chcp 65001 > nul
echo.
echo ============================================
echo    ATUALIZANDO PIP NO AMBIENTE VIRTUAL
echo ============================================
echo.

if not exist ".venv" (
    echo ❌ Pasta .venv nao encontrada!
    echo    Execute a Opcao 2 no menu.py primeiro.
    pause
    exit /b 1
)

echo ✓ Ambiente virtual encontrado.
echo.
echo Atualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

if %errorlevel% == 0 (
    echo.
    echo ✅ Pip atualizado com sucesso!
) else (
    echo.
    echo ❌ Erro ao atualizar pip. Verifique sua conexao ou antivirus.
)

echo.
echo Pressione qualquer tecla para fechar...
pause > nul