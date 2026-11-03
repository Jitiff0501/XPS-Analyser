@echo off
chcp 65001 >nul
REM 在 Windows 上双击本文件，生成 dist\Avantage2XPSPEAK.exe（只需本机临时有 Python）
cd /d %~dp0
python -m pip install --upgrade pyinstaller pandas openpyxl xlrd || goto :err
python -m PyInstaller --noconfirm --onefile --windowed --name Avantage2XPSPEAK ^
  --exclude-module matplotlib --exclude-module scipy --exclude-module IPython ^
  avantage_to_xpspeak.py || goto :err
echo.
echo 完成：dist\Avantage2XPSPEAK.exe
explorer dist
pause
exit /b 0
:err
echo 构建失败，请检查是否已安装 Python 3 并加入 PATH。
pause
exit /b 1
