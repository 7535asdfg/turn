@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==========================================
echo   Pic2Mesh - 照片转 3D
echo   正在启动本地服务: http://localhost:8000
echo   关闭本窗口即可停止服务
echo ==========================================

start "Pic2Mesh Server" cmd /c "python -m http.server 8000 --bind 127.0.0.1 & pause"
timeout /t 2 >nul
start "" "http://localhost:8000/index.html"