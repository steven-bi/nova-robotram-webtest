@echo off
title Robot Arm Control
cd /d D:\brainstem-master\nova-robotarm

echo 正在关闭旧进程...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Robot Arm Control" >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":8888.*LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)
timeout /t 4 /nobreak >nul

echo 正在启动机械臂控制服务器...
echo 浏览器将自动打开 http://localhost:8888
echo 关闭此窗口即停止服务器
echo.
python joint_gui_web.py
pause
