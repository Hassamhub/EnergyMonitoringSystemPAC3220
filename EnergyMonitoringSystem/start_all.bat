@echo off
echo ================================================
echo PAC3220 Energy Monitoring System - STARTING
echo ================================================

echo Starting Backend API Server...
start "PAC3220-Backend" cmd /k "cd /d %~dp0 && python backend/main.py"

timeout /t 3 /nobreak > nul

if /I "%START_POLLER%"=="1" (
  echo Starting Modbus Poller...
  start "PAC3220-Poller" cmd /k "cd /d %~dp0 && cd .. && python app.py"
) else (
  echo Skipping Modbus Poller (set START_POLLER=1 to enable)
)

timeout /t 2 /nobreak > nul

echo Starting Frontend Server...
start "PAC3220-Frontend" cmd /k "cd /d %~dp0 && cd frontend && python -m http.server 3000"

echo.
echo ================================================
echo ALL SERVICES STARTING...
echo.
echo Access URLs:
echo - Admin Dashboard: http://localhost:3000
echo - API Documentation: http://localhost:8000/docs
echo - API Health Check: http://localhost:8000/health
echo.
echo.
echo Close individual windows to stop services.
echo ================================================

pause
