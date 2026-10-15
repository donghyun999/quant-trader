@echo off
REM 실전 자동매매 1회 실행. 기본은 dry-run(주문 전송 안 함).
REM 실제 주문을 보내려면: run_live.bat --execute
cd /d %~dp0
python cli.py live run %*
