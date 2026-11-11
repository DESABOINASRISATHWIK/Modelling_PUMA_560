@echo off
cd /d "%~dp0"
if not exist output mkdir output
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
if errorlevel 1 exit /b 1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output main.tex
