@echo off
setlocal
rem =====================================================
rem AnythingPixie -- 別フォルダ起動ランチャー
rem =====================================================
rem このbatを対象プロジェクトのフォルダに置いてダブルクリックするか、
rem 対象フォルダを このbat にドラッグ＆ドロップするか、
rem   StartPixieHere.bat "D:\Work\ProjectB"
rem のようにパスを引数で渡すと、そのフォルダを作業対象として Pixie を起動します。
rem 引数もドロップも無い場合は、このbatを実行したフォルダ（カレント）が対象になります。
rem
rem Pixie 本体の場所は固定（下記 PIXIE_ROOT）。このbatは自由にコピーして使えます。

set "PIXIE_ROOT=D:\Workspace\AnythingWithPixie"

rem 引数（ドラッグ＆ドロップ含む）があればそのフォルダへ移動。無ければカレントのまま。
if not "%~1"=="" (
    if not exist "%~1\" (
        echo [Error] 指定したフォルダが見つかりません: %~1
        pause
        exit /b 1
    )
    cd /d "%~1"
)

echo [System] 作業対象フォルダ: "%CD%"
"%PIXIE_ROOT%\.venv\Scripts\python.exe" "%PIXIE_ROOT%\src\main.py"

endlocal
pause
