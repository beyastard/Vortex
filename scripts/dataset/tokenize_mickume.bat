@echo off
REM ============================================================
REM  tokenize_mickume.bat
REM  Tokenizes all 11 mickume datasets into Arrow shards.
REM  Run from the project root: scripts\dataset\tokenize_mickume.bat
REM ============================================================

set TOKENIZER=amd/AMD-Llama-135m
set BLOCK_SIZE=1024
set COMPRESSION=zstd
set VAL_SPLIT=0.05
set SCRIPT=scripts\tokenize_universal.py

echo.
echo ============================================================
echo  Mickume Dataset Batch Tokenizer
echo ============================================================
echo  Tokenizer : %TOKENIZER%
echo  Block size: %BLOCK_SIZE%
echo  Val split : %VAL_SPLIT%
echo.

REM ── 1. alt_fantasy (already dry-run tested) ──────────────────
echo [1/11] mickume/alt_fantasy
python %SCRIPT% ^
    --hf_dataset mickume/alt_fantasy ^
    --output_dir data\arrow\alt_fantasy ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] alt_fantasy failed & goto :error )

REM ── 2. alt_pantheon ──────────────────────────────────────────
echo [2/11] mickume/alt_pantheon
python %SCRIPT% ^
    --hf_dataset mickume/alt_pantheon ^
    --output_dir data\arrow\alt_pantheon ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] alt_pantheon failed & goto :error )

REM ── 3. alt_nsfw ──────────────────────────────────────────────
echo [3/11] mickume/alt_nsfw
python %SCRIPT% ^
    --hf_dataset mickume/alt_nsfw ^
    --output_dir data\arrow\alt_nsfw ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] alt_nsfw failed & goto :error )

REM ── 4. alt_potterverse ───────────────────────────────────────
echo [4/11] mickume/alt_potterverse
python %SCRIPT% ^
    --hf_dataset mickume/alt_potterverse ^
    --output_dir data\arrow\alt_potterverse ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] alt_potterverse failed & goto :error )

REM ── 5. dark_granger ──────────────────────────────────────────
echo [5/11] mickume/dark_granger
python %SCRIPT% ^
    --hf_dataset mickume/dark_granger ^
    --output_dir data\arrow\dark_granger ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] dark_granger failed & goto :error )

REM ── 6. alt_manga ─────────────────────────────────────────────
echo [6/11] mickume/alt_manga
python %SCRIPT% ^
    --hf_dataset mickume/alt_manga ^
    --output_dir data\arrow\alt_manga ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] alt_manga failed & goto :error )

REM ── 7. alt_dnd ───────────────────────────────────────────────
echo [7/11] mickume/alt_dnd
python %SCRIPT% ^
    --hf_dataset mickume/alt_dnd ^
    --output_dir data\arrow\alt_dnd ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] alt_dnd failed & goto :error )

REM ── 8. dnd_drow ──────────────────────────────────────────────
echo [8/11] mickume/dnd_drow
python %SCRIPT% ^
    --hf_dataset mickume/dnd_drow ^
    --output_dir data\arrow\dnd_drow ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] dnd_drow failed & goto :error )

REM ── 9. wow ───────────────────────────────────────────────────
echo [9/11] mickume/wow
python %SCRIPT% ^
    --hf_dataset mickume/wow ^
    --output_dir data\arrow\wow ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] wow failed & goto :error )

REM ── 10. alt_tentacles ────────────────────────────────────────
echo [10/11] mickume/alt_tentacles
python %SCRIPT% ^
    --hf_dataset mickume/alt_tentacles ^
    --output_dir data\arrow\alt_tentacles ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] alt_tentacles failed & goto :error )

REM ── 11. harry_potter_tiny ────────────────────────────────────
echo [11/11] mickume/harry_potter_tiny
python %SCRIPT% ^
    --hf_dataset mickume/harry_potter_tiny ^
    --output_dir data\arrow\harry_potter_tiny ^
    --tokenizer %TOKENIZER% ^
    --block_size %BLOCK_SIZE% ^
    --compression %COMPRESSION% ^
    --val_split %VAL_SPLIT%
if %errorlevel% neq 0 ( echo [ERROR] harry_potter_tiny failed & goto :error )

REM ── Generate combined manifest ────────────────────────────────
echo.
echo ============================================================
echo  Generating combined manifest...
echo ============================================================
python scripts\dataset\combine_manifests.py ^
    --input_dirs ^
        data\arrow\alt_fantasy ^
        data\arrow\alt_pantheon ^
        data\arrow\alt_nsfw ^
        data\arrow\alt_potterverse ^
        data\arrow\dark_granger ^
        data\arrow\alt_manga ^
        data\arrow\alt_dnd ^
        data\arrow\dnd_drow ^
        data\arrow\wow ^
        data\arrow\alt_tentacles ^
        data\arrow\harry_potter_tiny ^
    --output_dir data\arrow\mickume_combined

echo.
echo ============================================================
echo  All datasets tokenized successfully.
echo  Train with: python scripts\train.py --data_format arrow
echo              --arrow_manifest data\arrow\mickume_combined\manifest.json
echo ============================================================
goto :eof

:error
echo.
echo [BATCH FAILED] Check the error above.
exit /b 1
