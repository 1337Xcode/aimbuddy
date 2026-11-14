@echo off
REM Pipeline كامل لمعالجة البيانات من الخام إلى التصدير النهائي
REM العمل مع YOLO26n-Pose

setlocal enabledelayedexpansion

echo.
echo ================================
echo 🚀 AimBuddy Data Annotation Pipeline
echo ================================
echo.

REM تحديد المسارات
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set DATASET_DIR=%PROJECT_ROOT%\dataset
set RAW_SCREENSHOTS=%DATASET_DIR%\raw\screenshots
set PROCESSED_FILTERED=%DATASET_DIR%\processed\filtered
set ANNOTATED_DIR=%DATASET_DIR%\annotated
set SPLITS_DIR=%DATASET_DIR%\splits
set QA_DIR=%PROJECT_ROOT%\qa_reports

echo 📁 مسارات المشروع:
echo   - مجلد المشروع: %PROJECT_ROOT%
echo   - مجلد البيانات: %DATASET_DIR%
echo   - الصور الخام: %RAW_SCREENSHOTS%
echo.

REM المرحلة 1: فلترة البيانات
echo.
echo 📋 المرحلة 1: فلترة ومعالجة البيانات الخام
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if exist "%RAW_SCREENSHOTS%\*.jpg" (
    echo ✅ الصور الخام موجودة
    echo.
    echo 🔧 تشغيل Data Cleaner Agent...
    python "%SCRIPT_DIR%..\src\data_cleaner.py" ^
        --input "%RAW_SCREENSHOTS%" ^
        --output "%PROCESSED_FILTERED%" ^
        --size 640 ^
        --blur-threshold 100.0 ^
        --similarity 0.95
    echo.
) else (
    echo ⚠️ لم يتم العثور على صور خام!
    echo 📍 يرجى وضع الصور في: %RAW_SCREENSHOTS%
    echo.
    echo يمكنك المتابعة إذا كانت الصور الموسومة جاهزة في:
    echo    %ANNOTATED_DIR%\images
    echo.
)

REM المرحلة 2: التوسيم (يدوي عبر CVAT أو Roboflow)
echo.
echo 📋 المرحلة 2: التوسيم اليدوي
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ⏸️  يرجى توسيم الصور يدويًا باستخدام:
echo    - CVAT (https://cvat.ai)
echo    - Roboflow (https://roboflow.com)
echo.
echo 📝 قواعد التوسيم متوفرة هنا:
echo    %ANNOTATED_DIR%\annotations\annotation_guidelines.md
echo.
echo 💾 الملفات يجب أن تكون بصيغة YOLO-Pose:
echo    - الصور: %ANNOTATED_DIR%\images\*.jpg
echo    - التسميات: %ANNOTATED_DIR%\labels\*.txt
echo.
pause

REM المرحلة 3: فحص جودة التوسيم
echo.
echo 📋 المرحلة 3: فحص جودة التوسيم (QA)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if exist "%ANNOTATED_DIR%\images" (
    echo 🔧 تشغيل QA Export Agent...
    python "%SCRIPT_DIR%..\src\qa_export_agent.py" ^
        --annotated "%ANNOTATED_DIR%" ^
        --qa-dir "%QA_DIR%" ^
        --sample-size 0.1
    echo.
) else (
    echo ❌ لم يتم العثور على صور موسومة!
    exit /b 1
)

REM المرحلة 4: تقسيم البيانات
echo.
echo 📋 المرحلة 4: تقسيم البيانات (70/20/10)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🔧 تشغيل Dataset Splitter Agent...
python "%SCRIPT_DIR%..\src\dataset_splitter.py" ^
    --annotated "%ANNOTATED_DIR%" ^
    --splits "%SPLITS_DIR%" ^
    --seed 42
echo.

REM الملخص النهائي
echo.
echo ════════════════════════════════════════════════════
echo ✅ اكتملت معالجة البيانات بنجاح!
echo ════════════════════════════════════════════════════
echo.
echo 📊 الملفات النهائية:
echo   ✓ البيانات الموسومة: %ANNOTATED_DIR%
echo   ✓ التقسيمات (Train/Val/Test): %SPLITS_DIR%
echo   ✓ ملف الإعدادات: %SPLITS_DIR%\dataset.yaml
echo   ✓ تقارير الجودة: %QA_DIR%
echo.
echo 🚀 الخطوة التالية: البدء بالتدريب
echo    python "%SCRIPT_DIR%..\src\train.py" --data "%SPLITS_DIR%\dataset.yaml"
echo.

pause
