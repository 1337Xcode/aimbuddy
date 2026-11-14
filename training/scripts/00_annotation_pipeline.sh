#!/bin/bash
# Pipeline كامل لمعالجة البيانات من الخام إلى التصدير النهائي
# العمل مع YOLO26n-Pose

set -e  # التوقف عند أي خطأ

echo "================================"
echo "🚀 AimBuddy Data Annotation Pipeline"
echo "================================"
echo ""

# الألوان للطباعة
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# المتغيرات
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="$PROJECT_ROOT/dataset"
RAW_SCREENSHOTS="$DATASET_DIR/raw/screenshots"
PROCESSED_FILTERED="$DATASET_DIR/processed/filtered"
ANNOTATED_DIR="$DATASET_DIR/annotated"
SPLITS_DIR="$DATASET_DIR/splits"
QA_DIR="$PROJECT_ROOT/qa_reports"

echo -e "${BLUE}📁 مسارات المشروع:${NC}"
echo "  • مجلد المشروع: $PROJECT_ROOT"
echo "  • مجلد البيانات: $DATASET_DIR"
echo "  • الصور الخام: $RAW_SCREENSHOTS"
echo ""

# المرحلة 1: فلترة البيانات
echo -e "${YELLOW}📋 المرحلة 1: فلترة ومعالجة البيانات الخام${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "$RAW_SCREENSHOTS" ] && [ "$(find "$RAW_SCREENSHOTS" -type f -name '*.jpg' -o -name '*.png' | wc -l)" -gt 0 ]; then
    echo "✅ الصور الخام موجودة"
    echo ""
    echo "🔧 تشغيل Data Cleaner Agent..."
    python3 "$SCRIPT_DIR/data_cleaner.py" \
        --input "$RAW_SCREENSHOTS" \
        --output "$PROCESSED_FILTERED" \
        --size 640 \
        --blur-threshold 100.0 \
        --similarity 0.95
    echo ""
else
    echo -e "${RED}⚠️ لم يتم العثور على صور خام!${NC}"
    echo "📍 يرجى وضع الصور في: $RAW_SCREENSHOTS"
    echo ""
    echo "يمكنك المتابعة إذا كانت الصور الموسومة جاهزة في:"
    echo "   $ANNOTATED_DIR/images"
    echo ""
fi

# المرحلة 2: التوسيم (يدوي عبر CVAT أو Roboflow)
echo -e "${YELLOW}📋 المرحلة 2: التوسيم اليدوي${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏸️  يرجى توسيم الصور يدويًا باستخدام:"
echo "   - CVAT (https://cvat.ai)"
echo "   - Roboflow (https://roboflow.com)"
echo ""
echo "📝 قواعد التوسيم متوفرة هنا:"
echo "   $ANNOTATED_DIR/annotations/annotation_guidelines.md"
echo ""
echo "💾 الملفات يجب أن تكون بصيغة YOLO-Pose:"
echo "   - الصور: $ANNOTATED_DIR/images/*.jpg"
echo "   - التسميات: $ANNOTATED_DIR/labels/*.txt"
echo ""
read -p "اضغط Enter بعد اكتمال التوسيم..."
echo ""

# المرحلة 3: فحص جودة التوسيم
echo -e "${YELLOW}📋 المرحلة 3: فحص جودة التوسيم (QA)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "$ANNOTATED_DIR/images" ] && [ "$(find "$ANNOTATED_DIR/images" -type f | wc -l)" -gt 0 ]; then
    echo "🔧 تشغيل QA Export Agent..."
    python3 "$SCRIPT_DIR/qa_export_agent.py" \
        --annotated "$ANNOTATED_DIR" \
        --qa-dir "$QA_DIR" \
        --sample-size 0.1
    echo ""
else
    echo -e "${RED}❌ لم يتم العثور على صور موسومة!${NC}"
    exit 1
fi

# المرحلة 4: تقسيم البيانات
echo -e "${YELLOW}📋 المرحلة 4: تقسيم البيانات (70/20/10)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 تشغيل Dataset Splitter Agent..."
python3 "$SCRIPT_DIR/dataset_splitter.py" \
    --annotated "$ANNOTATED_DIR" \
    --splits "$SPLITS_DIR" \
    --seed 42
echo ""

# الملخص النهائي
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ اكتملت معالجة البيانات بنجاح!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""
echo "📊 الملفات النهائية:"
echo "  ✓ البيانات الموسومة: $ANNOTATED_DIR"
echo "  ✓ التقسيمات (Train/Val/Test): $SPLITS_DIR"
echo "  ✓ ملف الإعدادات: $SPLITS_DIR/dataset.yaml"
echo "  ✓ تقارير الجودة: $QA_DIR"
echo ""
echo "🚀 الخطوة التالية: البدء بالتدريب"
echo "   python3 $SCRIPT_DIR/train.py --data $SPLITS_DIR/dataset.yaml"
echo ""
