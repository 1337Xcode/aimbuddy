# 🚀 AimBuddy YOLO26n Annotation Pipeline - البدء السريع

## ✅ ما تم إنجازه

```
✓ تم إنشاء الهيكل الكامل لمعالجة البيانات
✓ تم كتابة 3 Agents متخصصة (Cleaner, Annotation, QA)
✓ تم إعداد السكربتات (Windows + Linux/Mac)
✓ تم إنشاء دليل شامل (Guidelines + Checklists)
```

---

## 📁 الهيكل النهائي

```
training/
├── dataset/
│   ├── raw/screenshots/         ← ضع صورك هنا
│   ├── processed/filtered/      ← صور معالجة
│   ├── annotated/               ← صور موسومة
│   ├── splits/                  ← (Train 70% / Val 20% / Test 10%)
│   └── final_export/
├── src/
│   ├── data_cleaner.py          ← Agent 1: فلترة
│   ├── qa_export_agent.py       ← Agent 2: فحص جودة
│   └── dataset_splitter.py      ← Agent 3: تقسيم
├── scripts/
│   ├── 00_annotation_pipeline.bat  ← Pipeline كامل (Windows)
│   └── 00_annotation_pipeline.sh   ← Pipeline كامل (Linux)
├── ANNOTATION_PIPELINE.md       ← دليل شامل
└── WORKFLOW_CHECKLIST.md        ← قائمة مهام
```

---

## 🎯 الخطوات الأساسية

### 1️⃣ تثبيت المتطلبات
```bash
cd training
pip install -r requirements.txt
```

### 2️⃣ وضع الصور الخام
```
training/dataset/raw/screenshots/valorant/  ← صور Valorant
training/dataset/raw/screenshots/csgo/      ← صور CS:GO
```

### 3️⃣ تشغيل Pipeline الكامل
**Windows:**
```batch
cd training\scripts
00_annotation_pipeline.bat
```

**Linux/Mac:**
```bash
cd training/scripts
bash 00_annotation_pipeline.sh
```

---

## 📊 مراحل Pipeline

```
1. تنظيف البيانات (Data Cleaner)
   ├── حذف صور مكررة (>95% تشابه)
   ├── حذف صور ضبابية
   └── تغيير الحجم إلى 640x640
   
2. التوسيم اليدوي (يدويًا)
   ├── استخدم CVAT أو Roboflow
   ├── وسّم Bounding Box + Keypoints
   └── صدّر بصيغة YOLO-Pose
   
3. فحص الجودة (QA Agent)
   ├── فحص عشوائي (10% من البيانات)
   ├── التحقق من صحة التوسيم
   └── تقارير مفصلة
   
4. تقسيم البيانات (Splitter)
   ├── Train: 70%
   ├── Val: 20%
   └── Test: 10%
   
5. تصدير نهائي
   └── ملف YAML جاهز للتدريب
```

---

## 🎓 النقاط المفصلية (Keypoints)

| ID | الاسم | الوصف | الأهمية |
|----|-------|-------|--------|
| 0️⃣ | Head Center | مركز الرأس (Hitbox) | 🔴 **الأهم** |
| 1️⃣ | Left Shoulder | الكتف الأيسر | 🟠 عالية |
| 2️⃣ | Right Shoulder | الكتف الأيمن | 🟠 عالية |
| 3️⃣ | Torso Center | مركز الجذع | 🟡 متوسطة |
| 4️⃣ | Weapon | موقع السلاح | 🟡 متوسطة |

---

## 📝 صيغة الملفات (YOLO-Pose Format)

**ملف واحد = صورة واحدة**

```
dataset/annotated/
├── images/image_001.jpg
├── labels/image_001.txt  ← بنفس الاسم

dataset/annotated/labels/image_001.txt:
0 0.45 0.50 0.35 0.60 0.45 0.25 1 0.35 0.45 1 0.55 0.45 1 0.40 0.70 1 0.50 0.20 0.8
```

**معنى القيم:**
```
0           = class_id (Enemy)
0.45 0.50   = x_center, y_center (normalized 0-1)
0.35 0.60   = width, height (normalized)

Keypoint 1 (Head): 0.45, 0.25, visibility=1
Keypoint 2 (L_Shoulder): 0.35, 0.45, visibility=1
Keypoint 3 (R_Shoulder): 0.55, 0.45, visibility=1
Keypoint 4 (Torso): 0.40, 0.70, visibility=1
Keypoint 5 (Weapon): 0.50, 0.20, visibility=0.8
```

---

## 🔧 الاستخدام المباشر للـ Agents

### Data Cleaner
```bash
python src/data_cleaner.py \
    --input dataset/raw/screenshots \
    --output dataset/processed/filtered \
    --size 640 \
    --blur-threshold 100.0
```

**المخرجات:**
- `dataset/processed/filtered/images/` - صور نظيفة
- `dataset/processed/filtered/cleaning_report.json` - تقرير

---

### QA Agent
```bash
python src/qa_export_agent.py \
    --annotated dataset/annotated \
    --qa-dir qa_reports \
    --sample-size 0.1
```

**المخرجات:**
- `qa_reports/qa_report_*.json` - تقرير جودة
- `qa_reports/problematic_samples/` - صور بمشاكل

---

### Dataset Splitter
```bash
python src/dataset_splitter.py \
    --annotated dataset/annotated \
    --splits dataset/splits
```

**المخرجات:**
- `dataset/splits/train/` - 70%
- `dataset/splits/val/` - 20%
- `dataset/splits/test/` - 10%
- `dataset/splits/dataset.yaml` - ملف الإعدادات

---

## 📚 الملفات المرجعية

```
📖 دليل التوسيم الكامل
└── dataset/annotated/annotations/annotation_guidelines.md

📋 مخطط النقاط المفصلية
└── dataset/annotated/annotations/keypoints_schema.json

📊 إحصائيات التوسيم
└── dataset/annotated/annotations/annotation_stats.json

📝 قائمة المهام
└── WORKFLOW_CHECKLIST.md

📖 البرنامج الكامل
└── ANNOTATION_PIPELINE.md
```

---

## ⏱️ الإجدولة المتوقعة

| المرحلة | الوقت |
|--------|-------|
| تنظيف البيانات | 5-10 دقائق |
| التوسيم اليدوي | 30-60 دقيقة (لـ 50 صورة) |
| فحص الجودة | 2-3 دقائق |
| تقسيم البيانات | < 1 دقيقة |
| **المجموع (لـ 50 صورة)** | **~1-1.5 ساعة** |

---

## 🎯 الهدف النهائي

```
✅ مجموعة بيانات نظيفة وموسومة بدقة
✅ معدل جودة > 95%
✅ ملف YAML جاهز للتدريب
✅ يمكن البدء فوراً بـ:
   python src/train.py --data dataset/splits/dataset.yaml
```

---

## 💡 نصائح مهمة

1. **جودة > كمية**
   - 100 صورة بشكل مثالي > 1000 صورة بشكل رديء

2. **ركّز على Head Center**
   - هذه النقطة الأهم للـ Aimbot
   - اجعلها Pixel-perfect (± 2-3 pixels)

3. **الاتساق**
   - استخدم نفس الأسلوب عبر جميع الصور
   - راجع Guidelines بانتظام

4. **معالجة HUD Overlap**
   - إذا كانت خريطة تغطي الخصم: visibility = 0
   - إذا كان 80%+ مخفي: اطرح الصورة

---

## 🆘 استكشاف الأخطاء

| المشكلة | الحل |
|--------|------|
| "No images found" | تأكد من وجود الصور في `dataset/raw/screenshots/` |
| Mismatch images/labels | عدد صور ≠ عدد ملفات .txt |
| QA issues > 5% | قيّم التوسيم اليدوي وصحّحه |
| Blur threshold عالي | جرب `--blur-threshold 80.0` |

---

## 🚀 الخطوات التالية

**بعد اكتمال جميع المراحل:**

```bash
# اختبار مجموعة البيانات
python src/check_model_contract.py --data dataset/splits/dataset.yaml

# البدء بالتدريب
python src/train.py \
    --data dataset/splits/dataset.yaml \
    --epochs 180 \
    --imgsz 640 \
    --batch 16 \
    --device auto
```

---

## 📞 الدعم

- 📖 دليل شامل: `ANNOTATION_PIPELINE.md`
- ✅ قائمة مهام: `WORKFLOW_CHECKLIST.md`
- 📋 قواعس التوسيم: `dataset/annotated/annotations/annotation_guidelines.md`

---

**تم الإنشاء:** 2026-11-06  
**النموذج:** YOLO26n-Pose  
**الحالة:** 🟢 جاهز للاستخدام الفوري
