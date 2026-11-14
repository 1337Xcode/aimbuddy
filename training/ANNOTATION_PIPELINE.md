# 📊 AimBuddy Data Annotation Pipeline
## خط أنابيب التوسيم الكامل (YOLO26n-Pose)

---

## 🎯 نظرة عامة

هذا البرنامج يوفر **نظام معالجة بيانات متكامل** لمشروع AimBuddy:

```
الصور الخام (RAW) → تنظيف وفلترة → توسيم يدوي → فحص جودة → تقسيم البيانات → تصدير نهائي
```

---

## 📁 الهيكل الكامل

```
training/
├── dataset/
│   ├── raw/                          # البيانات الخام
│   │   └── screenshots/
│   │       ├── valorant/
│   │       ├── csgo/
│   │       └── other_games/
│   │
│   ├── processed/                    # بعد التنظيف
│   │   ├── filtered/
│   │   │   ├── images/               # صور نظيفة
│   │   │   ├── removed_duplicates/
│   │   │   └── removed_blur/
│   │   └── resized/
│   │       ├── 640x640/
│   │       └── 1024x1024/
│   │
│   ├── annotated/                    # بعد التوسيم
│   │   ├── images/                   # الصور الموسومة
│   │   ├── labels/                   # ملفات .txt (YOLO format)
│   │   ├── annotations/
│   │   │   ├── keypoints_schema.json
│   │   │   ├── annotation_guidelines.md
│   │   │   └── annotation_stats.json
│   │   └── qa_samples/
│   │
│   ├── splits/                       # التقسيم النهائي
│   │   ├── train/ (70%)
│   │   ├── val/   (20%)
│   │   ├── test/  (10%)
│   │   ├── dataset.yaml
│   │   └── split_report.json
│   │
│   └── final_export/                 # التصدير النهائي
│
├── qa_reports/                       # تقارير الجودة
│   ├── qa_report_*.json
│   └── problematic_samples/
│
└── logs/                             # السجلات
    ├── cleaning_log_*.log
    └── annotation_log_*.log
```

---

## 🚀 البدء السريع

### الخطوة 0️⃣: المتطلبات
```bash
pip install -r requirements.txt
# متطلبات إضافية:
pip install opencv-python scikit-image ultralytics
```

### الخطوة 1️⃣: وضع الصور الخام
ضع الصور في:
```
training/dataset/raw/screenshots/valorant/
training/dataset/raw/screenshots/csgo/
```

### الخطوة 2️⃣: تشغيل Pipeline الكامل

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

## 📋 المراحل التفصيلية

### المرحلة 1️⃣: Data Cleaner Agent (تنظيف البيانات)

**المهام:**
- ❌ حذف الصور المتكررة (تشابه > 95%)
- ❌ استبعاد الصور الضبابية (Motion Blur)
- ✅ تغيير الحجم إلى 640x640 مع padding

**الاستخدام المباشر:**
```bash
python src/data_cleaner.py \
    --input dataset/raw/screenshots \
    --output dataset/processed/filtered \
    --size 640 \
    --blur-threshold 100.0 \
    --similarity 0.95
```

**المخرجات:**
- ✓ `dataset/processed/filtered/images/` - صور نظيفة
- ✓ `dataset/processed/filtered/removed_duplicates/` - صور مكررة
- ✓ `dataset/processed/filtered/removed_blur/` - صور ضبابية
- ✓ `dataset/processed/filtered/cleaning_report.json` - تقرير مفصل

**معاملات إضافية:**
```bash
--blur-threshold    # حد الضبابية (أقل = ضبابية أكثر)
--similarity        # حد التشابه (0.0-1.0)
```

---

### المرحلة 2️⃣: Annotation Agent (التوسيم اليدوي)

**أدوات موصى بها:**
- [CVAT](https://cvat.ai) - مفتوح المصدر، احترافي
- [Roboflow](https://roboflow.com) - واجهة سهلة، تصدير سريع
- [LabelImg](https://github.com/heartexlabs/labelImg) - خفيف الوزن

**قواعد التوسيم:**

#### الصندوق المحيط (Bounding Box)
```
- ✅ يشمل الرأس حتى القدمين
- ✅ بدون فراغات كبيرة خارج الجسم
- ✅ لا يدخل في الخلفية
```

#### النقاط المفصلية (Keypoints)
```
ID | الاسم          | الوصف
---|----------------|------------------------
0  | head_center    | مركز الرأس (CRITICAL) 🎯
1  | left_shoulder  | الكتف الأيسر
2  | right_shoulder | الكتف الأيمن
3  | torso_center   | مركز الجذع
4  | weapon         | موقع السلاح
```

#### الحجب (Occlusion)
- `visibility = 1` → مرئية
- `visibility = 0` → مخفية خلف جدار/كائن

**صيغة YOLO-Pose (.txt):**
```
<class> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_vis> ... <kpN_x> <kpN_y> <kpN_vis>
```

**مثال:**
```
0 0.45 0.50 0.35 0.60 0.45 0.25 1 0.35 0.45 1 0.55 0.45 1 0.40 0.70 1 0.50 0.20 0.8
```

**الملفات المطلوبة:**
```
dataset/annotated/
├── images/
│   ├── screenshot_001.jpg
│   ├── screenshot_002.jpg
│   └── ...
└── labels/
    ├── screenshot_001.txt
    ├── screenshot_002.txt
    └── ...
```

---

### المرحلة 3️⃣: QA & Export Agent (فحص الجودة)

**المهام:**
- ✓ فحص عشوائي (10% من البيانات)
- ✓ التحقق من صحة التوسيم
- ✓ رسم بصري للتوسيم للمراجعة

**الاستخدام:**
```bash
python src/qa_export_agent.py \
    --annotated dataset/annotated \
    --qa-dir qa_reports \
    --sample-size 0.1
```

**فحوصات الجودة:**
- ✓ صحة الـ Bounding Box
- ✓ صحة النقاط المفصلية
- ✓ قيم Visibility صحيحة
- ✓ عدم الخروج عن حدود الصورة

**المخرجات:**
- ✓ `qa_reports/qa_report_*.json` - تقرير مفصل
- ✓ `qa_reports/problematic_samples/` - صور بمشاكل

---

### المرحلة 4️⃣: Dataset Splitter (تقسيم البيانات)

**التوزيع:**
```
70% تدريب (Train)
20% تحقق (Val)
10% اختبار (Test)
```

**الاستخدام:**
```bash
python src/dataset_splitter.py \
    --annotated dataset/annotated \
    --splits dataset/splits \
    --seed 42
```

**المخرجات:**
```
dataset/splits/
├── train/
│   ├── images/ (70%)
│   └── labels/
├── val/
│   ├── images/ (20%)
│   └── labels/
├── test/
│   ├── images/ (10%)
│   └── labels/
├── dataset.yaml        # ✨ ملف الإعدادات لـ YOLOv8
└── split_report.json
```

---

## 📊 ملف YAML النهائي

**مثال: `dataset/splits/dataset.yaml`**
```yaml
path: /path/to/dataset/splits
train: train/images
val: val/images
test: test/images

nc: 1                    # عدد الفئات (Enemy فقط)
names: ['enemy']         # أسماء الفئات

# النقاط المفصلية
kpt_shape: [5, 3]        # 5 نقاط × (x, y, visibility)
kpt_names: ['head_center', 'left_shoulder', 'right_shoulder', 'torso_center', 'weapon_position']
```

---

## 🎓 نصائح وأفضل الممارسات

### 1️⃣ جودة البيانات أولاً
> من الأفضل توسيم 100 صورة **بشكل مثالي** بدلاً من 1000 صورة **بشكل رديء**

### 2️⃣ التركيز على Head Center
- هذه النقطة **الأهم** للـ Aimbot
- اجعلها **Pixel-perfect** (± 2-3 pixels)

### 3️⃣ الاتساق
- استخدم نفس الأسلوب عبر جميع الصور
- راجع قواعد التوسيم بشكل منتظم

### 4️⃣ التعامل مع HUD Overlap
- إذا كانت خريطة مصغرة تغطي الخصم: ضع `visibility = 0`
- إذا كان الجسم 80%+ مخفي: اطرح الصورة

### 5️⃣ الضبابية والحركة السريعة
- إذا لم تستطع تمييز الرأس بوضوح: اطرح الصورة
- Data Cleaner سيساعد بشكل تلقائي

---

## 📈 المقاييس المتوقعة

| المقياس | المتوقع |
|--------|---------|
| عدد الصور الموسومة | 500-2000 |
| نسبة عدم الاتساق | < 5% |
| معدل الصور السليمة (QA) | > 95% |
| وقت التوسيم لكل صورة | 1-3 دقائق |

---

## 🔧 استكشاف الأخطاء

### الصور لم تُنسخ إلى splits/
```bash
# تحقق من أن ملفات .txt موجودة في annotated/labels/
ls dataset/annotated/labels/
```

### QA يُظهر مشاكل كثيرة
```bash
# افحص الصور المشكلة
ls qa_reports/problematic_samples/
```

### الـ Blur Threshold عالي جداً
```bash
# جرب قيمة أقل
python src/data_cleaner.py --blur-threshold 80.0
```

---

## 📞 الدعم والملاحظات

- 📝 وثائق التوسيم: [`annotation_guidelines.md`](dataset/annotated/annotations/annotation_guidelines.md)
- 📋 مخطط النقاط: [`keypoints_schema.json`](dataset/annotated/annotations/keypoints_schema.json)
- 📊 التقارير: `qa_reports/` و `dataset/processed/filtered/cleaning_report.json`

---

## 🚀 الخطوة التالية: التدريب

بعد اكتمال جميع المراحل:
```bash
python src/train.py --data dataset/splits/dataset.yaml --epochs 180
```

أو استخدم السكربت المدمج:
```bash
scripts/04_train_adaptive.bat
```

---

**تم الإنشاء:** 2026-11-06  
**الإصدار:** 1.0  
**النموذج:** YOLO26n-Pose
