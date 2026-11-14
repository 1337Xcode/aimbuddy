📋 AimBuddy Data Annotation Workflow Checklist
=====================================================

## 🎯 المرحلة 1: تحضير البيانات الخام
- [ ] جمع لقطات من Valorant/CS:GO
- [ ] تنظيم الصور في:
  ```
  dataset/raw/screenshots/
  ├── valorant/
  ├── csgo/
  └── other_games/
  ```
- [ ] تشغيل Data Cleaner:
  ```bash
  python src/data_cleaner.py --input dataset/raw/screenshots --output dataset/processed/filtered
  ```
- [ ] مراجعة التقرير: `dataset/processed/filtered/cleaning_report.json`
- [ ] التحقق من:
  - ✓ عدد الصور المحفوظة
  - ✓ الصور المكررة المحذوفة
  - ✓ الصور الضبابية المحذوفة

---

## 🎯 المرحلة 2: التوسيم اليدوي (Annotation)

### الخطوة 1: اختيار أداة التوسيم
- [ ] **خيار A - CVAT (مفتوح المصدر)**
  ```
  1. اذهب إلى https://cvat.ai أو استضف محلياً
  2. أنشئ مشروع جديد
  3. حمّل الصور من: dataset/processed/filtered/images/
  4. اختر تنسيق التصدير: YOLO 1.1
  ```

- [ ] **خيار B - Roboflow**
  ```
  1. اذهب إلى https://roboflow.com
  2. أنشئ مشروع جديد
  3. اختر نوع المشروع: Instance Segmentation (أو Pose Estimation)
  4. حمّل الصور والوسّم
  5. صدّر بصيغة YOLO-Pose
  ```

### الخطوة 2: قواعد التوسيم
قبل البدء، اقرأ: `dataset/annotated/annotations/annotation_guidelines.md`

- [ ] فهمت قواعد رسم الـ Bounding Box
- [ ] فهمت مواقع النقاط المفصلية الخمس:
  - [ ] Head Center (ID: 0) 🎯 **الأهم**
  - [ ] Left Shoulder (ID: 1)
  - [ ] Right Shoulder (ID: 2)
  - [ ] Torso Center (ID: 3)
  - [ ] Weapon Position (ID: 4)
- [ ] فهمت قيم Visibility (0 = مخفي، 1 = مرئي)

### الخطوة 3: التوسيم الفعلي
- [ ] وسّمت **الحد الأدنى 50 صورة** للاختبار الأولي
- [ ] التحقق من الاتساق عبر الصور
- [ ] جودة رسم الصناديق (± 5-10 pixels)
- [ ] دقة النقاط المفصلية (± 2-3 pixels خاصة Head)

### الخطوة 4: التصدير
بعد اكتمال التوسيم:
- [ ] صدّر الملفات بصيغة YOLO-Pose
- [ ] تأكد من التنسيق:
  ```
  class_id x_center y_center width height kp1_x kp1_y kp1_vis ... kpN_x kpN_y kpN_vis
  ```
- [ ] انسخ الملفات إلى المجلدات:
  ```
  dataset/annotated/images/     → جميع الصور
  dataset/annotated/labels/     → ملفات .txt المقابلة
  ```
- [ ] تحقق من أن عدد الصور = عدد ملفات .txt

---

## 🎯 المرحلة 3: فحص الجودة (QA)
- [ ] تشغيل QA Agent:
  ```bash
  python src/qa_export_agent.py --annotated dataset/annotated --qa-dir qa_reports
  ```
- [ ] مراجعة التقرير: `qa_reports/qa_report_*.json`
- [ ] مراجعة الصور المشكلة: `qa_reports/problematic_samples/`
- [ ] معدل الصور السليمة **> 95%**

### إذا وُجدت مشاكل:
- [ ] صحح التوسيم اليدوي للصور المشكلة
- [ ] أعد تشغيل QA Agent
- [ ] كرر حتى تصل إلى > 95% صور سليمة

---

## 🎯 المرحلة 4: تقسيم البيانات
- [ ] تشغيل Dataset Splitter:
  ```bash
  python src/dataset_splitter.py --annotated dataset/annotated --splits dataset/splits
  ```
- [ ] التحقق من:
  - [ ] عدد صور Train: ~70%
  - [ ] عدد صور Val: ~20%
  - [ ] عدد صور Test: ~10%
- [ ] تأكد من وجود: `dataset/splits/dataset.yaml`

---

## 🎯 المرحلة 5: الاختبار المبدئي
- [ ] اختبر أن مجموعة البيانات جاهزة:
  ```bash
  python src/check_model_contract.py --data dataset/splits/dataset.yaml
  ```
- [ ] تحقق من عدم وجود أخطاء

---

## 📊 الإحصائيات المتوقعة

بعد اكتمال جميع المراحل:

| المقياس | المتوقع |
|--------|---------|
| **عدد الصور الإجمالي** | 500-2000 |
| **عدد صور Train** | 350-1400 |
| **عدد صور Val** | 100-400 |
| **عدد صور Test** | 50-200 |
| **معدل الصور السليمة** | > 95% |
| **معدل الاتساق** | > 95% |

---

## 🚀 الخطوة التالية: التدريب

بعد اكتمال جميع المراحل:

```bash
# تدريب تلقائي (يختار عدد الـ epochs تلقائياً)
scripts/04_train_adaptive.bat

# أو تدريب يدوي
python src/train.py \
    --data dataset/splits/dataset.yaml \
    --epochs 180 \
    --imgsz 640 \
    --batch 16
```

---

## ⚠️ الأخطاء الشائعة

### ❌ الخطأ: "No such file or directory: dataset/annotated/labels"
**الحل:** تأكد من أن ملفات .txt موجودة في المجلد
```bash
ls dataset/annotated/labels/
```

### ❌ الخطأ: "Mismatch between images and labels"
**الحل:** عدد الصور يجب أن يساوي عدد ملفات .txt
```bash
ls dataset/annotated/images/ | wc -l
ls dataset/annotated/labels/ | wc -l
```

### ❌ الخطأ: "Invalid keypoint format"
**الحل:** تأكد من أن كل سطر في .txt يحتوي على:
- class_id: 0
- bbox: 4 قيم (normalized)
- keypoints: 5 نقاط × 3 قيم (x, y, visibility) = 15 قيمة

المجموع: 1 + 4 + 15 = 20 قيمة

---

## 📞 الملفات المرجعية

- 📖 دليل التوسيم: `dataset/annotated/annotations/annotation_guidelines.md`
- 📋 مخطط النقاط: `dataset/annotated/annotations/keypoints_schema.json`
- 📊 إحصائيات: `dataset/annotated/annotations/annotation_stats.json`
- 📝 البرنامج الكامل: `ANNOTATION_PIPELINE.md`

---

**تم الإنشاء:** 2026-11-06  
**الحالة:** جاهز للاستخدام
