# ✅ AimBuddy YOLO26n Annotation Pipeline - Setup Complete

## 📌 Summary

تم إنشاء نظام كامل ومتكامل لمعالجة وتوسيم البيانات لنموذج **YOLO26n-Pose** في مشروع AimBuddy.

---

## 🎯 ما تم إنجازه

### 1️⃣ الهيكل الكامل للمشروع
```
training/
├── dataset/
│   ├── raw/screenshots/       ← ضع صورك هنا
│   ├── processed/             ← معالجة
│   ├── annotated/             ← موسوم
│   ├── splits/                ← تقسيم (Train/Val/Test)
│   └── final_export/
├── src/
│   ├── data_cleaner.py
│   ├── qa_export_agent.py
│   └── dataset_splitter.py
└── scripts/
    ├── 00_annotation_pipeline.bat
    └── 00_annotation_pipeline.sh
```

### 2️⃣ ثلاثة Agents متخصصة

| Agent | المهمة | الملف |
|-------|--------|-------|
| **Data Cleaner** | فلترة وتنظيف | `data_cleaner.py` |
| **QA Export** | فحص جودة وتقسيم | `qa_export_agent.py` |
| **Splitter** | تقسيم 70/20/10 | `dataset_splitter.py` |

### 3️⃣ وثائق شاملة

| الملف | الغرض |
|------|--------|
| `QUICK_START.md` | البدء السريع (⭐ اقرأ هذا أولاً) |
| `ANNOTATION_PIPELINE.md` | دليل شامل |
| `WORKFLOW_CHECKLIST.md` | قائمة مهام خطوة بخطوة |
| `sub_agents_config.py` | تكوين الـ Agents |

### 4️⃣ ملفات معايير التوسيم

| الملف | المحتوى |
|------|---------|
| `annotation_guidelines.md` | قواعس التوسيم (🔴 مهم جداً) |
| `keypoints_schema.json` | تعريف النقاط المفصلية |
| `annotation_stats.json` | إحصائيات التوسيم |

---

## 🚀 البدء الفوري

### الخطوة 1: التثبيت
```bash
cd training
pip install -r requirements.txt
```

### الخطوة 2: تحضير الصور
```
ضع صورك في:
training/dataset/raw/screenshots/valorant/
training/dataset/raw/screenshots/csgo/
```

### الخطوة 3: تشغيل Pipeline
**Windows:**
```batch
cd scripts && 00_annotation_pipeline.bat
```

**Linux/Mac:**
```bash
cd scripts && bash 00_annotation_pipeline.sh
```

---

## 📊 مراحل Pipeline

```
🖼️  الصور الخام
    ↓ (Data Cleaner)
✅  صور نظيفة
    ↓ (توسيم يدوي)
📋  صور موسومة
    ↓ (QA Agent)
✔️  صور معتمدة (95%+)
    ↓ (Splitter)
📂  Train/Val/Test (70/20/10)
    ↓
🎓  جاهزة للتدريب!
```

---

## 🎯 النقاط المفصلية

```
0️⃣ Head Center       (مركز الرأس - CRITICAL! 🎯)
1️⃣ Left Shoulder    (الكتف الأيسر)
2️⃣ Right Shoulder   (الكتف الأيمن)
3️⃣ Torso Center     (مركز الجذع)
4️⃣ Weapon Position  (موقع السلاح)
```

**صيغة الملف:**
```
class_id x_center y_center width height kp1_x kp1_y kp1_vis ... kp5_x kp5_y kp5_vis
```

---

## ⚙️ تخصيص المعاملات

### Data Cleaner
```bash
python src/data_cleaner.py \
    --blur-threshold 100.0      # أقل = ضبابية أكثر
    --similarity 0.95           # حد التشابه (0-1)
    --size 640                  # حجم الصورة
```

### QA Agent
```bash
python src/qa_export_agent.py \
    --sample-size 0.1           # نسبة الفحص (10%)
```

### Dataset Splitter
```bash
python src/dataset_splitter.py \
    --seed 42                   # للعشوائية المتكررة
```

---

## 📈 المؤشرات المتوقعة

| المؤشر | القيمة |
|---------|--------|
| نسبة الصور النظيفة | > 70% |
| معدل الصور السليمة (QA) | > 95% |
| عدد صور Train | 350-1400 (70%) |
| عدد صور Val | 100-400 (20%) |
| عدد صور Test | 50-200 (10%) |

---

## 📁 ملفات مهمة

### قراءة أولاً:
```
📖 QUICK_START.md                ← البدء السريع
📖 WORKFLOW_CHECKLIST.md          ← قائمة مهام
📖 annotation_guidelines.md       ← قواعس التوسيم
```

### للمرجعية:
```
📋 ANNOTATION_PIPELINE.md         ← دليل شامل
📋 sub_agents_config.py           ← تكوين Agents
📊 keypoints_schema.json          ← النقاط المفصلية
```

---

## 🔧 أوامر سريعة

```bash
# تنظيف البيانات
python src/data_cleaner.py --input dataset/raw/screenshots --output dataset/processed/filtered

# فحص الجودة
python src/qa_export_agent.py --annotated dataset/annotated --qa-dir qa_reports

# تقسيم البيانات
python src/dataset_splitter.py --annotated dataset/annotated --splits dataset/splits

# Pipeline كامل
python sub_agents_config.py
```

---

## ✨ الخطوات التالية

1. **اقرأ** `QUICK_START.md`
2. **ضع الصور** في `dataset/raw/screenshots/`
3. **شغّل Pipeline** `00_annotation_pipeline.bat` أو `.sh`
4. **وسّم الصور** يدويًا (CVAT/Roboflow)
5. **راجع التقارير** في `qa_reports/`
6. **ابدأ التدريب** بـ `train.py`

---

## 🎓 ملاحظات مهمة

### جودة > كمية
```
✅ 100 صورة مثالية → نموذج أفضل
❌ 1000 صورة سيئة → نموذج ضعيف
```

### Head Center الأهم
```
- ركّز على دقة مركز الرأس
- يجب أن تكون Pixel-perfect (± 2-3 pixels)
- هذه النقطة هي الهدف الأساسي للـ Aimbot
```

### الاتساق حاسم
```
- استخدم نفس الأسلوب عبر جميع الصور
- راجع Guidelines بانتظام
- اطلب رأي ثاني في حالة الشك
```

---

## 📞 الدعم والموارد

| الموضوع | المرجع |
|---------|--------|
| البدء السريع | `QUICK_START.md` |
| قائمة المهام | `WORKFLOW_CHECKLIST.md` |
| قواعس التوسيم | `annotation_guidelines.md` |
| تكوين Agents | `sub_agents_config.py` |
| دليل شامل | `ANNOTATION_PIPELINE.md` |

---

## 🎯 الهدف النهائي

✅ **مجموعة بيانات عالية الجودة**
- تم تنظيفها تلقائياً
- موسومة بدقة يدويًا
- معتمدة من خلال QA
- جاهزة للتدريب فوراً

---

**تم الإنشاء:** 2026-11-06  
**النموذج:** YOLO26n-Pose  
**الإصدار:** 1.0  
**الحالة:** 🟢 جاهز للاستخدام الفوري

**الخطوة التالية:** اقرأ `QUICK_START.md` وابدأ! 🚀
