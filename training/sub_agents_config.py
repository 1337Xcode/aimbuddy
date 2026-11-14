"""
AimBuddy Sub-Agents Configuration
نظام توزيع المهام على وكلاء متخصصين
"""

# ============================================
# AGENT 1: Data Cleaner Agent (تنظيف البيانات)
# ============================================

DATA_CLEANER_CONFIG = {
    "name": "Data Cleaner Agent",
    "role": "معالج البيانات الخام",
    "description": "فلترة ومعالجة الصور الخام",
    "responsibilities": [
        "حذف الصور المتكررة (تشابه > 95%)",
        "استبعاد الصور الضبابية (Motion Blur)",
        "توحيد أحجام الصور (640x640 مع Padding)",
    ],
    "input": {
        "raw_screenshots_dir": "dataset/raw/screenshots/",
        "filter_config": {
            "target_size": 640,
            "blur_threshold": 100.0,  # أقل = ضبابية أكثر
            "similarity_threshold": 0.95,
        }
    },
    "output": {
        "clean_images_dir": "dataset/processed/filtered/images/",
        "removed_duplicates_dir": "dataset/processed/filtered/removed_duplicates/",
        "removed_blur_dir": "dataset/processed/filtered/removed_blur/",
        "cleaning_report": "dataset/processed/filtered/cleaning_report.json",
    },
    "script": "src/data_cleaner.py",
    "command_template": """
    python src/data_cleaner.py \
        --input {input_dir} \
        --output {output_dir} \
        --size {target_size} \
        --blur-threshold {blur_threshold} \
        --similarity {similarity_threshold}
    """,
    "success_metrics": {
        "min_clean_images_percentage": 70,  # على الأقل 70% من الصور الأصلية
        "max_processing_errors": 5,
    }
}

# ============================================
# AGENT 2: Annotation Agent (التوسيم اليدوي)
# ============================================

ANNOTATION_AGENT_CONFIG = {
    "name": "Annotation Agent",
    "role": "موسّم البيانات المتخصص",
    "description": "توسيم الصور يدويًا بصيغة YOLO-Pose",
    "responsibilities": [
        "رسم Bounding Boxes دقيقة حول أجسام الأعداء",
        "تحديد 5 نقاط مفصلية (Keypoints)",
        "تحديد قيم Visibility (0/1) للنقاط المخفية",
    ],
    "keypoints_to_annotate": [
        {"id": 0, "name": "head_center", "critical": True, "description": "مركز الرأس (Hitbox)"},
        {"id": 1, "name": "left_shoulder", "critical": False, "description": "الكتف الأيسر"},
        {"id": 2, "name": "right_shoulder", "critical": False, "description": "الكتف الأيمن"},
        {"id": 3, "name": "torso_center", "critical": False, "description": "مركز الجذع"},
        {"id": 4, "name": "weapon_position", "critical": False, "description": "موقع السلاح"},
    ],
    "input": {
        "clean_images_dir": "dataset/processed/filtered/images/",
        "annotation_tool": "CVAT or Roboflow",
        "guidelines": "dataset/annotated/annotations/annotation_guidelines.md",
    },
    "output": {
        "annotated_images_dir": "dataset/annotated/images/",
        "labels_dir": "dataset/annotated/labels/",  # .txt files
        "annotation_stats": "dataset/annotated/annotations/annotation_stats.json",
    },
    "format": "YOLO-Pose",
    "format_example": "0 0.45 0.50 0.35 0.60 0.45 0.25 1 0.35 0.45 1 0.55 0.45 1 0.40 0.70 1 0.50 0.20 0.8",
    "quality_requirements": {
        "bbox_accuracy": "± 5-10 pixels",
        "keypoint_accuracy": "± 2-3 pixels (especially head_center)",
        "consistency": "نفس الأسلوب عبر جميع الصور",
        "min_quality_score": 95,  # من 100
    },
    "tools": [
        {
            "name": "CVAT",
            "url": "https://cvat.ai",
            "pros": ["مفتوح المصدر", "دعم كامل للـ Pose", "تصدير YOLO"],
            "cons": ["تثبيت محلي قد يكون معقداً"],
        },
        {
            "name": "Roboflow",
            "url": "https://roboflow.com",
            "pros": ["واجهة سهلة", "تصدير سريع", "دعم Pose Estimation"],
            "cons": ["الخطة المجانية محدودة"],
        },
        {
            "name": "LabelImg",
            "url": "https://github.com/heartexlabs/labelImg",
            "pros": ["خفيف الوزن", "لا يحتاج إنترنت"],
            "cons": ["دعم محدود للـ Pose"],
        }
    ],
}

# ============================================
# AGENT 3: QA & Export Agent (فحص الجودة)
# ============================================

QA_EXPORT_AGENT_CONFIG = {
    "name": "QA & Export Agent",
    "role": "ضامن جودة البيانات والتصدير",
    "description": "فحص جودة التوسيم والتصدير النهائي",
    "responsibilities": [
        "فحص عشوائي (10% من البيانات)",
        "التحقق من صحة الـ Bounding Boxes",
        "التحقق من صحة Keypoints والـ Visibility",
        "توليد تقارير مفصلة",
        "تقسيم البيانات (70/20/10)",
        "إنشاء ملف YAML النهائي",
    ],
    "input": {
        "annotated_dir": "dataset/annotated/",
        "sample_size": 0.1,  # 10% للفحص
    },
    "qa_checks": [
        {
            "name": "BBox Validity",
            "checks": ["موجود داخل الصورة", "حجم موجب", "غير فارغ"],
        },
        {
            "name": "Keypoint Validity",
            "checks": ["موجود داخل الصورة", "عدد صحيح", "قيم x,y صحيحة"],
        },
        {
            "name": "Visibility Correctness",
            "checks": ["قيمة 0 أو 1 فقط", "متسقة عبر الصور"],
        },
        {
            "name": "Image-Label Matching",
            "checks": ["كل صورة لها ملف .txt مقابل", "الأسماء متطابقة"],
        }
    ],
    "output": {
        "qa_report": "qa_reports/qa_report_*.json",
        "problematic_samples": "qa_reports/problematic_samples/",
        "splits_train": "dataset/splits/train/",
        "splits_val": "dataset/splits/val/",
        "splits_test": "dataset/splits/test/",
        "dataset_yaml": "dataset/splits/dataset.yaml",
    },
    "scripts": [
        "src/qa_export_agent.py",
        "src/dataset_splitter.py",
    ],
    "success_metrics": {
        "min_valid_annotations": 95,  # %
        "max_qa_issues": 5,  # %
    }
}

# ============================================
# PIPELINE ORCHESTRATOR (منسق خط الأنابيب)
# ============================================

PIPELINE_CONFIG = {
    "name": "AimBuddy Data Annotation Pipeline",
    "version": "1.0",
    "target_model": "YOLO26n-Pose",
    "stages": [
        {
            "stage": 1,
            "name": "Data Cleaning",
            "agent": DATA_CLEANER_CONFIG,
            "dependencies": [],
            "estimated_time": "5-10 minutes",
        },
        {
            "stage": 2,
            "name": "Manual Annotation",
            "agent": ANNOTATION_AGENT_CONFIG,
            "dependencies": ["stage_1"],
            "estimated_time": "30-60 minutes (for 50 images)",
            "manual": True,  # يتطلب تدخل يدوي
        },
        {
            "stage": 3,
            "name": "QA & Export",
            "agent": QA_EXPORT_AGENT_CONFIG,
            "dependencies": ["stage_2"],
            "estimated_time": "3-5 minutes",
        },
    ],
    "total_estimated_time": "45-80 minutes",
    "quality_gates": [
        {
            "gate": "Stage 1 Output",
            "condition": "min_clean_images_percentage >= 70",
            "action_on_fail": "Review raw images, adjust blur_threshold",
        },
        {
            "gate": "Stage 2 Output",
            "condition": "image_label_match == 100%",
            "action_on_fail": "Ensure all images have corresponding .txt files",
        },
        {
            "gate": "Stage 3 QA",
            "condition": "valid_annotations >= 95%",
            "action_on_fail": "Re-annotate problematic samples",
        },
    ],
}

# ============================================
# QUICK COMMANDS
# ============================================

QUICK_COMMANDS = {
    "clean_data": "python src/data_cleaner.py --input dataset/raw/screenshots --output dataset/processed/filtered",
    "run_qa": "python src/qa_export_agent.py --annotated dataset/annotated --qa-dir qa_reports",
    "split_data": "python src/dataset_splitter.py --annotated dataset/annotated --splits dataset/splits",
    "run_full_pipeline": {
        "windows": "cd scripts && 00_annotation_pipeline.bat",
        "linux": "cd scripts && bash 00_annotation_pipeline.sh",
    },
}

if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("🤖 AimBuddy Sub-Agents Configuration")
    print("=" * 60)
    print()
    
    print("📋 Pipeline Stages:")
    for stage in PIPELINE_CONFIG["stages"]:
        print(f"  {stage['stage']}. {stage['name']} ({stage['estimated_time']})")
    
    print()
    print("🚀 Quick Commands:")
    for cmd_name, cmd in QUICK_COMMANDS.items():
        if isinstance(cmd, dict):
            print(f"  {cmd_name}:")
            for os_type, command in cmd.items():
                print(f"    - {os_type}: {command}")
        else:
            print(f"  {cmd_name}: {cmd}")
    
    print()
    print("=" * 60)
