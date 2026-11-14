#!/usr/bin/env python3
"""
Dataset Split Agent - تقسيم البيانات الموسومة
توزيع: 70% تدريب، 20% تحقق، 10% اختبار
"""

import shutil
import json
from pathlib import Path
from typing import Dict, Tuple
import random
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    def __init__(self, annotated_dir: Path, splits_dir: Path, seed: int = 42):
        self.annotated_dir = Path(annotated_dir)
        self.splits_dir = Path(splits_dir)
        self.seed = seed
        
        random.seed(seed)
        
        # التأكد من وجود المجلدات
        for split in ["train", "val", "test"]:
            (self.splits_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.splits_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def get_image_files(self) -> list:
        """الحصول على جميع ملفات الصور الموسومة"""
        images_dir = self.annotated_dir / "images"
        image_files = sorted(
            list(images_dir.glob("*.jpg")) + 
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpeg"))
        )
        return image_files

    def split_files(self, files: list, train_ratio: float = 0.7, val_ratio: float = 0.2) -> Tuple[list, list, list]:
        """
        تقسيم الملفات عشوائياً
        train_ratio: 70%
        val_ratio: 20%
        test_ratio: 10% (الباقي)
        """
        random.shuffle(files)
        
        total = len(files)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]
        
        return train_files, val_files, test_files

    def copy_files(self, file_list: list, split_name: str):
        """نسخ الصور وملفات التسميات إلى مجلد التقسيم"""
        for img_path in file_list:
            # نسخ الصورة
            dest_img = self.splits_dir / split_name / "images" / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # نسخ ملف التسمية (.txt)
            label_path = self.annotated_dir / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                dest_label = self.splits_dir / split_name / "labels" / label_path.name
                shutil.copy2(label_path, dest_label)
            else:
                logger.warning(f"⚠️ لم يتم العثور على ملف تسمية: {label_path.name}")

    def process(self):
        """تنفيذ التقسيم"""
        logger.info("🔍 جاري جمع ملفات الصور الموسومة...")
        
        image_files = self.get_image_files()
        total_images = len(image_files)
        
        if total_images == 0:
            logger.error("❌ لم يتم العثور على أي صور موسومة!")
            return
        
        logger.info(f"📊 عدد الصور المكتشفة: {total_images}")
        
        # تقسيم الملفات
        train_files, val_files, test_files = self.split_files(image_files)
        
        logger.info("📋 جاري نسخ الملفات...")
        
        # نسخ التقسيمات
        self.copy_files(train_files, "train")
        self.copy_files(val_files, "val")
        self.copy_files(test_files, "test")
        
        # حفظ التقرير
        stats = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "total_images": total_images,
            "splits": {
                "train": {
                    "images": len(train_files),
                    "percentage": f"{(len(train_files)/total_images*100):.1f}%"
                },
                "val": {
                    "images": len(val_files),
                    "percentage": f"{(len(val_files)/total_images*100):.1f}%"
                },
                "test": {
                    "images": len(test_files),
                    "percentage": f"{(len(test_files)/total_images*100):.1f}%"
                }
            }
        }
        
        report_path = self.splits_dir / "split_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "="*50)
        logger.info("✅ اكتملت عملية التقسيم!")
        logger.info("="*50)
        logger.info(f"📚 بيانات التدريب (Train): {len(train_files)} صورة ({stats['splits']['train']['percentage']})")
        logger.info(f"📚 بيانات التحقق (Val): {len(val_files)} صورة ({stats['splits']['val']['percentage']})")
        logger.info(f"📚 بيانات الاختبار (Test): {len(test_files)} صورة ({stats['splits']['test']['percentage']})")
        logger.info("="*50 + "\n")


def create_yaml_config(splits_dir: Path, keypoints_schema: Path):
    """إنشاء ملف YAML لـ YOLOv8"""
    
    with open(keypoints_schema, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    # استخراج أسماء النقاط المفصلية
    kpt_names = [kp["name"] for kp in schema["keypoints"]]
    
    yaml_content = f"""# AimBuddy Dataset Configuration for YOLOv8-Pose

path: {splits_dir.absolute()}
train: train/images
val: val/images
test: test/images

nc: 1  # عدد الفئات (Enemy Player فقط)
names: ['enemy']  # أسماء الفئات

# نقاط مفصلية (Keypoints)
kpt_shape: [{len(kpt_names)}, 3]  # [عدد النقاط، x+y+visibility]
kpt_names: {kpt_names}
"""
    
    yaml_path = splits_dir / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    logger.info(f"✅ ملف YAML محفوظ: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="تقسيم البيانات الموسومة")
    parser.add_argument("--annotated", type=str, default="dataset/annotated",
                       help="مسار مجلد البيانات الموسومة")
    parser.add_argument("--splits", type=str, default="dataset/splits",
                       help="مسار مجلد التقسيمات")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed للتقسيم العشوائي")
    
    args = parser.parse_args()
    
    splitter = DatasetSplitter(Path(args.annotated), Path(args.splits), args.seed)
    splitter.process()
    
    # إنشاء ملف YAML
    keypoints_schema = Path(args.annotated) / "annotations" / "keypoints_schema.json"
    if keypoints_schema.exists():
        create_yaml_config(Path(args.splits), keypoints_schema)
