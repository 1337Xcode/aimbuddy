#!/usr/bin/env python3
"""
Data Cleaner Agent - فلترة ومعالجة الصور الخام
المهام:
1. حذف الإطارات المتكررة (تشابه >95%)
2. استبعاد الإطارات ذات الضبابية الحركة الشديدة
3. توحيد أحجام الصور (Resizing) مع الحفاظ على نسبة العرض
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import argparse
from tqdm import tqdm
import logging

# إعداد Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/cleaning_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, input_dir: Path, output_dir: Path, target_size: int = 640):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "removed_duplicates").mkdir(exist_ok=True)
        (self.output_dir / "removed_blur").mkdir(exist_ok=True)
        
        self.stats = {
            "total_images": 0,
            "duplicates_removed": 0,
            "blur_removed": 0,
            "resized_success": 0,
            "processing_errors": 0,
            "duplicates_list": [],
            "blur_list": []
        }

    def calculate_blur_score(self, image: np.ndarray) -> float:
        """
        حساب درجة الضبابية باستخدام Laplacian Variance
        درجة عالية = صورة واضحة
        درجة منخفضة = صورة ضبابية
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        حساب نسبة التشابه بين صورتين باستخدام SSIM
        1.0 = متطابقة تماماً
        0.0 = مختلفة تماماً
        """
        # تصغير الصور لتسريع المقارنة
        h, w = 256, 256
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        similarity, _ = ssim(gray1, gray2, full=True)
        return similarity

    def resize_with_padding(self, image: np.ndarray, target_size: int = 640) -> np.ndarray:
        """
        تغيير حجم الصورة مع الحفاظ على نسبة العرض والارتفاع
        إضافة padding أسود إذا لزم الأمر
        """
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # تغيير الحجم
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # إضافة padding أسود
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        return padded

    def process(self, blur_threshold: float = 100.0, similarity_threshold: float = 0.95):
        """
        معالجة كاملة: فلترة + تغيير حجم
        blur_threshold: الحد الأدنى لدرجة الضبابية (أقل = ضبابية أكثر)
        similarity_threshold: الحد الأدنى للتشابه (أعلى = متطابقة أكثر)
        """
        logger.info(f"🔍 جاري البحث عن الصور في: {self.input_dir}")
        
        # البحث عن جميع الصور
        image_files = list(self.input_dir.glob("*.jpg")) + \
                     list(self.input_dir.glob("*.png")) + \
                     list(self.input_dir.glob("*.jpeg"))
        
        self.stats["total_images"] = len(image_files)
        logger.info(f"📊 عدد الصور المكتشفة: {self.stats['total_images']}")
        
        if self.stats["total_images"] == 0:
            logger.warning("⚠️ لم يتم العثور على أي صور!")
            return
        
        processed_images = []
        
        for idx, img_path in enumerate(tqdm(image_files, desc="معالجة الصور")):
            try:
                # قراءة الصورة
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"❌ فشل قراءة: {img_path.name}")
                    self.stats["processing_errors"] += 1
                    continue
                
                # فحص الضبابية
                blur_score = self.calculate_blur_score(image)
                if blur_score < blur_threshold:
                    logger.info(f"🌫️ صورة ضبابية: {img_path.name} (درجة: {blur_score:.2f})")
                    cv2.imwrite(
                        str(self.output_dir / "removed_blur" / img_path.name),
                        image
                    )
                    self.stats["blur_removed"] += 1
                    self.stats["blur_list"].append({
                        "filename": img_path.name,
                        "blur_score": float(blur_score)
                    })
                    continue
                
                # فحص التكرار مع الصور المعالجة سابقاً
                is_duplicate = False
                for prev_img, prev_path in processed_images:
                    similarity = self.calculate_similarity(image, prev_img)
                    if similarity > similarity_threshold:
                        logger.info(
                            f"🔄 صورة مكررة: {img_path.name} ≈ {prev_path.name} "
                            f"(تشابه: {similarity:.2%})"
                        )
                        cv2.imwrite(
                            str(self.output_dir / "removed_duplicates" / img_path.name),
                            image
                        )
                        self.stats["duplicates_removed"] += 1
                        self.stats["duplicates_list"].append({
                            "duplicate": img_path.name,
                            "similar_to": prev_path.name,
                            "similarity": float(similarity)
                        })
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                # تغيير الحجم والحفظ
                resized = self.resize_with_padding(image, self.target_size)
                output_path = self.output_dir / "images" / img_path.name
                cv2.imwrite(str(output_path), resized)
                
                logger.info(f"✅ معالجة: {img_path.name} → {output_path.name}")
                processed_images.append((resized, img_path.name))
                self.stats["resized_success"] += 1
                
            except Exception as e:
                logger.error(f"❌ خطأ في معالجة {img_path.name}: {str(e)}")
                self.stats["processing_errors"] += 1
        
        self.save_report()
        logger.info("✅ اكتملت معالجة البيانات!")

    def save_report(self):
        """حفظ تقرير المعالجة"""
        report_path = self.output_dir / "cleaning_report.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "target_size": self.target_size,
            "statistics": self.stats,
            "summary": {
                "total_input": self.stats["total_images"],
                "total_output": self.stats["resized_success"],
                "removed": self.stats["duplicates_removed"] + self.stats["blur_removed"] + self.stats["processing_errors"],
                "success_rate": f"{(self.stats['resized_success'] / max(1, self.stats['total_images']) * 100):.1f}%"
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 التقرير محفوظ: {report_path}")
        
        # طباعة الملخص
        print("\n" + "="*50)
        print("📊 ملخص معالجة البيانات")
        print("="*50)
        print(f"✅ صور نظيفة: {self.stats['resized_success']}")
        print(f"🔄 صور مكررة: {self.stats['duplicates_removed']}")
        print(f"🌫️ صور ضبابية: {self.stats['blur_removed']}")
        print(f"❌ أخطاء معالجة: {self.stats['processing_errors']}")
        print(f"📈 معدل النجاح: {report['summary']['success_rate']}")
        print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="فلترة ومعالجة الصور الخام")
    parser.add_argument("--input", type=str, default="dataset/raw/screenshots",
                       help="مسار مجلد الصور الخام")
    parser.add_argument("--output", type=str, default="dataset/processed/filtered",
                       help="مسار مجلد الصور المعالجة")
    parser.add_argument("--size", type=int, default=640,
                       help="حجم الصورة المستهدف (640 أو 1024)")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                       help="حد الضبابية (أقل = ضبابية أكثر)")
    parser.add_argument("--similarity", type=float, default=0.95,
                       help="حد التشابه للتكرار (0.0-1.0)")
    
    args = parser.parse_args()
    
    cleaner = DataCleaner(args.input, args.output, args.size)
    cleaner.process(args.blur_threshold, args.similarity)
