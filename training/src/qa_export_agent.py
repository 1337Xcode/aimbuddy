#!/usr/bin/env python3
"""
QA & Export Agent - ضمان جودة والتصدير النهائي
التحقق من دقة التوسيم وإنشاء مجموعة البيانات النهائية
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QAExportAgent:
    def __init__(self, annotated_dir: Path, qa_dir: Path, sample_size: int = 0.1):
        self.annotated_dir = Path(annotated_dir)
        self.qa_dir = Path(qa_dir)
        self.sample_size = sample_size  # 10% من البيانات
        
        self.qa_dir.mkdir(parents=True, exist_ok=True)
        (self.qa_dir / "problematic_samples").mkdir(exist_ok=True)
        
        self.qa_results = {
            "timestamp": datetime.now().isoformat(),
            "total_samples_reviewed": 0,
            "issues_found": 0,
            "issues_by_type": {},
            "samples_checked": []
        }

    def parse_yolo_label(self, label_file: Path) -> list:
        """قراءة ملف التسمية بصيغة YOLO"""
        if not label_file.exists():
            return []
        
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts])
        return labels

    def denormalize_bbox(self, norm_bbox: list, img_h: int, img_w: int) -> tuple:
        """تحويل الـ normalized bbox إلى pixel coordinates"""
        x_center, y_center, width, height = norm_bbox[:4]
        
        x_center_px = x_center * img_w
        y_center_px = y_center * img_h
        width_px = width * img_w
        height_px = height * img_h
        
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        return x1, y1, x2, y2

    def check_annotation_quality(self, image: np.ndarray, label_data: list) -> dict:
        """فحص جودة التوسيم"""
        issues = []
        img_h, img_w = image.shape[:2]
        
        for label in label_data:
            class_id = int(label[0])
            bbox = label[1:5]
            keypoints = label[5:]
            
            # فحص 1: صحة الـ Bounding Box
            if bbox[2] <= 0 or bbox[3] <= 0:
                issues.append("❌ Bounding Box غير صحيح (حجم سالب)")
            
            if bbox[0] < 0 or bbox[0] > 1 or bbox[1] < 0 or bbox[1] > 1:
                issues.append("❌ Bounding Box خارج حدود الصورة")
            
            # فحص 2: صحة النقاط المفصلية
            if len(keypoints) % 3 != 0:
                issues.append(f"❌ عدد النقاط المفصلية غير صحيح ({len(keypoints)} يجب أن يكون قابل للقسمة على 3)")
            
            kpt_count = len(keypoints) // 3
            for i in range(kpt_count):
                x = keypoints[i * 3]
                y = keypoints[i * 3 + 1]
                vis = keypoints[i * 3 + 2]
                
                if x < 0 or x > 1 or y < 0 or y > 1:
                    issues.append(f"❌ النقطة {i} خارج حدود الصورة")
                
                if vis not in [0, 1]:
                    issues.append(f"❌ قيمة Visibility غير صحيحة للنقطة {i} ({vis})")
        
        return {
            "has_issues": len(issues) > 0,
            "issue_count": len(issues),
            "issues": issues
        }

    def visualize_annotation(self, image: np.ndarray, label_data: list, output_path: Path):
        """رسم التوسيم على الصورة للتحقق البصري"""
        img_copy = image.copy()
        img_h, img_w = image.shape[:2]
        
        for label in label_data:
            class_id = int(label[0])
            bbox = label[1:5]
            keypoints = label[5:]
            
            # رسم Bounding Box
            x1, y1, x2, y2 = self.denormalize_bbox(bbox, img_h, img_w)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # رسم Keypoints
            kpt_colors = {
                0: (0, 0, 255),      # Red - Head
                1: (255, 0, 0),      # Blue - Left Shoulder
                2: (255, 255, 0),    # Cyan - Right Shoulder
                3: (0, 255, 255),    # Yellow - Torso
                4: (255, 0, 255)     # Magenta - Weapon
            }
            
            kpt_names = ["HEAD", "L_SHOULDER", "R_SHOULDER", "TORSO", "WEAPON"]
            
            for i in range(len(keypoints) // 3):
                x = keypoints[i * 3]
                y = keypoints[i * 3 + 1]
                vis = keypoints[i * 3 + 2]
                
                if x > 0 and y > 0:
                    x_px = int(x * img_w)
                    y_px = int(y * img_h)
                    
                    color = kpt_colors.get(i, (255, 255, 255))
                    if vis == 1:
                        cv2.circle(img_copy, (x_px, y_px), 5, color, -1)
                        cv2.putText(img_copy, kpt_names[i] if i < len(kpt_names) else f"KP{i}",
                                   (x_px + 10, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    else:
                        cv2.circle(img_copy, (x_px, y_px), 5, (128, 128, 128), 1)
        
        cv2.imwrite(str(output_path), img_copy)

    def process(self):
        """تنفيذ فحص الجودة"""
        logger.info("🔍 جاري جمع الصور الموسومة...")
        
        images_dir = self.annotated_dir / "images"
        labels_dir = self.annotated_dir / "labels"
        
        image_files = sorted(list(images_dir.glob("*.jpg")) + 
                           list(images_dir.glob("*.png")) +
                           list(images_dir.glob("*.jpeg")))
        
        if not image_files:
            logger.warning("⚠️ لم يتم العثور على صور موسومة!")
            return
        
        # اختيار عينة عشوائية
        sample_count = max(1, int(len(image_files) * self.sample_size))
        sample_files = random.sample(image_files, sample_count)
        
        logger.info(f"📊 سيتم فحص {sample_count} صورة من أصل {len(image_files)}")
        
        for img_path in sample_files:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"❌ فشل قراءة: {img_path.name}")
                continue
            
            label_file = labels_dir / (img_path.stem + ".txt")
            label_data = self.parse_yolo_label(label_file)
            
            # فحص الجودة
            quality_check = self.check_annotation_quality(image, label_data)
            
            # رسم التوسيم
            vis_output = self.qa_dir / "problematic_samples" / f"qa_{img_path.name}"
            self.visualize_annotation(image, label_data, vis_output)
            
            # تسجيل النتائج
            self.qa_results["total_samples_reviewed"] += 1
            
            if quality_check["has_issues"]:
                self.qa_results["issues_found"] += 1
                for issue in quality_check["issues"]:
                    issue_type = issue.split()[0]
                    self.qa_results["issues_by_type"][issue_type] = \
                        self.qa_results["issues_by_type"].get(issue_type, 0) + 1
                
                self.qa_results["samples_checked"].append({
                    "filename": img_path.name,
                    "issues": quality_check["issues"]
                })
            
            logger.info(f"✅ فحص: {img_path.name} - {'✓ بدون مشاكل' if not quality_check['has_issues'] else f'⚠️ {quality_check[\"issue_count\"]} مشكلة'}")
        
        # حفظ التقرير
        self.save_qa_report()
        logger.info("✅ اكتملت عملية فحص الجودة!")

    def save_qa_report(self):
        """حفظ تقرير الجودة"""
        report_path = self.qa_dir / f"qa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.qa_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 تقرير الجودة محفوظ: {report_path}")
        
        # طباعة الملخص
        print("\n" + "="*50)
        print("📊 تقرير فحص الجودة (QA)")
        print("="*50)
        print(f"✅ صور مفحوصة: {self.qa_results['total_samples_reviewed']}")
        print(f"⚠️ صور بها مشاكل: {self.qa_results['issues_found']}")
        print(f"📈 معدل الصور السليمة: {((self.qa_results['total_samples_reviewed'] - self.qa_results['issues_found']) / self.qa_results['total_samples_reviewed'] * 100):.1f}%")
        
        if self.qa_results["issues_by_type"]:
            print("\n📝 تفاصيل المشاكل:")
            for issue_type, count in self.qa_results["issues_by_type"].items():
                print(f"  - {issue_type}: {count}")
        
        print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="فحص جودة التوسيم")
    parser.add_argument("--annotated", type=str, default="dataset/annotated",
                       help="مسار مجلد البيانات الموسومة")
    parser.add_argument("--qa-dir", type=str, default="qa_reports",
                       help="مسار مجلد تقارير الجودة")
    parser.add_argument("--sample-size", type=float, default=0.1,
                       help="نسبة العينة (0.1 = 10%)")
    
    args = parser.parse_args()
    
    qa_agent = QAExportAgent(Path(args.annotated), Path(args.qa_dir), args.sample_size)
    qa_agent.process()
