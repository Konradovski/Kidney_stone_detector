import cv2
import os
import glob
from stone_detector import KidneyStoneDetector

def run_test():
    detector = KidneyStoneDetector()
    
    # Directories to test
    directories = [
        'photos_to_test',       # Normal images
        'photos',               # Stone images
        'photos_to_test_stone'  # Additional Stone images
    ]
    
    print(f"{'Image':<30} | {'Type':<10} | {'Candidates':<10}")
    print("-" * 60)
    
    for folder in directories:
        # Search for .jpg files
        files = glob.glob(os.path.join(folder, '*.jpg'))
        
        category = "Normal" if "Normal" in str(files) or "photos_to_test" in folder and "stone" not in folder else "Stone"
        if folder == 'photos' or folder == 'photos_to_test_stone':
             category = "Stone"
        
        for file_path in files:
            try:
                # Run detection
                results = detector.process(file_path)
                image_name = os.path.basename(file_path)
                num_candidates = results['num_candidates']
                
                # Determine type based on filename if possible, else folder
                if "Normal" in image_name:
                    img_type = "Normal"
                elif "Stone" in image_name:
                    img_type = "Stone"
                else:
                    img_type = category
                    
                print(f"{image_name:<30} | {img_type:<10} | {num_candidates:<10}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    run_test()
