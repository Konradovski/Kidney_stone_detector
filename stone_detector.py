import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import Config

class KidneyStoneDetector:
    """
    Klasa odpowiedzialna za detekcję kamieni nerkowych na obrazach CT.
    Wykorzystuje klasyczne techniki przetwarzania obrazu.
    """

    def __init__(self):
        self.cfg = Config()

    def load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Nie można załadować obrazu: {path}")
        return image

    def preprocess(self, image):
        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Noise Reduction
        blurred = cv2.medianBlur(gray, self.cfg.BLUR_KERNEL_SIZE)

        # 3. CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.cfg.CLAHE_CLIP_LIMIT, 
                                tileGridSize=self.cfg.CLAHE_TILE_GRID_SIZE)
        enhanced = clahe.apply(blurred)

        return gray, blurred, enhanced

    def get_body_mask(self, gray_image):
        """
        Tworzy maskę obszaru ciała, odcinając zewnętrzne krawędzie (skórę, żebra).
        Ulepszona wersja: skleja fragmenty i bierze wszystkie duże kontury.
        """
        # 1. Progowanie
        _, body_thresh = cv2.threshold(gray_image, self.cfg.BODY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # 2. Sklejanie fragmentów (Morphological Closing)
        # Używamy dużego kernela, żeby zamknąć ewentualne przerwy między tkankami
        closing_kernel = np.ones((self.cfg.BODY_CLOSE_KERNEL_SIZE, self.cfg.BODY_CLOSE_KERNEL_SIZE), np.uint8)
        body_thresh = cv2.morphologyEx(body_thresh, cv2.MORPH_CLOSE, closing_kernel, iterations=2)

        # 3. Znajdź kontury
        contours, _ = cv2.findContours(body_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray_image)
        
        if contours:
            # Filtrujemy kontury - bierzemy tylko te wystarczająco duże
            # np. większe niż 1% powierzchni obrazu, żeby nie brać szumu z tła
            total_pixels = gray_image.shape[0] * gray_image.shape[1]
            min_body_part_area = total_pixels * 0.01 
            
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_body_part_area]
            
            # Wypełnij WSZYSTKIE znaczące kontury ciała
            if significant_contours:
                cv2.drawContours(mask, significant_contours, -1, 255, thickness=cv2.FILLED)
            
                # Erozja maski - "skurczenie" jej
                if self.cfg.BODY_EROSION_ITERATIONS > 0:
                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=self.cfg.BODY_EROSION_ITERATIONS)
                
        return mask

    def segment(self, enhanced_image, body_mask=None):
        """
        Segmentacja obrazu z uwzględnieniem maski ciała i kręgosłupa.
        """
        # 1. Thresholding
        _, binary = cv2.threshold(enhanced_image, 
                                  self.cfg.BINARY_THRESHOLD, 
                                  255, 
                                  cv2.THRESH_BINARY)

        # 2. Nakładanie Masek
        
        # Maska Kręgosłupa
        if self.cfg.MASK_CENTER_SPINE:
            h, w = binary.shape
            center_x = w // 2
            half_width = self.cfg.SPINE_MASK_WIDTH // 2
            binary[:, center_x - half_width : center_x + half_width] = 0

        # Maska Ciała
        if body_mask is not None:
            binary = cv2.bitwise_and(binary, binary, mask=body_mask)

        # 3. Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg.MORPH_KERNEL_SIZE)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.cfg.MORPH_CLOSE_ITERATIONS)
        result = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=self.cfg.MORPH_DILATE_ITERATIONS)

        return result

    def filter_candidates(self, contours):
        valid_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.cfg.MIN_AREA <= area <= self.cfg.MAX_AREA):
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.cfg.MIN_CIRCULARITY:
                continue
            
            valid_candidates.append(cnt)
        return valid_candidates

    def process(self, image_path):
        original = self.load_image(image_path)
        
        gray, blurred, enhanced = self.preprocess(original)
        
        body_mask = None
        if self.cfg.ENABLE_BODY_MASK:
            body_mask = self.get_body_mask(blurred)

        binary_mask = self.segment(enhanced, body_mask)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = self.filter_candidates(contours)
        
        result_vis = original.copy()
        for cnt in candidates:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_vis, (x, y), (x + w, y + h), 
                          self.cfg.BBOX_COLOR, self.cfg.BBOX_THICKNESS)

        return {
            "original": original,
            "gray": gray,
            "enhanced": enhanced,
            "body_mask": body_mask,
            "binary_mask": binary_mask,
            "result_vis": result_vis,
            "num_candidates": len(candidates)
        }

    def visualize_steps(self, results):
        rows = 2
        cols = 3
        
        plt.figure(figsize=self.cfg.FIG_SIZE)
        
        plt.subplot(rows, cols, 1)
        plt.imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
        plt.title('1. Oryginał')
        plt.axis('off')
        
        plt.subplot(rows, cols, 2)
        if results['body_mask'] is not None:
            plt.imshow(results['body_mask'], cmap='gray')
            plt.title('2. Maska Ciała (Obszar Roboczy)')
        else:
            plt.text(0.5, 0.5, "Maska wyłączona", ha='center')
        plt.axis('off')
        
        plt.subplot(rows, cols, 3)
        plt.imshow(results['enhanced'], cmap='gray')
        plt.title('3. CLAHE')
        plt.axis('off')
        
        plt.subplot(rows, cols, 4)
        plt.imshow(results['binary_mask'], cmap='gray')
        plt.title('4. Ostateczna Maska Binarna')
        plt.axis('off')
        
        plt.subplot(rows, cols, 5)
        plt.imshow(cv2.cvtColor(results['result_vis'], cv2.COLOR_BGR2RGB))
        plt.title(f'5. Wynik: {results["num_candidates"]} kandydatów')
        plt.axis('off')
        
        plt.subplot(rows, cols, 6)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
