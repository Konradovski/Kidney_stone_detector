import os
import glob
import sys
from stone_detector import KidneyStoneDetector

def main():
    # Domyślny folder to bieżący katalog
    input_folder = "."
    
    # Jeśli podano argument, użyj go jako ścieżki
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]

    extensions = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]
    image_files = []
    
    # Szukamy plików ze wszystkimi rozszerzeniami
    for ext in extensions:
        # search_pattern np. "./*.jpg"
        search_pattern = os.path.join(input_folder, ext)
        found = glob.glob(search_pattern)
        image_files.extend(found)
        
    print(f"Szukanie obrazów w: {os.path.abspath(input_folder)}")
    print(f"Znaleziono {len(image_files)} obrazów.")
    
    if not image_files:
        print("Brak obrazów do przetworzenia. Upewnij się, że jesteś we właściwym folderze lub podaj ścieżkę jako argument.")
        return

    detector = KidneyStoneDetector()
    
    print("-" * 50)
    for img_path in image_files:
        print(f"Przetwarzanie pliku: {os.path.basename(img_path)}")
        try:
            results = detector.process(img_path)
            cand_count = results['num_candidates']
            print(f"  -> Sukces. Znaleziono kandydatów: {cand_count}")
            
            # Wyświetlamy wizualizację
            print("  -> Wyświetlanie wyników... (Zimknij okno wykresu, aby przejść do następnego zdjęcia)")
            detector.visualize_steps(results)
            
        except Exception as e:
            print(f"  -> BŁĄD przetwarzania: {e}")
        print("-" * 50)

if __name__ == "__main__":
    main()
