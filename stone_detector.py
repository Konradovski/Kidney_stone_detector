# ===== IMPORTY BIBLIOTEK =====
# cv2 - OpenCV, główna biblioteka do przetwarzania obrazów
# numpy - operacje matematyczne na macierzach i tablicach
# matplotlib - wizualizacja wyników przetwarzania
# config - plik konfiguracyjny z parametrami detekcji
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
        """
        Konstruktor klasy - inicjalizuje obiekt z parametrami z pliku konfiguracyjnego.
        """
        self.cfg = Config()

    def load_image(self, path):
        """
        Ładuje obraz CT z podanej ścieżki.
        
        Args:
            path: Ścieżka do pliku obrazu
            
        Returns:
            Załadowany obraz w formacie BGR (OpenCV)
            
        Raises:
            ValueError: Gdy nie można załadować obrazu z podanej ścieżki
        """
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Nie można załadować obrazu: {path}")
        return image

    def preprocess(self, image):
        """
        Preprocessing obrazu - przygotowanie do detekcji kamieni.
        
        Etapy przetwarzania:
        1. Konwersja do skali szarości (grayscale) - uproszczenie obrazu
        2. Redukcja szumów (median blur) - wygładzenie obrazu
        3. CLAHE (Contrast Limited Adaptive Histogram Equalization) - poprawa kontrastu
        
        Args:
            image: Oryginalny obraz kolorowy lub szary
            
        Returns:
            Tuple (gray, blurred, enhanced):
                - gray: Obraz w skali szarości
                - blurred: Obraz po redukcji szumów
                - enhanced: Obraz po poprawie kontrastu (CLAHE)
        """
        # ==== KROK 1: KONWERSJA DO SKALI SZAROŚCI ====
        # Sprawdzamy czy obraz jest kolorowy (3 kanały) czy już szary
        if len(image.shape) == 3:
            # Konwersja BGR -> Grayscale (wartości 0-255)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # Obraz już jest w skali szarości
            gray = image.copy()

        # ==== KROK 2: REDUKCJA SZUMÓW ====
        # Median Blur - usuwanie szumów przy zachowaniu krawędzi
        # Rozmiar kernela z konfiguracji (np. 5x5)
        blurred = cv2.medianBlur(gray, self.cfg.BLUR_KERNEL_SIZE)

        # ==== KROK 3: POPRAWA KONTRASTU - CLAHE ====
        # CLAHE poprawia lokalny kontrast, dzięki czemu kamienie są lepiej widoczne
        # clipLimit - ograniczenie wzmocnienia kontrastu (zapobiega nadmiernemu wzmocnieniu szumu)
        # tileGridSize - rozmiar siatki dla lokalnej adaptacji (np. 8x8)
        clahe = cv2.createCLAHE(clipLimit=self.cfg.CLAHE_CLIP_LIMIT, 
                                tileGridSize=self.cfg.CLAHE_TILE_GRID_SIZE)
        enhanced = clahe.apply(blurred)

        return gray, blurred, enhanced

    def get_body_mask(self, gray_image):
        """
        Tworzy maskę obszaru ciała, odcinając zewnętrzne krawędzie (skórę, żebra).
        Ulepszona wersja: skleja fragmenty i bierze wszystkie duże kontury.
        
        Cel: Eliminacja fałszywych detekcji na krawędziach obrazu (żebra, skóra)
        poprzez utworzenie maski "obszaru roboczego" - wnętrza ciała.
        
        Args:
            gray_image: Obraz w skali szarości
            
        Returns:
            Maska binarna obszaru ciała (białe piksele = obszar roboczy)
        """
        # ==== KROK 1: PROGOWANIE - ODDZIELENIE TKANEK OD TŁA ====
        # Threshold oddziela jasne tkanki (ciało) od ciemnego tła (powietrze)
        # BODY_THRESHOLD z config.py określa próg (np. 20-30)
        _, body_thresh = cv2.threshold(gray_image, self.cfg.BODY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # ==== KROK 2: SKLEJANIE FRAGMENTÓW - MORPHOLOGICAL CLOSING ====
        # Problem: Czasem obszar ciała jest podzielony na fragmenty (np. przez powietrze w jelitach)
        # Rozwiązanie: Closing (dylatacja + erozja) zamyka luki i skleja fragmenty
        # Używamy dużego kernela, żeby zamknąć przerwy między tkankami
        closing_kernel = np.ones((self.cfg.BODY_CLOSE_KERNEL_SIZE, self.cfg.BODY_CLOSE_KERNEL_SIZE), np.uint8)
        body_thresh = cv2.morphologyEx(body_thresh, cv2.MORPH_CLOSE, closing_kernel, iterations=2)

        # ==== KROK 3: ZNAJDOWANIE KONTURÓW CIAŁA ====
        # Znajdujemy zewnętrzne kontury - każdy oddzielny fragment ciała
        contours, _ = cv2.findContours(body_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Inicjalizacja pustej maski (czarnej)
        mask = np.zeros_like(gray_image)
        
        if contours:
            # ==== KROK 4: FILTROWANIE KONTURÓW - ELIMINACJA SZUMÓW ====
            # Filtrujemy kontury - bierzemy tylko te wystarczająco duże
            # Małe kontury to szum z tła, który nas nie interesuje
            total_pixels = gray_image.shape[0] * gray_image.shape[1]
            # Minimum 1% powierzchni obrazu (można dostroić)
            min_body_part_area = total_pixels * 0.01 
            
            # Lista znaczących konturów (duże fragmenty ciała)
            significant_contours = [c for c in contours if cv2.contourArea(c) > min_body_part_area]
            
            # ==== KROK 5: WYPEŁNIENIE MASKI ====
            # Wypełniamy WSZYSTKIE znaczące kontury na biało
            # Dzięki temu maska obejmuje cały obszar ciała (nawet jeśli jest podzielony)
            if significant_contours:
                cv2.drawContours(mask, significant_contours, -1, 255, thickness=cv2.FILLED)
            
                # ==== KROK 6: EROZJA MASKI - "SKURCZENIE" OBSZARU ====
                # Cel: Odsunięcie się od krawędzi ciała (żebra, skóra)
                # Erozja zmniejsza maskę od krawędzi, eliminując fałszywe detekcje na obrzeżach
                # Liczba iteracji z config.py (np. 10-20)
                if self.cfg.BODY_EROSION_ITERATIONS > 0:
                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=self.cfg.BODY_EROSION_ITERATIONS)
                
        return mask

    def segment(self, enhanced_image, body_mask=None):
        """
        Segmentacja obrazu z uwzględnieniem maski ciała i kręgosłupa.
        
        Cel: Wyodrębnienie potencjalnych kamieni nerkowych poprzez binaryzację
        i zastosowanie masek eliminujących fałszywe detekcje.
        
        Args:
            enhanced_image: Obraz po CLAHE (poprawiony kontrast)
            body_mask: Opcjonalna maska obszaru ciała (None = bez maski)
            
        Returns:
            Binarna maska z potencjalnymi kamieniami (białe piksele = kandydaci)
        """
        # ==== KROK 1: PROGOWANIE - BINARYZACJA OBRAZU ====
        # Oddzielamy jasne struktury (potencjalne kamienie) od ciemnego tła
        # BINARY_THRESHOLD z config.py określa próg jasności (np. 180-200)
        # Wartości powyżej progu → 255 (białe), poniżej → 0 (czarne)
        _, binary = cv2.threshold(enhanced_image, 
                                  self.cfg.BINARY_THRESHOLD, 
                                  255, 
                                  cv2.THRESH_BINARY)

        # ==== KROK 2: NAKŁADANIE MASEK - ELIMINACJA FAŁSZYWYCH DETEKCJI ====
        
        # --- MASKA KRĘGOSŁUPA ---
        # Problem: Kręgosłup (środek obrazu) często daje fałszywe wykrycia
        # Rozwiązanie: Zerujemy pionowy pas po środku obrazu
        if self.cfg.MASK_CENTER_SPINE:
            h, w = binary.shape
            center_x = w // 2  # Środek obrazu w poziomie
            half_width = self.cfg.SPINE_MASK_WIDTH // 2  # Połowa szerokości paska
            # Zerowanie pikseli w pasie [center_x - half_width : center_x + half_width]
            binary[:, center_x - half_width : center_x + half_width] = 0

        # --- MASKA CIAŁA ---
        # Jeśli maska ciała jest dostępna, zachowujemy tylko detekcje wewnątrz ciała
        # Operacja AND: binary AND body_mask → tylko piksele w obu maskach
        if body_mask is not None:
            binary = cv2.bitwise_and(binary, binary, mask=body_mask)

        # ==== KROK 3: OPERACJE MORFOLOGICZNE - OCZYSZCZANIE I ŁĄCZENIE ====
        # Cel: Usunięcie małych szumów i połączenie bliskich fragmentów
        
        # Kernel (element strukturalny) do operacji morfologicznych
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.cfg.MORPH_KERNEL_SIZE)
        
        # --- CLOSING (zamykanie) ---
        # Closing = Dylatacja + Erozja
        # Zamyka małe dziury wewnątrz obiektów i łączy bliskie fragmenty
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.cfg.MORPH_CLOSE_ITERATIONS)
        
        # --- DILATE (dylatacja) ---
        # Rozszerza obiekty, łącząc bliskie struktury
        # Pomaga wykryć kamienie, które mogą być częściowo rozmyte
        result = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=self.cfg.MORPH_DILATE_ITERATIONS)

        return result

    def filter_candidates(self, contours):
        """
        Filtruje kandydatów na kamienie nerkowe na podstawie cech geometrycznych.
        
        Kryteria filtracji:
        1. Obszar (area) - kamienie mają określony zakres wielkości
        2. Kolistość (circularity) - kamienie są zazwyczaj okrągławe
        
        Args:
            contours: Lista konturów znalezionych w obrazie binarnym
            
        Returns:
            Lista zatwierdzonych konturów (potencjalne kamienie)
        """
        valid_candidates = []
        
        # Iteracja po wszystkich znalezionych konturach
        for cnt in contours:
            # ==== KRYTERIUM 1: OBSZAR (WIELKOŚĆ) ====
            # Obliczamy pole powierzchni konturu w pikselach
            area = cv2.contourArea(cnt)
            
            # Sprawdzamy czy obszar mieści się w dozwolonym zakresie
            # MIN_AREA i MAX_AREA z config.py (np. 10-2000 pikseli)
            # Zbyt małe → szum, zbyt duże → inne struktury (np. pęcherz)
            if not (self.cfg.MIN_AREA <= area <= self.cfg.MAX_AREA):
                continue  # Odrzucamy kandydata
                
            # ==== KRYTERIUM 2: KOLISTOŚĆ (CIRCULARITY) ====
            # Obliczamy obwód konturu
            perimeter = cv2.arcLength(cnt, True)
            
            # Zabezpieczenie przed dzieleniem przez zero
            if perimeter == 0:
                continue
            
            # Wzór na kolistość: circularity = 4π * area / perimeter²
            # Wartość 1.0 = idealne koło
            # Wartości < 1.0 = mniej okrągłe kształty
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Sprawdzamy czy kolistość jest powyżej minimalnego progu
            # MIN_CIRCULARITY z config.py (np. 0.3-0.5)
            # Odrzuca wydłużone struktury (naczynia, kości)
            if circularity < self.cfg.MIN_CIRCULARITY:
                continue
            
            # ==== KRYTERIUM 3: SOLIDNOŚĆ (SOLIDITY) ====
            # Solidność = Pole konturu / Pole otoczki wypukłej (convex hull)
            # Kamienie są wypukłe (wartość bliska 1.0)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = float(area) / hull_area
            else:
                solidity = 0
                
            # Sprawdzamy czy solidność jest powyżej minimalnego progu
            if solidity < self.cfg.MIN_SOLIDITY:
                continue

            # ==== ZATWIERDZENIE KANDYDATA ====
            # Kontur spełnia wszystkie kryteria → dodajemy do listy
            valid_candidates.append(cnt)
            
        return valid_candidates

    def process(self, image_path):
        """
        Główna metoda pipeline - kompletny proces detekcji kamieni nerkowych.
        
        Etapy:
        1. Wczytanie obrazu
        2. Preprocessing (grayscale, blur, CLAHE)
        3. Utworzenie maski ciała (opcjonalnie)
        4. Segmentacja (binaryzacja + maski + morfologia)
        5. Wykrycie konturów
        6. Filtracja kandydatów (area, circularity)
        7. Wizualizacja wyników
        
        Args:
            image_path: Ścieżka do obrazu CT
            
        Returns:
            Słownik z wynikami wszystkich etapów przetwarzania:
                - original: Oryginalny obraz
                - gray: Obraz w skali szarości
                - enhanced: Obraz po CLAHE
                - body_mask: Maska obszaru ciała (lub None)
                - binary_mask: Maska binarna po segmentacji
                - result_vis: Obraz z zaznaczonymi kamieniami
                - num_candidates: Liczba wykrytych kandydatów
        """
        # ==== ETAP 1: WCZYTANIE OBRAZU ====
        original = self.load_image(image_path)
        
        # ==== ETAP 2: PREPROCESSING ====
        # Konwersja, redukcja szumów, poprawa kontrastu
        gray, blurred, enhanced = self.preprocess(original)
        
        # ==== ETAP 3: MASKA CIAŁA (OPCJONALNIE) ====
        # Jeśli włączona w config, tworzymy maskę eliminującą krawędzie
        body_mask = None
        if self.cfg.ENABLE_BODY_MASK:
            body_mask = self.get_body_mask(blurred)

        # ==== ETAP 4: SEGMENTACJA ====
        # Binaryzacja + nakładanie masek + operacje morfologiczne
        binary_mask = self.segment(enhanced, body_mask)
        
        # ==== ETAP 5: WYKRYCIE KONTURÓW ====
        # Znajdujemy wszystkie oddzielne białe obiekty w masce binarnej
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ==== ETAP 6: FILTRACJA KANDYDATÓW ====
        # Odrzucamy kontury nie spełniające kryteriów (wielkość, kolistość)
        candidates = self.filter_candidates(contours)
        
        # ==== ETAP 7: WIZUALIZACJA WYNIKÓW ====
        # Rysujemy prostokąty wokół zatwierdzonych kandydatów
        result_vis = original.copy()
        for cnt in candidates:
            # Bounding box (prostokąt opisujący kontur)
            x, y, w, h = cv2.boundingRect(cnt)
            # Rysowanie prostokąta na obrazie (kolor i grubość z config)
            cv2.rectangle(result_vis, (x, y), (x + w, y + h), 
                          self.cfg.BBOX_COLOR, self.cfg.BBOX_THICKNESS)

        # ==== ZWRACANIE WYNIKÓW ====
        # Zwracamy wszystkie etapy przetwarzania dla celów analizy i wizualizacji
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
        """
        Wizualizacja wszystkich etapów przetwarzania w jednym oknie.
        
        Tworzy siatkę 2x3 z 6 panelami pokazującymi kolejne etapy pipeline:
        1. Oryginalny obraz
        2. Maska ciała (obszar roboczy)
        3. Obraz po CLAHE
        4. Maska binarna
        5. Wynik końcowy z zaznaczonymi kamieniami
        6. Pusty panel (rezerwowy)
        
        Args:
            results: Słownik zwrócony przez metodę process()
        """
        # ==== KONFIGURACJA SIATKI ====
        rows = 2  # 2 wiersze
        cols = 3  # 3 kolumny
        
        # Utworzenie figury matplotlib o rozmiarze z config
        plt.figure(figsize=self.cfg.FIG_SIZE)
        
        # ==== PANEL 1: ORYGINALNY OBRAZ ====
        plt.subplot(rows, cols, 1)
        # Konwersja BGR (OpenCV) → RGB (matplotlib)
        plt.imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
        plt.title('1. Oryginał')
        plt.axis('off')  # Wyłączenie osi współrzędnych
        
        # ==== PANEL 2: MASKA CIAŁA ====
        plt.subplot(rows, cols, 2)
        if results['body_mask'] is not None:
            # Wyświetlenie maski w skali szarości
            plt.imshow(results['body_mask'], cmap='gray')
            plt.title('2. Maska Ciała (Obszar Roboczy)')
        else:
            # Jeśli maska wyłączona, wyświetlamy informację
            plt.text(0.5, 0.5, "Maska wyłączona", ha='center')
        plt.axis('off')
        
        # ==== PANEL 3: OBRAZ PO CLAHE ====
        plt.subplot(rows, cols, 3)
        plt.imshow(results['enhanced'], cmap='gray')
        plt.title('3. CLAHE')
        plt.axis('off')
        
        # ==== PANEL 4: MASKA BINARNA ====
        plt.subplot(rows, cols, 4)
        plt.imshow(results['binary_mask'], cmap='gray')
        plt.title('4. Ostateczna Maska Binarna')
        plt.axis('off')
        
        # ==== PANEL 5: WYNIK KOŃCOWY ====
        plt.subplot(rows, cols, 5)
        # Wyświetlenie obrazu z zaznaczonymi prostokątami
        plt.imshow(cv2.cvtColor(results['result_vis'], cv2.COLOR_BGR2RGB))
        # Tytuł zawiera liczbę wykrytych kandydatów
        plt.title(f'5. Wynik: {results["num_candidates"]} kandydatów')
        plt.axis('off')
        
        # ==== PANEL 6: PUSTY (REZERWOWY) ====
        plt.subplot(rows, cols, 6)
        plt.axis('off')  # Wyłączamy osie, panel pozostaje pusty
        
        # ==== FINALIZACJA ====
        plt.tight_layout()  # Automatyczne dopasowanie odstępów
        plt.show()  # Wyświetlenie okna z wizualizacją

