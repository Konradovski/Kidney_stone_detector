# Kidney Stone Detector (OpenCV)

Prosty i skuteczny detektor kamieni nerkowych na obrazach z tomografii komputerowej (CT), napisany w języku Python przy użyciu biblioteki OpenCV. 

Projekt wykorzystuje klasyczne metody przetwarzania obrazu (nie Deep Learning), co czyni go szybkim, łatwym w konfiguracji i niewymagającym kart graficznych GPU.

![Wizualizacja działania](https://via.placeholder.com/800x400?text=Visualization+Placeholder) 
*(Tutaj możesz wstawić zrzut ekranu z działania programu)*

## Funkcje

*   **Automatyczne Wykrywanie**: Znajduje potencjalne kamienie nerkowe (zwapnienia) na podstawie jasności i kształtu.
*   **Maskowanie Ciała**: Automatycznie ignoruje elementy poza ciałem pacjenta (stół, tło) oraz zewnętrzne warstwy (skóra, żebra), koncentrując się na wnętrzu jamy brzusznej.
*   **Usuwanie Kręgosłupa**: Wyklucza z analizy centralną część obrazu, gdzie znajduje się kręgosłup (częste źródło fałszywych detekcji).
*   **Wizualizacja Etapów**: Wyświetla podgląd każdego kroku: Oryginał -> Maska Ciała -> CLAHE -> Wynik.
*   **Pełna Konfiguracja**: Wszystkie parametry (progi, rozmiary filtrów, czułość) są łatwo dostępne w pliku `config.py`.

## Wymagania

*   Python 3.x
*   Biblioteki:
    *   `opencv-python`
    *   `numpy`
    *   `matplotlib`

Zainstaluj wymagane pakiety komendą:
```bash
pip install opencv-python numpy matplotlib
```

## Użycie

1.  Umieść obrazy CT (format `.jpg`, `.png`, `.bmp`) w folderze z projektem lub dowolnym innym katalogu.
2.  Uruchom skrypt `main.py`:

    ```bash
    # Przetwarzanie obrazów w bieżącym katalogu
    python main.py

    # Lub podaj ścieżkę do folderu ze zdjęciami
    python main.py "sciezka/do/obrazow"
    ```
3.  Program wyświetli wyniki dla każdego znalezionego obrazu. Zamknij okno wykresu, aby przejść do następnego zdjęcia.

## Konfiguracja (`config.py`)

Możesz dostosować działanie algorytmu edytując plik `config.py`. Najważniejsze parametry:

### Czułość Detekcji
*   `BINARY_THRESHOLD` (domyślnie 200): Próg jasności (0-255). Obniż, jeśli kamienie nie są wykrywane; zwiększ, jeśli wykrywa za dużo tkanek.
*   `MIN_AREA` (domyślnie 5): Minimalna powierzchnia obiektu. Mniejsze wartości pozwalają wykryć drobniejsze kamienie.
*   `MIN_CIRCULARITY` (domyślnie 0.15): Jak bardzo "okrągły" musi być obiekt (0.0-1.0).

### Maskowanie (Redukcja Błędów)
*   `ENABLE_BODY_MASK`: Włącza/wyłącza wycinanie tła i żeber.
*   `BODY_EROSION_ITERATIONS`: Jak mocno odcinać krawędzie ciała. Zwiększ, jeśli wykrywa elementy na skórze/żebrach.
*   `MASK_CENTER_SPINE`: Włącza/wyłącza pionową maskę zasłaniającą kręgosłup.

## Jak to działa? (Algorytm)

1.  **Preprocessing**: Obraz jest zamieniany na odcienie szarości i odszumiany (`Median Blur`).
2.  **Poprawa Kontrastu**: Algorytm **CLAHE** (Contrast Limited Adaptive Histogram Equalization) uwydatnia lokalne szczegóły.
3.  **Maskowanie Ciała**: Wykrywany jest kontur pacjenta, wypełniany i poddawany erozji, aby usunąć zewnętrzne artefakty.
4.  **Segmentacja**: Progowanie (Thresholding) wyodrębnia najjaśniejsze obiekty (kości, kamienie).
5.  **Filtracja**: Znalezione kontury są mierzone. Obiekty zbyt małe, zbyt duże lub niekształtne są odrzucane.
