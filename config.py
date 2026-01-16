
class Config:
    """
    Konfiguracja parametrów dla detektora kamieni nerkowych.
    Wszystkie parametry sterujące przetwarzaniem obrazu znajdują się tutaj.
    """
    
    # --- Preprocessing ---
    BLUR_KERNEL_SIZE = 5
    
    # --- CLAHE ---
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # --- Segmentacja ---
    BINARY_THRESHOLD = 200
    
    # --- Maskowanie Ciała ---
    ENABLE_BODY_MASK = True
    # Obniżony próg, aby lepiej łapać całe ciało nawet na ciemniejszych skanach
    BODY_THRESHOLD = 5 
    # Erozja maski - margines od krawędzi
    BODY_EROSION_ITERATIONS = 30
    # Rozmiar kernela do "sklejania" (closing) maski ciała przed detekcją
    BODY_CLOSE_KERNEL_SIZE = 7
    
    # --- Morfologia (dla kamieni) ---
    MORPH_KERNEL_SIZE = (3, 3)
    MORPH_CLOSE_ITERATIONS = 1
    MORPH_DILATE_ITERATIONS = 1

    # Filtrowanie Obiektów (Kluczowe) ---
    # Minimalne pole powierzchni (w pikselach), aby obiekt był uznany za kamień.
    # Odsiewa szumy.
    MIN_AREA = 5
    # Maksymalne pole powierzchni. Odsiewa kręgosłup i duże organy.
    MAX_AREA = 300 
    
    # Minimalna cyrkularność (0.0 - 1.0). 1.0 to idealne koło.
    # Kamienie są często owalne/okrągłe.
    MIN_CIRCULARITY = 0.15

    # --- Maskowanie Kręgosłupa ---
    MASK_CENTER_SPINE = True
    SPINE_MASK_WIDTH = 100

    # --- Wizualizacja ---
    BBOX_COLOR = (0, 0, 255) # Czerwony
    BBOX_THICKNESS = 1
    FIG_SIZE = (12, 10)
