# Zespół

* Daiana Henina
* Aneliia Henina

# Projekt

**K*-Means: Parameter-free Clustering Algorithm**

# Opis

Projekt stanowi autorską implementację adaptacyjnego algorytmu klasteryzacji K*-Means, opartego na pracy naukowej: Louis Mahon, Mirella Lapata — „K*-Means: A Parameter-free Clustering Algorithm” (maj 2025).

Głównym celem jest automatyczne i dynamiczne wyznaczanie optymalnej liczby klastrów (k) podczas jednego przebiegu algorytmu, bez konieczności ręcznego podawania tego parametru przez użytkownika. Proces optymalizacji opiera się na kryterium teorii informacji MDL (Minimum Description Length), wyważającym dokładność podziału (SSE) oraz złożoność modelu (k). Algorytm startuje od struktury k=1 i adaptacyjnie modyfikuje przestrzeń za pomocą operacji podziału klastrów (Maybe-Split) oraz ich fuzji (Maybe-Merge).

# Idea algorytmu K*-Means

Algorytm K*-Means rozwiązuje problem optymalnego doboru grup przy użyciu zasady Minimum Description Length (MDL). Model wybiera taką liczbę klastrów, która minimalizuje całkowity koszt opisu danych:
MDL = L_data + L_model

W projekcie zastosowano następujący matematyczny wzór jawny:
MDL = n * d * log2(SSE / (n * d)) + k * d * log2(n) + n * log2(k)

Gdzie:
* n – liczba punktów danych
* d – liczba cech (wymiarowość)
* k – bieżąca liczba klastrów
* SSE – suma kwadratów błędów (Sum of Squared Errors)

Główną ideą jest znalezienie idealnej równowagi pomiędzy:
* jakością dopasowania danych (niskie SSE)
* złożonością strukturalną modelu (małe k)

# Struktura projektu

k-star-means-project/
│
├── src/
│   ├── kmeans.py          # Klasyczna implementacja bazowa K-Means
│   ├── kstar_means.py     # Główna pętla adaptacyjna K*-Means z operacjami split/merge
│   ├── mdl.py             # Matematyczne obliczanie funkcji kosztu MDL oraz SSE
│   └── plots.py           # Skrypty do generowania i wizualizacji wykresów
│
├── experiments/
│   ├── run_synthetic.py   # Główny eksperyment na wieloformatowych danych syntetycznych
│   └── compare_methods.py # Eksperyment porównujący K*-Means z innymi podejściami
│
├── results/               # Folder zawierający automatycznie generowane wykresy i pliki CSV
│
├── EXAMPLE.md             # Dokładna, szczegółowa analiza przeprowadzonych eksperymentów
├── README.md              # Główny plik z opisem technicznym projektu
└── requirements.txt       # Definicje zależności i wymaganych bibliotek

# Sposób uruchomienia

Wymagania:
* Python 3.11+
* virtualenv / venv

Zależności:
* numpy
* pandas
* matplotlib
* scikit-learn

Instalacja i uruchomienie:

# 1. Stworzenie i aktywacja środowiska wirtualnego
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalacja wymaganych bibliotek
pip install -r requirements.txt

# 3. Uruchomienie głównego eksperymentu (dane syntetyczne)
python3 experiments/run_synthetic.py

Skrypt ten automatycznie wykona następujące kroki:
* wygeneruje syntetyczne zbiory danych (scenariusze FAR, MEDIUM, CLOSE)
* uruchomi adaptacyjny proces K*-Means dla każdego przypadku
* obliczy wartości SSE, Silhouette Score oraz MDL na przestrzeni iteracji
* automatycznie dobierze najlepszą strukturę klastrów za pomocą kryterium MDL
* zapisze wyniki liczbowe oraz gotowe wykresy w folderze wynikowym

# Pliki wynikowe (Output)

Po uruchomieniu głównego skryptu, w katalogu results/ zostaną automatycznie utworzone następujące pliki:
* synthetic_results.csv – tabela zbiorcza zawierająca parametry k, SSE, Silhouette oraz MDL
* sse_vs_iteration.png – wykres spadku błędu SSE w kolejnych krokach optymalizacji
* mdl_vs_iteration.png – wykres zmian całkowitego kosztu informacyjnego MDL
* k_vs_iteration.png – wykres prezentujący dynamiczną zmianę liczby klastrów (linie split/merge)
* clusters_kstar.png – ostateczna wizualizacja graficzna podziału przestrzeni przez algorytm

# Porównanie metod (Method comparison)

Aby dokonać ewaluacji i porównać nasz algorytm z tradycyjnymi technikami doboru k, należy uruchomić skrypt:
python3 experiments/compare_methods.py

Spowoduje to wygenerowanie pliku zbiorczego:
* method_comparison.csv – zestawienie wydajnościowe porównujące:
  - manualny dobór parametru k (K-Means)
  - selekcję opartą o Silhouette Score
  - w pełni automatyczny algorytm K*-Means oparty na MDL

# Dodatkowe informacje

Projekt bazuje bezpośrednio na matematycznych założeniach i algorytmice opisanej w artykule naukowym:
* Mahon, L., & Lapata, M. (2025). K*-Means: A Parameter-free Clustering Algorithm. arXiv preprint arXiv:2505.11904.