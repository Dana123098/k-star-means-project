
## Postawienie problemu

Klasyczny algorytm $k$-means jest jednym z najpopularniejszych i najbardziej intuicyjnych narzędzi do klasteryzacji danych. Jego głównym celem jest podział zbioru danych na $k$ rozłącznych grup (klastrów) w taki sposób, aby punkty wewnątrz jednego klastra były jak najbardziej podobne do siebie.

Osiąga się to poprzez minimalizację błędu SSE (*Sum of Squared Errors*):

$$SSE = \sum_{i=1}^{k} \sum_{x \in C_i} \| x - \mu_i \|^2$$

gdzie:
* $x$ — punkt danych,
* $\mu_i$ — centroid klastra $C_i$.

---

Niestety, algorytm ten posiada istotną wadę: wymaga od użytkownika zdefiniowania dokładnej liczby klastrów ($k$) przed rozpoczęciem obliczeń. W praktycznych zastosowaniach struktura danych jest zazwyczaj nieznana, dlatego wybór odpowiedniego $k$ stanowi poważny problem.

Błędny wybór parametru $k$ prowadzi do pogorszenia jakości analizy:
* Zbyt małe $k$ powoduje sztuczne łączenie naturalnie odseparowanych grup danych.
* Zbyt duże $k$ prowadzi do nadmiernego dzielenia spójnych klastrów na małe i mało użyteczne fragmenty.

Standardowe metody rozwiązywania tego problemu, takie jak metoda łokcia (*Elbow Method*), *silhouette score* lub kryterium BIC, wymagają wielokrotnego uruchamiania algorytmu $k$-means dla różnych wartości $k$. Proces ten jest kosztowny obliczeniowo, mało efektywny i często prowadzi do przeszacowania optymalnej liczby klastrów.

---

### DBSCAN / HDBSCAN
Algorytmy te nie wymagają podania liczby klastrów, ale są bardzo czułe na parametry gęstości (`eps` oraz `min_samples`). Nawet niewielka zmiana tych parametrów może znacząco zmienić liczbę wykrytych klastrów.

### X-Means
Algorytmy te wykorzystują kryterium informacyjne BIC do automatycznego podziału klastrów, jednak nadal wymagają określenia odgórnych ograniczeń, takich jak $k_{max}$ i $k_{min}$. Dodatkowo wymaga to wielokrotnego trenowania modeli aż do uzyskania zbieżności.

---

## Przedstawienie nowego podejścia: K*-Means

Projekt implementuje algorytm $K^*$-Means oparty na pracy naukowej: *Louis Mahon, Mirella Lapata — „K*-Means: A Parameter-free Clustering Algorithm”.* Głównym celem algorytmu jest całkowite wyeliminowanie potrzeby ręcznego ustawiania liczby klastrów oraz innych parametrów konfiguracyjnych. Dzięki temu proces klasteryzacji staje się w pełni automatyczny (*parameter-free clustering*).

### Zasada Minimum Description Length (MDL)

Podstawą działania algorytmu jest zasada MDL (*Minimum Description Length*). Zgodnie z teorią MDL najlepszy model to taki, który pozwala opisać dane przy użyciu możliwie najmniejszej liczby bitów. W projekcie zastosowano następującą funkcję kosztu modelującą całkowitą długość opisu:

$$MDL = n \cdot d \cdot \log_2\left(\frac{SSE}{n \cdot d}\right) + k \cdot d \cdot \log_2(n) + n \cdot \log_2(k)$$

gdzie:
* $n$ — liczba punktów danych,
* $d$ — liczba wymiarów (cech),
* $k$ — liczba klastrów,
* $SSE$ — suma kwadratów błędów.

Funkcja kosztu MDL balansuje pomiędzy dwoma skrajnymi sytuacjami:
1. **Zbyt duża liczba klastrów ($k$):** Powoduje wzrost kosztu opisu samego modelu (*Model Cost*), ponieważ trzeba przechowywać większą liczbę centroidów oraz więcej informacji o przynależności punktów.
2. **Zbyt mała liczba klastrów ($k$):** Powoduje drastyczny wzrost błędu danych (*Residual Cost*), ponieważ punkty znajdują się daleko od swoich centroidów, co zwiększa wartość $SSE$.

Dzięki temu algorytm automatycznie znajduje idealny kompromis pomiędzy prostotą modelu a dokładnością klasteryzacji.

---

## Jak działa algorytm K*-Means?

W przeciwieństwie do klasycznych metod selekcji modeli, $K^*$-Means nie wykonuje kosztownego przeglądu wielu stałych wartości $k$. Algorytm uruchamiany jest tylko raz z początkową, minimalną liczbą klastrów:

$$k = 1$$

Podczas działania algorytm dynamicznie modyfikuje strukturę przestrzeni za pomocą dwóch naprzemiennych kroków:

* **Maybe-Split:** Algorytm sprawdza, czy próbny podział danego klastra na dwa podklastry za pomocą lokalnego $k$-means zmniejszy globalny koszt kosztu $MDL$. Jeśli warunek jest spełniony — klaster zostaje trwale podzielony.
* **Maybe-Merge:** Algorytm analizuje parę najbliższych klastrów. Jeśli ich fuzja i zastąpienie jednym wspólnym środkiem obniży lub utrzyma optymalny koszt $MDL$, klastry zostają scalone.

Dzięki temu liczba klastrów może dynamicznie rosnąć lub maleć podczas jednego, adaptacyjnego przebiegu algorytmu.

---

## Wyniki eksperymentów

W projekcie przeprowadzono eksperymenty na syntetycznych zbiorach danych o różnym poziomie separacji klastrów oraz na rzeczywistym zbiorze Iris.

| Scenariusz | Prawdziwe $k$ | Wykryte $k$ |
| :--- | :---: | :---: |
| **FAR** | $4$ | $4$ |
| **MEDIUM** | $4$ | $4$ |
| **CLOSE** | $4$ | $10$ |
| **IRIS** | $3$ | $2$ |

Scenariusze **FAR** oraz **MEDIUM** zostały idealnie rozpoznane przez algorytm. W scenariuszu **CLOSE** oraz zbiorze **IRIS** klastry silnie nakładały się na siebie w przestrzeni cech, dlatego kryterium $MDL$ dostosowało strukturę do lokalnych uwarunkowań gęstości, optymalizując długość opisu.

---

## Podsumowanie i zalety podejścia K*-Means

Projekt pokazuje, że algorytm $K^*$-Means skutecznie łączy zalety różnych podejść:
* **Brak parametrów:** Całkowity brak konieczności ręcznego wyboru liczby klastrów $k$ lub parametrów gęstości.
* **Efektywność:** Brak potrzeby wielokrotnego, iteracyjnego uruchamiania pełnych modeli dla różnych struktur.
* **Podstawa teoretyczna:** Wykorzystanie matematycznie ugruntowanej teorii informacji (kryterium $MDL$).
* **Dynamika:** Elastyczne dostosowywanie się do struktur dzięki lokalnym operacjom *split* i *merge*.
