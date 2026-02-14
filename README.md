#Hangul Recognizer (MLP od podstaw)

Desktopowa aplikacja WPF do rozpoznawania znaków alfabetu koreańskiego (Hangul) przy użyciu własnoręcznie zaimplementowanej sieci neuronowej typu Multi-Layer Perceptron (MLP).

Projekt został napisany w czystym C# bez użycia bibliotek ML (np. ML.NET, TensorFlow, itp.).

---

##Funkcjonalności

- Własna implementacja MLP (4096 → 128 → 24)
- Obsługa biasów (warstwa ukryta i wyjściowa)
- Funkcja aktywacji ReLU
- Warstwa wyjściowa Softmax
- Zapisywanie i wczytywanie modelu (JSON)
- Wykres accuracy podczas treningu
- Macierz pomyłek (Confusion Matrix)
- Raport klasyfikacji (precision / recall / F1)
- Top-3 przewidywane klasy z prawdopodobieństwem
- Automatyczne tworzenie folderów:
  - `Models`
  - `Reports`
- Nowoczesny ciemny interfejs (WPF)

---

##Architektura

Projekt podzielony jest na logiczne warstwy:

- `MLP/NeuralNetwork.cs` – implementacja sieci neuronowej
- `MLP/Trainer.cs` – logika treningu
- `ImageProcessor.cs` – preprocessing obrazu (64x64 → wektor 4096)
- `ConfusionMatrix.cs` – generowanie macierzy pomyłek
- `ClassificationReport.cs` – raport jakości klasyfikacji
- `MainWindow.xaml` – warstwa UI

---

##Technologie

- .NET 8
- WPF
- C#
- Brak zewnętrznych bibliotek ML (pełna implementacja własna)

---

##Uruchomienie

Build w trybie Release:

dotnet publish -c Release -r win-x64 --self-contained true /p:PublishSingleFile=true

Plik `.exe` znajdziesz w:

bin/Release/net8.0-windows/win-x64/publish/

---

## Autor

Filip Masłowski