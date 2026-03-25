using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Shapes;
using HangulRecognizer.MLP;

namespace HangulRecognizer
{
    public partial class MainWindow : Window
    {
        private NeuralNetwork? nn;
        private const string ModelDir = "Models";

        public MainWindow()
        {
            InitializeComponent();
            LoadModelList();
        }

        //WCZYTANIE LISTY MODELI
        private void LoadModelList()
        {
            ModelComboBox.Items.Clear();

            if (!Directory.Exists(ModelDir))
                Directory.CreateDirectory(ModelDir);

            var files = Directory.GetFiles(ModelDir, "*.json");
            foreach (var f in files)
                ModelComboBox.Items.Add(System.IO.Path.GetFileName(f));

            if (ModelComboBox.Items.Count > 0)
                ModelComboBox.SelectedIndex = 0;
        }

        //ZAŁADUJ MODEL
        private void LoadModel_Click(object sender, RoutedEventArgs e)
        {
            if (ModelComboBox.SelectedItem == null)
            {
                StatusText.Text = "Wybierz model z listy";
                return;
            }

            string file = ModelComboBox.SelectedItem.ToString()!;
            string path = System.IO.Path.Combine(ModelDir, file);

            nn = ModelStorage.Load(path);
            StatusText.Text = $"Załadowano model: {file}";
        }

        //TRENING
        private async void Train_Click(object sender, RoutedEventArgs e)
        {
            StatusText.Text = "Trening...";
            AccuracyCanvas.Children.Clear();
            TrainingProgress.Value = 0;

            await Task.Run(() =>
            {
                var rawData = DataLoader.Load("Data/train");

                //KONWERSJA DO (float[] x, int y)
                var data = rawData
                    .Select(d => (d.x, d.y))
                    .ToList();

                nn = new NeuralNetwork(4096, 128, 24);

                var history = new List<(int epoch, float acc)>();
                int totalEpochs = 200;

                nn.OnEpochEnd = (ep, acc) =>
                {
                    history.Add((ep, acc));

                    Dispatcher.Invoke(() =>
                    {
                        DrawAccuracy(history.Select(h => h.acc).ToList());
                        StatusText.Text =
                            $"Epoka {ep + 1}/{totalEpochs}, acc {acc:F1}%";
                        TrainingProgress.Value =
                            (ep + 1) * 100.0 / totalEpochs;
                    });
                };

                nn.Train(
                    data.Select(d => d.x).ToList(),
                    data.Select(d => d.y).ToList(),
                    totalEpochs,
                    0.0001f
                );

                //ZAPIS MODELU
                ModelStorage.Save("Models/mlp.json", nn);

                //RAPORTY
                Directory.CreateDirectory("Reports");

                TrainerReport.SaveAccuracyPerEpoch(
                    history,
                    "Reports/training_accuracy.csv");

                ClassificationReport.Save(
                    nn,
                    data,
                    "Reports/classification_train.csv");

                var cm = ConfusionMatrix.Compute(
                    nn,
                    data,
                    24);

                ConfusionMatrix.SaveCsv(
                    cm,
                    "Reports/confusion_matrix_train.csv");       
            });

            LoadModelList();
            StatusText.Text = "Trening zakończony + wygenerowano raporty";
        }


        //LOSOWA PRÓBKA
        private async void RandomSample_Click(object sender, RoutedEventArgs e)
        {
            if (nn == null)
            {
                StatusText.Text = "Brak modelu";
                return;
            }

            StatusText.Text = "Predykcja...";

            var data = DataLoader.Load("Data/train");
            var rnd = new Random();
            var s = data[rnd.Next(data.Count)];

            int runs = 100;

            var result = await Task.Run(() =>
            {
                var watch = Stopwatch.StartNew();

                float[] probs = null!;
                for (int i = 0; i < runs; i++)
                {
                    probs = nn.PredictProbs(s.x);
                }

                watch.Stop();
                double avgTime = watch.Elapsed.TotalMilliseconds / runs;

                var top3 = probs
                    .Select((v, i) => new { Index = i, Value = v })
                    .OrderByDescending(x => x.Value)
                    .Take(3)
                    .ToList();

                string top3Text = $"Plik: {s.file}\n" +
                                  $"Oczekiwane: {LabelMap.Name(s.y)}\n\n" +
                                  $"TOP 3:\n";

                int rank = 1;
                foreach (var item in top3)
                {
                    top3Text += $"{rank}. {LabelMap.Name(item.Index)} ({item.Value * 100:F2}%)\n";
                    rank++;
                }

                return (top3Text, avgTime);
            });

            Dispatcher.Invoke(() =>
            {
                PredictionText.Text = result.top3Text;
                PredictionTimeText.Text = $"Średni czas predykcji: {result.avgTime:F6} ms";
                StatusText.Text = "Gotowe";
            });
        }


        //WYKRES
        private void DrawAccuracy(List<float> history)
        {
            AccuracyCanvas.Children.Clear();
            if (history.Count < 2) return;

            double w = AccuracyCanvas.ActualWidth;
            double h = AccuracyCanvas.ActualHeight;

            var line = new Polyline
            {
                Stroke = Brushes.LimeGreen,
                StrokeThickness = 2
            };

            for (int i = 0; i < history.Count; i++)
            {
                double x = i * (w / (history.Count - 1));
                double y = h - history[i] / 100.0 * h;
                line.Points.Add(new Point(x, y));
            }

            AccuracyCanvas.Children.Add(line);
        }


        private void About_Click(object sender, RoutedEventArgs e)
        {
            string msg =
                "Hangul Recognizer\n" +
                "Wersja: 1.0\n\n" +
                "Prosty klasyfikator znaków Hangul\n" +
                "Sieć: MLP (4096 → 128 → 24)\n\n" +
                "Instrukcja:\n" +
                "1. Załaduj model lub wytrenuj nowy\n" +
                "2. Kliknij 'Losowa próbka'\n" +
                "3. Zobacz TOP-3 predykcje\n" +
                "4. Sprawdź raporty z treningu\n\n" +
                "Autor: Filip Masłowski";

            MessageBox.Show(msg, "O aplikacji");
        }


        private void OpenReports_Click(object sender, RoutedEventArgs e)
        {
            string path = System.IO.Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Reports");

            if (!Directory.Exists(path))
            {
                MessageBox.Show(
                    "Brak raportów. Najpierw wytrenuj model.",
                    "Raporty");
                return;
            }

            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = path,
                UseShellExecute = true
            });
        }
    }
}
