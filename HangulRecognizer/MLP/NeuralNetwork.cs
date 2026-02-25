using System;
using System.Linq;
using System.Collections.Generic;

namespace HangulRecognizer.MLP
{
    public class NeuralNetwork
    {
        private int inputSize, hiddenSize, outputSize;

        private float[,] weightsIH;
        private float[,] weightsHO;

        private float[] biasH;
        private float[] biasO;

        private float[] hidden;
        private float[] hiddenZ;
        private float[] output;

        private Random rnd = new();

        //CALLBACK DO UI (epoch, accuracy)
        public Action<int, float>? OnEpochEnd;

        public NeuralNetwork(int input, int hidden, int output)
        {
            inputSize = input;
            hiddenSize = hidden;
            outputSize = output;

            weightsIH = new float[inputSize, hiddenSize];
            weightsHO = new float[hiddenSize, outputSize];

            biasH = new float[hiddenSize];
            biasO = new float[outputSize];

            this.hidden = new float[hiddenSize];
            this.hiddenZ = new float[hiddenSize];
            this.output = new float[outputSize];

            InitWeights();
        }

        //INIT

        private void InitWeights()
        {
            float sIH = (float)Math.Sqrt(2.0 / inputSize);
            float sHO = (float)Math.Sqrt(2.0 / hiddenSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    weightsIH[i, j] = ((float)rnd.NextDouble() * 2 - 1) * sIH;

            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < outputSize; j++)
                    weightsHO[i, j] = ((float)rnd.NextDouble() * 2 - 1) * sHO;

            for (int i = 0; i < hiddenSize; i++)
                biasH[i] = 0f;

            for (int i = 0; i < outputSize; i++)
                biasO[i] = 0f;
        }

        //ACTIVATIONS

        // Funkcja aktywacji ReLU: f(x) = max(0, x)
        private float ReLU(float x) => x > 0 ? x : 0;

        // Pochodna ReLU używana w backpropagation:
        // f'(x) = 1 dla x > 0
        // f'(x) = 0 dla x <= 0
        private float dReLU(float x) => x > 0 ? 1f : 0f;   

        private float[] Softmax(float[] z)
        {
            float max = z.Max();
            float sum = 0f;

            float[] exp = new float[z.Length];
            for (int i = 0; i < z.Length; i++)
            {
                exp[i] = (float)Math.Exp(z[i] - max);
                sum += exp[i];
            }

            for (int i = 0; i < z.Length; i++)
                exp[i] /= sum;

            return exp;
        }

        //FORWARD

        private float[] Forward(float[] x)
        {
            // ===== WARSTWA UKRYTA =====
            // Obliczenie: z¹ = W¹x + b¹
            for (int j = 0; j < hiddenSize; j++)
            {
                float sum = biasH[j]; // dodanie biasu b¹
                for (int i = 0; i < inputSize; i++) 
                    sum += x[i] * weightsIH[i, j]; // mnożenie macierz-wektor (W¹x)

                hiddenZ[j] = sum; // zapis wartości przed aktywacją (z¹)
                hidden[j] = ReLU(sum); // funkcja aktywacji ReLU: a¹ = ReLU(z¹)
            }

            // ===== WARSTWA WYJŚCIOWA =====
            // Obliczenie: z² = W²a¹ + b²
            for (int j = 0; j < outputSize; j++)
            {
                float sum = biasO[j]; // dodanie biasu b²
                for (int i = 0; i < hiddenSize; i++)
                    sum += hidden[i] * weightsHO[i, j]; // W²a¹

                output[j] = sum; // logity (z²)
            }

            // ===== FUNKCJA SOFTMAX =====
            // Zamiana logitów na prawdopodobieństwa klas
            return Softmax(output); // ŷ = softmax(z²)
        }

        //TRAIN

        public void Train(
            List<float[]> X,
            List<int> y,
            int epochs,
            float lr)
        {
            if (X.Count != y.Count)
                throw new ArgumentException("X and y size mismatch");

            var indices = Enumerable.Range(0, X.Count).ToArray();

            for (int ep = 0; ep < epochs; ep++)
            {
                Shuffle(indices);

                int correct = 0;

                foreach (int idx in indices)
                {
                    var probs = Forward(X[idx]);

                    int pred = ArgMax(probs);
                    if (pred == y[idx]) correct++;

                    // ===== OBLICZENIE GRADIENTU DLA SOFTMAX + CROSS ENTROPY =====

                    // kopiujemy przewidywane prawdopodobieństwa
                    float[] gradOut = new float[outputSize];
                    for (int j = 0; j < outputSize; j++)
                        gradOut[j] = probs[j];
                    
                    // odejmujemy 1 dla prawdziwej klasy
                    gradOut[y[idx]] -= 1f;

                    // To jest uproszczony gradient funkcji:
                    // L = - Σ y log(ŷ)
                    // dla softmax + cross entropy: dL/dz = ŷ - y


                    // ===== BACKPROPAGATION =====
                    // Obliczenie gradientu dla warstwy ukrytej

                    float[] gradH = new float[hiddenSize];
                    for (int i = 0; i < hiddenSize; i++)
                    {
                        float sum = 0;
                        for (int j = 0; j < outputSize; j++)
                            sum += gradOut[j] * weightsHO[i, j];

                        // zastosowanie pochodnej ReLU
                        gradH[i] = sum * dReLU(hiddenZ[i]);
                    }

                    // ===== AKTUALIZACJA WAG (SGD) =====

                    // aktualizacja wag między warstwą ukrytą a wyjściową
                    for (int i = 0; i < hiddenSize; i++)
                        for (int j = 0; j < outputSize; j++)
                            weightsHO[i, j] -= lr * gradOut[j] * hidden[i];

                    // aktualizacja wag między wejściem a warstwą ukrytą
                    for (int i = 0; i < inputSize; i++)
                        for (int j = 0; j < hiddenSize; j++)
                            weightsIH[i, j] -= lr * gradH[j] * X[idx][i];

                    // aktualizacja biasów
                    for (int j = 0; j < outputSize; j++)
                        biasO[j] -= lr * gradOut[j] * 0.1f;

                    for (int j = 0; j < hiddenSize; j++)
                        biasH[j] -= lr * gradH[j] * 0.1f;
                }

                float acc = (float)correct / X.Count * 100f;
                OnEpochEnd?.Invoke(ep, acc);

                if (acc > 99f)
                    break;
            }
        }

        //PREDICTION

        public int Predict(float[] x)
        {
            var p = Forward(x);
            return ArgMax(p);
        }

        private int ArgMax(float[] v)
        {
            int idx = 0;
            float max = v[0];

            for (int i = 1; i < v.Length; i++)
            {
                if (v[i] > max)
                {
                    max = v[i];
                    idx = i;
                }
            }

            return idx;
        }

        private void Shuffle(int[] arr)
        {
            for (int i = arr.Length - 1; i > 0; i--)
            {
                int j = rnd.Next(i + 1);
                (arr[i], arr[j]) = (arr[j], arr[i]);
            }
        }

        //SERIALIZATION

        public NeuralNetworkModel Export() => new()
        {
            InputSize = inputSize,
            HiddenSize = hiddenSize,
            OutputSize = outputSize,
            WeightsIH = MatrixUtils.Flatten(weightsIH),
            WeightsHO = MatrixUtils.Flatten(weightsHO),
            BiasH = biasH.ToArray(),
            BiasO = biasO.ToArray()
        };

        public static NeuralNetwork FromModel(NeuralNetworkModel m)
        {
            var nn = new NeuralNetwork(
                m.InputSize,
                m.HiddenSize,
                m.OutputSize);

            nn.weightsIH = MatrixUtils.Restore(
                m.WeightsIH,
                m.InputSize,
                m.HiddenSize);

            nn.weightsHO = MatrixUtils.Restore(
                m.WeightsHO,
                m.HiddenSize,
                m.OutputSize);

            nn.biasH = m.BiasH.ToArray();
            nn.biasO = m.BiasO.ToArray();

            return nn;
        }

        //TOP 3
        public float[] PredictProbs(float[] x)
        {
            return Forward(x);
        }
    }
}
