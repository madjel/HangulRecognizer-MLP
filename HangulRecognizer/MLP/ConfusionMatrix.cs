using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace HangulRecognizer.MLP
{
    public static class ConfusionMatrix
    {
        public static int[,] Compute(
            NeuralNetwork nn,
            List<(float[] x, int y)> data,
            int classCount)
        {
            var matrix = new int[classCount, classCount];

            foreach (var (x, y) in data)
            {
                int pred = nn.Predict(x);
                matrix[y, pred]++;
            }

            return matrix;
        }

        public static void SaveCsv(
            int[,] matrix,
            string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            int n = matrix.GetLength(0);

            using var sw = new StreamWriter(path);

            //HEADER
            sw.Write("actual/pred");
            for (int i = 0; i < n; i++)
                sw.Write($",P{i}");
            sw.WriteLine();

            //ROWS
            for (int i = 0; i < n; i++)
            {
                sw.Write($"A{i}");
                for (int j = 0; j < n; j++)
                    sw.Write($",{matrix[i, j]}");
                sw.WriteLine();
            }
        }
    }
}
