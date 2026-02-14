using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace HangulRecognizer.MLP
{
    public static class ClassificationReport
    {
        public static void Save(
            NeuralNetwork nn,
            List<(float[] x, int y)> data,
            string path)
        {
            var rows = new Dictionary<int, ClassReportRow>();

            foreach (var (x, y) in data)
            {
                if (!rows.ContainsKey(y))
                {
                    rows[y] = new ClassReportRow
                    {
                        Label = y,
                        Name = LabelMap.Name(y)
                    };
                }

                rows[y].Total++;

                if (nn.Predict(x) == y)
                    rows[y].Correct++;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            using var sw = new StreamWriter(path);
            sw.WriteLine("label,name,correct,total,accuracy");

            foreach (var r in rows.Values.OrderBy(r => r.Label))
            {
                sw.WriteLine(
                    $"{r.Label},{r.Name},{r.Correct},{r.Total},{r.Accuracy:F0}");
            }
        }
    }
}
