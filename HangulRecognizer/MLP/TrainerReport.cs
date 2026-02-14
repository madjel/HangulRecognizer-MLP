using System.Collections.Generic;
using System.IO;

namespace HangulRecognizer.MLP
{
    public static class TrainerReport
    {
        public static void SaveAccuracyPerEpoch(
            List<(int epoch, float acc)> history,
            string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            using var sw = new StreamWriter(path);
            sw.WriteLine("epoch,accuracy");

            foreach (var h in history)
                sw.WriteLine($"{h.epoch},{h.acc:F2}");
        }
    }
}
