using System;
using System.IO;
using System.Linq;
using System.Windows.Media.Imaging;

namespace HangulRecognizer.MLP
{
    public static class ImageProcessor
    {
        public static float[] Process(string path, int size = 64)
        {
            BitmapImage bmp = new();
            bmp.BeginInit();
            bmp.UriSource = new Uri(Path.GetFullPath(path), UriKind.Absolute);
            bmp.DecodePixelWidth = size;
            bmp.DecodePixelHeight = size;
            bmp.EndInit();

            WriteableBitmap wb = new(bmp);
            byte[] px = new byte[size * size * 4];
            wb.CopyPixels(px, size * 4, 0);

            float[] v = new float[size * size];
            int k = 0;

            //1️⃣ RGB → GRAYSCALE [0..1]
            for (int i = 0; i < px.Length; i += 4)
            {
                v[k++] = 1f - (px[i] + px[i + 1] + px[i + 2]) / 765f;
            }

            //2️⃣ NORMALIZACJA (KRYTYCZNE)
            float mean = v.Average();
            float std = (float)Math.Sqrt(
                v.Select(x => (x - mean) * (x - mean)).Average()
                + 1e-6f
            );

            for (int i = 0; i < v.Length; i++)
                v[i] = (v[i] - mean) / std;

            return v;
        }
    }
}
