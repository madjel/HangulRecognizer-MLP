using System.Collections.Generic;
using System.IO;

namespace HangulRecognizer.MLP
{
    public static class DataLoader
    {
        private static readonly Dictionary<string, int> labels = new()
        {
            {"giyeok",0},{"nieun",1},{"digeut",2},{"rieul",3},
            {"mieum",4},{"bieup",5},{"siot",6},{"ieung",7},
            {"jieut",8},{"chieut",9},{"kieuk",10},{"tieut",11},
            {"pieup",12},{"hieut",13},{"a",14},{"ya",15},
            {"eo",16},{"yeo",17},{"o",18},{"yo",19},
            {"u",20},{"yu",21},{"eu",22},{"i",23}
        };

        //ZWRACAMY RÓWNIEŻ NAZWĘ PLIKU
        public static List<(float[] x, int y, string file)> Load(string root)
        {
            var data = new List<(float[], int, string)>();

            foreach (var dir in Directory.GetDirectories(root))
            {
                string name = Path.GetFileName(dir).ToLower();
                if (!labels.ContainsKey(name)) continue;

                int label = labels[name];

                foreach (var img in Directory.GetFiles(dir, "*.png"))
                {
                    data.Add((
                        ImageProcessor.Process(img),
                        label,
                        Path.GetFileName(img)
                    ));
                }
            }

            return data;
        }
    }
}
