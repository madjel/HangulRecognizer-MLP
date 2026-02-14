using System.Collections.Generic;

namespace HangulRecognizer.MLP
{
    public static class LabelMap
    {
        private static readonly Dictionary<int, string> map = new()
        {
            {0,"giyeok"},{1,"nieun"},{2,"digeut"},{3,"rieul"},
            {4,"mieum"},{5,"bieup"},{6,"siot"},{7,"ieung"},
            {8,"jieut"},{9,"chieut"},{10,"kieuk"},{11,"tieut"},
            {12,"pieup"},{13,"hieut"},{14,"a"},{15,"ya"},
            {16,"eo"},{17,"yeo"},{18,"o"},{19,"yo"},
            {20,"u"},{21,"yu"},{22,"eu"},{23,"i"}
        };

        public static string Name(int label)
            => map.TryGetValue(label, out var n) ? n : "unknown";
    }
}
