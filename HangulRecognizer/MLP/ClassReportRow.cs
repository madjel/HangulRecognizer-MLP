namespace HangulRecognizer.MLP
{
    public class ClassReportRow
    {
        public int Label { get; set; }
        public string Name { get; set; } = "";
        public int Correct { get; set; }
        public int Total { get; set; }

        public float Accuracy =>
            Total == 0 ? 0 : (float)Correct / Total * 100;
    }
}
