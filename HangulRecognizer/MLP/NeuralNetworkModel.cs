namespace HangulRecognizer.MLP
{
    public class NeuralNetworkModel
    {
        public int InputSize { get; set; }
        public int HiddenSize { get; set; }
        public int OutputSize { get; set; }

        //WAGI
        public float[] WeightsIH { get; set; } = null!;
        public float[] WeightsHO { get; set; } = null!;

        //BIASY (NOWE)
        public float[] BiasH { get; set; } = null!;
        public float[] BiasO { get; set; } = null!;
    }
}
