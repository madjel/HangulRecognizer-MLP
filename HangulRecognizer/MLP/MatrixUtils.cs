namespace HangulRecognizer.MLP
{
    public static class MatrixUtils
    {
        public static float[] Flatten(float[,] m)
        {
            int rows = m.GetLength(0);
            int cols = m.GetLength(1);

            float[] v = new float[rows * cols];
            int k = 0;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    v[k++] = m[i, j];

            return v;
        }

        public static float[,] Restore(float[] v, int rows, int cols)
        {
            float[,] m = new float[rows, cols];
            int k = 0;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = v[k++];

            return m;
        }
    }
}
