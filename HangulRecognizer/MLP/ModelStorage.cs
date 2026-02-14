using System.IO;
using System.Text.Json;
using HangulRecognizer.MLP;

public static class ModelStorage
{
    public static void Save(string path, NeuralNetwork nn)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);

        var model = nn.Export();
        var json = JsonSerializer.Serialize(
            model,
            new JsonSerializerOptions { WriteIndented = true });

        File.WriteAllText(path, json);
    }

    public static NeuralNetwork Load(string path)
    {
        var json = File.ReadAllText(path);
        var model = JsonSerializer.Deserialize<NeuralNetworkModel>(json)!;
        return NeuralNetwork.FromModel(model);
    }
}
