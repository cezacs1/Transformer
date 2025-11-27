var model = new ExelentTransformer(sequenceLength: 5, featureDim: 4, embedDim: 16, numHeads: 4);

// eğitim verisi - dataset
var data = new GameSequence[]
{
    new GameSequence
    {
        frames = new GameFrame[]
        {
            new GameFrame { features = new float[] { 0.5f, 0.3f, 0f, 0f } },
            new GameFrame { features = new float[] { 0.4f, 0.2f, 0f, 0f } },
            new GameFrame { features = new float[] { 0.3f, 0.1f, 0f, 0f } },
            new GameFrame { features = new float[] { 0.2f, 0.05f, 0f, 0f } },
            new GameFrame { features = new float[] { 0.1f, 0.02f, 0f, 0f } }
        },
        target = new float[] { -1f, -1f } 
    }
};

// Model eğitimi
model.Train(data, epochs: 100, learningRate: 0.001f);

// Test
var test = data[0].frames;
var tahmin = model.Predict(test);

Console.WriteLine($"\nTahmin: X={tahmin[0]:F2}, Y={tahmin[1]:F2}");
