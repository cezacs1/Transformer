using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace Exelent.NeuralNetworks
{
    public class GameFrame
    {
        public float[] features;
    }

    public class GameSequence
    {
        public GameFrame[] frames;
        public float[] target;
    }

    public class ExelentTransformer
    {
        private int sequenceLength;
        private int featureDim;
        private int embedDim;
        private int numHeads;
        private int headDim;
        private int mlpHiddenDim;
        private int outputDim;

        private float[,] featureEmbedding;
        private float[,] WQ, WK, WV, WO;
        private float[,] W1, W2;
        private float[] b1, b2;
        private float[,] posEmbeddings;
        private float[,] outputWeights;
        private float[] outputBias;

        private Random random = new Random(42);

        public ExelentTransformer(int sequenceLength, int featureDim, int embedDim, int numHeads = 4)
        {
            this.sequenceLength = sequenceLength;
            this.featureDim = featureDim;
            this.embedDim = embedDim;
            this.numHeads = numHeads;
            this.headDim = embedDim / numHeads;
            this.mlpHiddenDim = embedDim * 4;
            this.outputDim = 2;

            if (embedDim % numHeads != 0)
                throw new ArgumentException($"embedDim ({embedDim}) must be divisible by numHeads ({numHeads})");

            featureEmbedding = InitializeMatrix(featureDim, embedDim);

            WQ = InitializeMatrix(embedDim, embedDim);
            WK = InitializeMatrix(embedDim, embedDim);
            WV = InitializeMatrix(embedDim, embedDim);
            WO = InitializeMatrix(embedDim, embedDim);

            W1 = InitializeMatrix(embedDim, mlpHiddenDim);
            b1 = new float[mlpHiddenDim];
            W2 = InitializeMatrix(mlpHiddenDim, embedDim);
            b2 = new float[embedDim];

            posEmbeddings = InitializeMatrix(sequenceLength, embedDim);
            outputWeights = InitializeMatrix(embedDim, outputDim);
            outputBias = new float[outputDim];
        }

        public void SaveWeights(string filepath)
        {
            try
            {
                using (var writer = new BinaryWriter(File.Open(filepath, FileMode.Create)))
                {
                    writer.Write(sequenceLength);
                    writer.Write(featureDim);
                    writer.Write(embedDim);
                    writer.Write(numHeads);
                    writer.Write(headDim);
                    writer.Write(mlpHiddenDim);
                    writer.Write(outputDim);

                    WriteMatrix(writer, featureEmbedding);
                    WriteMatrix(writer, WQ);
                    WriteMatrix(writer, WK);
                    WriteMatrix(writer, WV);
                    WriteMatrix(writer, WO);
                    WriteMatrix(writer, W1);
                    WriteVector(writer, b1);
                    WriteMatrix(writer, W2);
                    WriteVector(writer, b2);
                    WriteMatrix(writer, posEmbeddings);
                    WriteMatrix(writer, outputWeights);
                    WriteVector(writer, outputBias);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Hata: Model kaydedilemedi - {ex.Message}");
            }
        }

        public void LoadWeights(string filepath)
        {
            try
            {
                using (var reader = new BinaryReader(File.Open(filepath, FileMode.Open)))
                {
                    int loadedSeqLen = reader.ReadInt32();
                    int loadedFeatureDim = reader.ReadInt32();
                    int loadedEmbedDim = reader.ReadInt32();
                    int loadedNumHeads = reader.ReadInt32();
                    int loadedHeadDim = reader.ReadInt32();
                    int loadedMlpDim = reader.ReadInt32();
                    int loadedOutputDim = reader.ReadInt32();

                    if (loadedSeqLen != sequenceLength || loadedFeatureDim != featureDim ||
                        loadedEmbedDim != embedDim || loadedNumHeads != numHeads ||
                        loadedHeadDim != headDim || loadedMlpDim != mlpHiddenDim ||
                        loadedOutputDim != outputDim)
                    {
                        throw new Exception("Model mimarisi uyuşmuyor!");
                    }

                    featureEmbedding = ReadMatrix(reader);
                    WQ = ReadMatrix(reader);
                    WK = ReadMatrix(reader);
                    WV = ReadMatrix(reader);
                    WO = ReadMatrix(reader);
                    W1 = ReadMatrix(reader);
                    b1 = ReadVector(reader);
                    W2 = ReadMatrix(reader);
                    b2 = ReadVector(reader);
                    posEmbeddings = ReadMatrix(reader);
                    outputWeights = ReadMatrix(reader);
                    outputBias = ReadVector(reader);
                }

                Console.WriteLine("✓ Model başarıyla yüklendi!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Hata: Model yüklenemedi - {ex.Message}");
                throw;
            }
        }

        private void WriteMatrix(BinaryWriter writer, float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            writer.Write(rows);
            writer.Write(cols);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    writer.Write(matrix[i, j]);
        }

        private float[,] ReadMatrix(BinaryReader reader)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            float[,] matrix = new float[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = reader.ReadSingle();

            return matrix;
        }

        private void WriteVector(BinaryWriter writer, float[] vector)
        {
            writer.Write(vector.Length);
            foreach (float v in vector)
                writer.Write(v);
        }

        private float[] ReadVector(BinaryReader reader)
        {
            int length = reader.ReadInt32();
            float[] vector = new float[length];
            for (int i = 0; i < length; i++)
                vector[i] = reader.ReadSingle();
            return vector;
        }

        private float[,] InitializeMatrix(int rows, int cols)
        {
            float[,] matrix = new float[rows, cols];
            float scale = (float)Math.Sqrt(2.0 / (rows + cols));
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = (float)(random.NextDouble() * 2 - 1) * scale;
            return matrix;
        }

        private float[] InitializeVector(int size)
        {
            float[] vec = new float[size];
            float scale = (float)Math.Sqrt(2.0 / size);
            for (int i = 0; i < size; i++)
                vec[i] = (float)(random.NextDouble() * 2 - 1) * scale;
            return vec;
        }

        private (float[,] output, float[,] embedded, AttentionCache attnCache, MLPCache mlpCache) Forward(GameFrame[] sequence)
        {
            float[,] embedded = new float[sequenceLength, embedDim];

            for (int i = 0; i < sequenceLength; i++)
            {
                for (int f = 0; f < featureDim; f++)
                {
                    float featureValue = sequence[i].features[f];
                    for (int d = 0; d < embedDim; d++)
                    {
                        embedded[i, d] += featureValue * featureEmbedding[f, d];
                    }
                }
            }

            float[,] XWithPos = AddPositionalEmbeddings(embedded);
            var (attnOutput, attnCache) = SelfAttention(XWithPos);
            float[,] afterAttn = AddResidual(XWithPos, attnOutput);
            var (mlpOutput, mlpCache) = MLP(afterAttn);
            float[,] output = AddResidual(afterAttn, mlpOutput);

            return (output, embedded, attnCache, mlpCache);
        }

        public float[] Predict(GameFrame[] sequence)
        {
            var (output, _, _, _) = Forward(sequence);

            float[] lastFrame = new float[embedDim];
            for (int i = 0; i < embedDim; i++)
                lastFrame[i] = output[sequenceLength - 1, i];

            float[] predictions = new float[outputDim];
            for (int o = 0; o < outputDim; o++)
            {
                float logit = outputBias[o];
                for (int i = 0; i < embedDim; i++)
                    logit += lastFrame[i] * outputWeights[i, o];

                predictions[o] = Tanh(logit);
            }

            return predictions;
        }

        public void Train(GameSequence[] data, int epochs, float learningRate)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float totalLoss = 0;

                foreach (var sequence in data)
                {
                    var (output, embedded, attnCache, mlpCache) = Forward(sequence.frames);
                    float[] predictions = Predict(sequence.frames);

                    float[] errors = new float[outputDim];
                    float loss = 0;
                    for (int o = 0; o < outputDim; o++)
                    {
                        errors[o] = predictions[o] - sequence.target[o];
                        loss += errors[o] * errors[o];
                    }
                    totalLoss += loss;

                    Backprop(sequence.frames, embedded, output, attnCache, mlpCache,
                            errors, predictions, learningRate);
                }

                Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss / data.Length:F6}");
            }
        }

        private void Backprop(GameFrame[] sequence, float[,] embedded, float[,] output,
                              AttentionCache attnCache, MLPCache mlpCache,
                              float[] errors, float[] predictions, float lr)
        {
            int seqLen = sequenceLength;
            float clipValue = 1.0f;

            float[,] dOutput = new float[seqLen, embedDim];

            for (int o = 0; o < outputDim; o++)
            {
                float grad = 2 * errors[o] * (1 - predictions[o] * predictions[o]);

                for (int i = 0; i < embedDim; i++)
                {
                    float g = grad * outputWeights[i, o];
                    g = Math.Max(-clipValue, Math.Min(clipValue, g));
                    dOutput[seqLen - 1, i] += g;
                }
            }

            for (int o = 0; o < outputDim; o++)
            {
                float lastFrameGrad = 2 * errors[o] * (1 - predictions[o] * predictions[o]);

                for (int i = 0; i < embedDim; i++)
                {
                    float lastFrameValue = output[seqLen - 1, i];
                    float dWeight = lastFrameGrad * lastFrameValue;
                    dWeight = Math.Max(-clipValue, Math.Min(clipValue, dWeight));
                    outputWeights[i, o] -= lr * dWeight;
                }

                float dBias = lastFrameGrad;
                dBias = Math.Max(-clipValue, Math.Min(clipValue, dBias));
                outputBias[o] -= lr * dBias;
            }

            float[,] dMLPOutput = CopyMatrix(dOutput);
            float[,] dAfterAttn = CopyMatrix(dOutput);
            BackpropMLP(mlpCache, dMLPOutput, lr, clipValue);

            float[,] dAttnOutput = CopyMatrix(dAfterAttn);
            float[,] dInputFromAttn = BackpropAttention(attnCache, dAttnOutput, lr, clipValue);

            float[,] dXWithPos = new float[seqLen, embedDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embedDim; j++)
                    dXWithPos[i, j] = dInputFromAttn[i, j] + dAfterAttn[i, j];

            UpdatePositionalEmbeddings(dXWithPos, lr, clipValue);
            BackpropFeatureEmbedding(sequence, dXWithPos, lr, clipValue);
        }

        private void BackpropFeatureEmbedding(GameFrame[] sequence, float[,] grads, float lr, float clip)
        {
            for (int i = 0; i < sequenceLength; i++)
            {
                for (int f = 0; f < featureDim; f++)
                {
                    float featureValue = sequence[i].features[f];
                    for (int d = 0; d < embedDim; d++)
                    {
                        float g = grads[i, d] * featureValue;
                        g = Math.Max(-clip, Math.Min(clip, g));
                        featureEmbedding[f, d] -= lr * g;
                    }
                }
            }
        }

        private float[,] AddPositionalEmbeddings(float[,] X)
        {
            int seqLen = X.GetLength(0);
            int dim = X.GetLength(1);
            float[,] result = new float[seqLen, dim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < dim; j++)
                    result[i, j] = X[i, j] + posEmbeddings[i, j];
            return result;
        }

        private (float[,] output, AttentionCache cache) SelfAttention(float[,] X)
        {
            int seqLen = X.GetLength(0);

            float[,] Q = MatMul(X, WQ);
            float[,] K = MatMul(X, WK);
            float[,] V = MatMul(X, WV);

            float[,,] Q_heads = Reshape3D(Q, seqLen, numHeads, headDim);
            float[,,] K_heads = Reshape3D(K, seqLen, numHeads, headDim);
            float[,,] V_heads = Reshape3D(V, seqLen, numHeads, headDim);

            float[,,] attn_outputs = new float[seqLen, numHeads, headDim];
            float[,,] attention_weights = new float[numHeads, seqLen, seqLen];

            for (int h = 0; h < numHeads; h++)
            {
                float[,] Q_h = GetHeadSlice(Q_heads, h);
                float[,] K_h = GetHeadSlice(K_heads, h);
                float[,] V_h = GetHeadSlice(V_heads, h);

                float[,] scores = MatMul(Q_h, Transpose(K_h));
                float scale = (float)Math.Sqrt(headDim);

                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        scores[i, j] /= scale;

                float[,] attn = Softmax(scores);

                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        attention_weights[h, i, j] = attn[i, j];

                float[,] attn_out = MatMul(attn, V_h);

                for (int i = 0; i < seqLen; i++)
                    for (int d = 0; d < headDim; d++)
                        attn_outputs[i, h, d] = attn_out[i, d];
            }

            float[,] concat = ReshapeBack2D(attn_outputs, seqLen, embedDim);
            float[,] output = MatMul(concat, WO);

            var cache = new AttentionCache
            {
                X = X,
                Q = Q,
                K = K,
                V = V,
                Q_heads = Q_heads,
                K_heads = K_heads,
                V_heads = V_heads,
                AttentionWeights = attention_weights,
                ConcatOutput = concat,
                Scale = (float)Math.Sqrt(headDim)
            };

            return (output, cache);
        }

        private (float[,] output, MLPCache cache) MLP(float[,] X)
        {
            int seqLen = X.GetLength(0);

            float[,] hidden = MatMul(X, W1);
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < mlpHiddenDim; j++)
                    hidden[i, j] += b1[j];

            float[,] activated = ReLU(hidden);

            float[,] output = MatMul(activated, W2);
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embedDim; j++)
                    output[i, j] += b2[j];

            return (output, new MLPCache { X = X, Hidden = hidden, Activated = activated });
        }

        private void BackpropMLP(MLPCache cache, float[,] dOutput, float lr, float clip)
        {
            int seqLen = cache.X.GetLength(0);

            float[,] dW2 = MatMul(Transpose(cache.Activated), dOutput);
            float[,] dActivated = MatMul(dOutput, Transpose(W2));
            float[,] dHidden = new float[seqLen, mlpHiddenDim];

            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < mlpHiddenDim; j++)
                    dHidden[i, j] = cache.Hidden[i, j] > 0 ? dActivated[i, j] : 0;

            float[,] dW1 = MatMul(Transpose(cache.X), dHidden);

            UpdateMatrix(W1, dW1, lr, clip);
            UpdateMatrix(W2, dW2, lr, clip);

            for (int j = 0; j < mlpHiddenDim; j++)
            {
                float sum = 0;
                for (int i = 0; i < seqLen; i++) sum += dHidden[i, j];
                b1[j] -= lr * Math.Max(-clip, Math.Min(clip, sum));
            }
            for (int j = 0; j < embedDim; j++)
            {
                float sum = 0;
                for (int i = 0; i < seqLen; i++) sum += dOutput[i, j];
                b2[j] -= lr * Math.Max(-clip, Math.Min(clip, sum));
            }
        }

        private float[,] BackpropAttention(AttentionCache cache, float[,] dOutput, float lr, float clip)
        {
            int seqLen = cache.X.GetLength(0);

            float[,] dConcat = MatMul(dOutput, Transpose(WO));
            float[,] dWO = MatMul(Transpose(cache.ConcatOutput), dOutput);

            float[,,] dAttnOutputs = Reshape3D(dConcat, seqLen, numHeads, headDim);

            float[,,] dQ_heads = new float[seqLen, numHeads, headDim];
            float[,,] dK_heads = new float[seqLen, numHeads, headDim];
            float[,,] dV_heads = new float[seqLen, numHeads, headDim];

            for (int h = 0; h < numHeads; h++)
            {
                float[,] Q_h = GetHeadSlice(cache.Q_heads, h);
                float[,] K_h = GetHeadSlice(cache.K_heads, h);
                float[,] V_h = GetHeadSlice(cache.V_heads, h);
                float[,] dAttnOut_h = GetHeadSlice(dAttnOutputs, h);

                float[,] attn = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        attn[i, j] = cache.AttentionWeights[h, i, j];

                float[,] dAttn = MatMul(dAttnOut_h, Transpose(V_h));
                float[,] dV_h = MatMul(Transpose(attn), dAttnOut_h);

                float[,] dScores = SoftmaxBackward(attn, dAttn);

                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < seqLen; j++)
                        dScores[i, j] /= cache.Scale;

                float[,] dQ_h = MatMul(dScores, K_h);
                float[,] dK_h = MatMul(Transpose(dScores), Q_h);

                SetHeadSlice(dQ_heads, h, dQ_h);
                SetHeadSlice(dK_heads, h, dK_h);
                SetHeadSlice(dV_heads, h, dV_h);
            }

            float[,] dQ = ReshapeBack2D(dQ_heads, seqLen, embedDim);
            float[,] dK = ReshapeBack2D(dK_heads, seqLen, embedDim);
            float[,] dV = ReshapeBack2D(dV_heads, seqLen, embedDim);

            float[,] dX_Q = MatMul(dQ, Transpose(WQ));
            float[,] dX_K = MatMul(dK, Transpose(WK));
            float[,] dX_V = MatMul(dV, Transpose(WV));

            float[,] dX = new float[seqLen, embedDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embedDim; j++)
                    dX[i, j] = dX_Q[i, j] + dX_K[i, j] + dX_V[i, j];

            float[,] dWQ = MatMul(Transpose(cache.X), dQ);
            float[,] dWK = MatMul(Transpose(cache.X), dK);
            float[,] dWV = MatMul(Transpose(cache.X), dV);

            UpdateMatrix(WQ, dWQ, lr, clip);
            UpdateMatrix(WK, dWK, lr, clip);
            UpdateMatrix(WV, dWV, lr, clip);
            UpdateMatrix(WO, dWO, lr, clip);

            return dX;
        }

        private float[,] SoftmaxBackward(float[,] softmax, float[,] dOutput)
        {
            int rows = softmax.GetLength(0);
            int cols = softmax.GetLength(1);
            float[,] dInput = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < cols; k++)
                    {
                        if (j == k)
                            sum += dOutput[i, k] * softmax[i, j] * (1 - softmax[i, j]);
                        else
                            sum += dOutput[i, k] * (-softmax[i, j] * softmax[i, k]);
                    }
                    dInput[i, j] = sum;
                }
            }

            return dInput;
        }

        private void UpdatePositionalEmbeddings(float[,] grads, float lr, float clip)
        {
            int rows = grads.GetLength(0);
            int cols = grads.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    posEmbeddings[i, j] -= lr * Math.Max(-clip, Math.Min(clip, grads[i, j]));
        }

        private void UpdateMatrix(float[,] M, float[,] G, float lr, float clip)
        {
            int r = M.GetLength(0), c = M.GetLength(1);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    M[i, j] -= lr * Math.Max(-clip, Math.Min(clip, G[i, j]));
        }

        private float[,,] Reshape3D(float[,] matrix, int seqLen, int numHeads, int headDim)
        {
            float[,,] result = new float[seqLen, numHeads, headDim];
            for (int i = 0; i < seqLen; i++)
                for (int h = 0; h < numHeads; h++)
                    for (int d = 0; d < headDim; d++)
                        result[i, h, d] = matrix[i, h * headDim + d];
            return result;
        }

        private float[,] ReshapeBack2D(float[,,] tensor, int seqLen, int embedDim)
        {
            int numHeads = tensor.GetLength(1);
            int headDim = tensor.GetLength(2);
            float[,] result = new float[seqLen, embedDim];

            for (int i = 0; i < seqLen; i++)
                for (int h = 0; h < numHeads; h++)
                    for (int d = 0; d < headDim; d++)
                        result[i, h * headDim + d] = tensor[i, h, d];
            return result;
        }

        private float[,] GetHeadSlice(float[,,] tensor, int head)
        {
            int seqLen = tensor.GetLength(0);
            int dim = tensor.GetLength(2);
            float[,] result = new float[seqLen, dim];

            for (int i = 0; i < seqLen; i++)
                for (int d = 0; d < dim; d++)
                    result[i, d] = tensor[i, head, d];
            return result;
        }

        private void SetHeadSlice(float[,,] tensor, int head, float[,] slice)
        {
            int seqLen = slice.GetLength(0);
            int dim = slice.GetLength(1);

            for (int i = 0; i < seqLen; i++)
                for (int d = 0; d < dim; d++)
                    tensor[i, head, d] = slice[i, d];
        }

        private float[,] CopyMatrix(float[,] matrix) => (float[,])matrix.Clone();

        private float[,] ReLU(float[,] X)
        {
            int r = X.GetLength(0), c = X.GetLength(1);
            float[,] res = new float[r, c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    res[i, j] = Math.Max(0, X[i, j]);
            return res;
        }

        private float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));

        private float Tanh(float x) => (float)Math.Tanh(x);

        private float[,] AddResidual(float[,] X, float[,] residual)
        {
            int r = X.GetLength(0), c = X.GetLength(1);
            float[,] res = new float[r, c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    res[i, j] = X[i, j] + residual[i, j];
            return res;
        }

        private float[,] MatMul(float[,] A, float[,] B)
        {
            int rA = A.GetLength(0), cA = A.GetLength(1), cB = B.GetLength(1);
            float[,] res = new float[rA, cB];
            for (int i = 0; i < rA; i++)
                for (int j = 0; j < cB; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < cA; k++) sum += A[i, k] * B[k, j];
                    res[i, j] = sum;
                }
            return res;
        }

        private float[,] Transpose(float[,] matrix)
        {
            int r = matrix.GetLength(0), c = matrix.GetLength(1);
            float[,] res = new float[c, r];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    res[j, i] = matrix[i, j];
            return res;
        }

        private float[,] Softmax(float[,] matrix)
        {
            int r = matrix.GetLength(0), c = matrix.GetLength(1);
            float[,] res = new float[r, c];
            for (int i = 0; i < r; i++)
            {
                float max = float.MinValue;
                for (int j = 0; j < c; j++) max = Math.Max(max, matrix[i, j]);
                float sum = 0;
                for (int j = 0; j < c; j++)
                {
                    res[i, j] = (float)Math.Exp(matrix[i, j] - max);
                    sum += res[i, j];
                }
                if (sum == 0) sum = 1e-9f;
                for (int j = 0; j < c; j++) res[i, j] /= sum;
            }
            return res;
        }

        public class AttentionCache
        {
            public float[,] X, Q, K, V;
            public float[,,] Q_heads, K_heads, V_heads;
            public float[,,] AttentionWeights;
            public float[,] ConcatOutput;
            public float Scale;
        }

        public class MLPCache
        {
            public float[,] X, Hidden, Activated;
        }
    }
}
