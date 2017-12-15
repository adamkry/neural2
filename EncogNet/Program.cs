using Encog.App.Analyst;
using Encog.App.Analyst.CSV.Normalize;
using Encog.App.Analyst.Wizard;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Lma;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Propagation.Manhattan;
using Encog.Neural.Networks.Training.Propagation.Quick;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.Networks.Training.Propagation.SCG;
using Encog.Util.Arrayutil;
using Encog.Util.CSV;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EncogNet
{
    class Program
    {
        static void Main(string[] args)
        {
            double[] p = new double[10];
            IMLData data = new BasicMLData(p);
                        
            var network = CreateNetwork();
            var trainingSet = GetTrainingSet();
            //var train = new ResilientPropagation(network, trainingSet);
            //var train = new Backpropagation(network, trainingSet, 0.7, 0.2);
            //var train = new ManhattanPropagation(network, trainingSet, 0.001);
            //var train = new ScaledConjugateGradient(network, trainingSet);
            //var train = new LevenbergMarquardtTraining(network, trainingSet);
            var train = new QuickPropagation(network, trainingSet, 2.0);


            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine($"Iteration No: {epoch++}, Error: {train.Error}");
            }
            while (train.Error > 0.01);

            foreach (var item in trainingSet)
            {
                var output = network.Compute(item.Input);
                Console.WriteLine($"Input: {item.Input[0]}, {item.Input[1]} Ideal: {item.Ideal[0]} Actual: {output[0]}");
            }
            Console.ReadLine();
        }

        static BasicMLDataSet GetTrainingSet()
        {
            //UCI Machine Learning Repository

            //INPUT
            double[][] XOR_Input =
            {
                new [] { 0.0, 0.0 },
                new [] { 1.0, 0.0 },
                new [] { 0.0, 1.0 },
                new [] { 1.0, 1.0 }
            };

            //OUTPUT
            double[][] XOR_Ideal =
            {
                new [] { 0.0 },
                new [] { 1.0 },
                new [] { 1.0 },
                new [] { 0.0 }
            };

            //TRAINING SET
            return new BasicMLDataSet(XOR_Input, XOR_Ideal);
        }

        static BasicNetwork CreateNetwork()
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();
            return network;
        }

        static void Normalization()
        {
            //Single value
            var weightNorm = new NormalizedField(NormalizationAction.Normalize, "Weights", ahigh: 40.0, alow: 50.0, nhigh: -1.0, nlow: 1.0);
            double normalizedValue = weightNorm.Normalize(42.5);
            double denormalizedValue = weightNorm.DeNormalize(normalizedValue);

            //Array
            double[] weights = new double[] { 40.0, 42.5, 43.0, 49.0, 50.0 };
            var weightNorm2 = new NormalizeArray();
            weightNorm2.NormalizedHigh = 1.0;
            weightNorm2.NormalizedLow = -1.0;
            double[] normalizedWeights = weightNorm2.Process(weights);
        }

        static void EncogAnalyst()
        {
            var sourceFile = new FileInfo("RawFile.csv");
            var targetFile = new FileInfo("NormalizedFile.csv");
            var analyst = new EncogAnalyst();
            var wizard = new AnalystWizard(analyst);
            wizard.Wizard(sourceFile, true, AnalystFileFormat.DecpntComma);
            var norm = new AnalystNormalizeCSV();
            norm.Analyze(sourceFile, true, CSVFormat.DecimalComma, analyst);
            norm.Normalize(targetFile);
        }
    }
}
