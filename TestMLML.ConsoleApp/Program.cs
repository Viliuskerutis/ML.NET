//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using TestMLML.Model.DataModels;


namespace TestMLML.ConsoleApp
{
    class Program
    {
        //Machine Learning model to load and use for predictions
        private const string MODEL_FILEPATH = @"MLModel.zip";

        //Dataset to use for predictions 
        private const string DATA_FILEPATH = @"new_bus_data.csv";

        private const string TRAIN_DATA_FILEPATH = @"new_train_bus_data.csv";

        private const string TEST_DATA_FILEPATH = @"new_test_bus_data.csv";

        private static string[] METHODS = new string[] { "LIGHT GBM", "FAST TREE", "FAST FOREST", "FAST TREE TWEEDIE", "NMEAN", "Binary" };

        static void Main(string[] args)
        {
            List<double> score = new List<double>();
            Console.WriteLine();
            for (int i = 0; i < 4; i++)
            {
                Console.WriteLine(METHODS[i]);
                Console.WriteLine();
                CreateModel(i);
                MLContext mlContext = new MLContext();

                // Training code used by ML.NET CLI and AutoML to generate the model
                //ModelBuilder.CreateModel();

                ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(MODEL_FILEPATH), out DataViewSchema inputSchema);
                var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

                List<ModelInput> list = new List<ModelInput>();
                int counter = 0;
                string line;

                StreamReader file =
                    new StreamReader(TEST_DATA_FILEPATH);
                file.ReadLine();
                while ((line = file.ReadLine()) != null)
                {
                    string[] words = line.Split(',');
                    ModelInput model = new ModelInput
                    {
                        Week_day = float.Parse(words[0]),
                        Hour = float.Parse(words[1]),
                        Rout_count = float.Parse(words[2]),
                        Temperature = float.Parse(words[3]),
                        Visibility = float.Parse(words[4]),
                      //  Humidity = float.Parse(words[5]),
                        Distance = float.Parse(words[5])
                    };
                    list.Add(model);

                    counter++;
                }
                file.Close();
                Console.WriteLine();
                System.Console.WriteLine("Prediction examples, for testing there were {0} lines in file.", counter);

                // Create sample data to do a single prediction with it 
                ModelInput[] sampleData = CreateSingleDataSample(mlContext, TEST_DATA_FILEPATH, list);

                // Try a single prediction
                ModelOutput[] predictionResult = new ModelOutput[list.Count];

                for (int j = 0; j < sampleData.Length; j++)
                {
                    ModelInput data = sampleData[j];
                    predictionResult[j] = predEngine.Predict(data);
                }

                for (int j = 0; j < 10; j++)
                {
                    if (predictionResult[j].Score >= 700)
                    {
                        Console.WriteLine($"Single Prediction --> Actual value: {sampleData[j].Distance,-10} " +
                            $"| Predicted value: {predictionResult[j].Score:00.00}  Bus will be late");
                    } else
                    {
                        Console.WriteLine($"Single Prediction --> Actual value: {sampleData[j].Distance,-10} " +
                            $"| Predicted value:  {predictionResult[j].Score:0.00}  Bus will be on time");
                    }
                }
                Console.WriteLine();
                Console.WriteLine("=============== End of testing process, starting cross validation ===============");

                // Load Data
                IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                                path: DATA_FILEPATH,
                                                hasHeader: true,
                                                separatorChar: ',',
                                                allowQuoting: true,
                                                allowSparse: false);

                // Build training pipeline
                IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext, i);
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 10, labelColumnName: "distance");
                score.Add(crossValidationResults.Select(r => r.Metrics.RSquared).Average());
                PrintRegressionFoldsAverageMetrics(crossValidationResults);
                Console.ReadKey();
            }
            Console.WriteLine();
            Console.WriteLine("=============== Selecting the result by voting method ===============");
            Console.WriteLine();
            Console.WriteLine(new string('-', 37));
            Console.WriteLine("| Method name              | Score  |");
            double max = 0.0;
            int saveIndex = 0;
            for (int i = 0; i < score.Count; i++)
            {
                Console.WriteLine(new string('-', 37));
                Console.WriteLine("| {0,-24} | {1:0.00}% |", METHODS[i], score[i]*100);
                if (max < score[i])
                {
                    max = score[i];
                    saveIndex = i;
                }
            }
            Console.WriteLine(new string('-', 37));
            Console.WriteLine();
            Console.WriteLine("Best method: {0}", METHODS[saveIndex]);
            Console.WriteLine("Score: {0:0.00}%", max*100);
        }

        // Method to load single row of data to try a single prediction
        // You can change this code and create your own sample data here (Hardcoded or from any source)
        private static ModelInput[] CreateSingleDataSample(MLContext mlContext, string dataFilePath, List<ModelInput> list)
        {
            // Input Data
            ModelInput[] input = new ModelInput[list.Count];
            for (int i = 0; i < list.Count; i++)
            {
                input[i] = new ModelInput
                {
                    Week_day = list[i].Week_day,
                    Hour = list[i].Hour,
                    Rout_count = list[i].Rout_count,
                    Temperature = list[i].Temperature,
                    Visibility = list[i].Visibility,
                  //  Humidity = list[i].Humidity,
                    Distance = list[i].Distance
                };
            }

            return input;
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }


        // Create MLContext to be shared across the model creation workflow objects 
        // Set a random seed for repeatable/deterministic results across multiple trainings.
        private static MLContext mlContext = new MLContext(seed: 1);

        public static void CreateModel(int method)
        {
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TRAIN_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Build training pipeline
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext, method);

            // Evaluate quality of Model
            //Evaluate(mlContext, trainingDataView, trainingPipeline);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // Save model
            SaveModel(mlContext, mlModel, MODEL_FILEPATH, trainingDataView.Schema);
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext, int method)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.NormalizeMinMax(
                                           outputColumnName: "week_day",
                                           maximumExampleCount: 4000)
                                       .Append(mlContext.Transforms.NormalizeMinMax(
                                           outputColumnName: "hour", 
                                           maximumExampleCount: 4000))
                                       .Append(mlContext.Transforms.NormalizeMinMax(
                                           outputColumnName: "rout_count", 
                                           maximumExampleCount: 4000))
                                       .Append(mlContext.Transforms.NormalizeMinMax(
                                           outputColumnName: "temperature", 
                                           maximumExampleCount: 4000))
                                       .Append(mlContext.Transforms.NormalizeMinMax(
                                           outputColumnName: "visibility",
                                           maximumExampleCount: 4000))

                                     //  .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "humidity", maximumExampleCount: 4000))
                                       .Append(mlContext.Transforms.Concatenate("Features", new[] { "week_day", "hour", "rout_count", "temperature", "visibility" }));
            // Set the training algorithm 

            if (method == 0)
                return 
                dataProcessPipeline.Append(mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options()
                { NumberOfIterations = 100,
                  LearningRate = 0.3f,
                  NumberOfLeaves = 15,
                  MinimumExampleCountPerLeaf = 10,
                  UseCategoricalSplit = true,
                  HandleMissingValue = true,
                  MinimumExampleCountPerGroup = 100,
                  MaximumCategoricalSplitPointCount = 16,
                  CategoricalSmoothing = 1,
                  L2CategoricalRegularization = 0.1,
                  Booster = new GradientBooster.Options()
                        {
                            L2Regularization = 0.5,
                            L1Regularization = 1 },
                            LabelColumnName = "distance",
                            FeatureColumnName = "Features"
                        }));
            if (method == 1)
                return 
                dataProcessPipeline.Append(mlContext.Regression.Trainers.FastTree(
                    labelColumnName: "distance", 
                    featureColumnName: "Features", 
                    numberOfLeaves: 15, 
                    numberOfTrees: 100, 
                    minimumExampleCountPerLeaf: 10, 
                    learningRate: 0.1));
            //if (method == 2)
             //   return dataProcessPipeline.Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "distance", featureColumnName: "Features", optimizationTolerance: 1E-07f, l1Regularization: 1, l2Regularization: 1, historySize: 20));
            if (method == 2)
                return dataProcessPipeline.Append(mlContext.Regression.Trainers.FastForest(labelColumnName: "distance", featureColumnName: "Features", numberOfLeaves: 15, numberOfTrees: 100, minimumExampleCountPerLeaf: 10));
            if (method == 3)
                return 
                dataProcessPipeline.Append(mlContext.Regression.Trainers.FastTreeTweedie(
                    labelColumnName: "distance", 
                    featureColumnName: "Features", 
                    numberOfLeaves: 15, 
                    numberOfTrees: 100, 
                    minimumExampleCountPerLeaf: 10, 
                    learningRate: 0.2));
            if (method == 4)
            return mlContext.Transforms.Conversion.MapValueToKey("distance", "distance")
                                          .Append(mlContext.Transforms.Concatenate("Features", new[] { "week_day", "hour", "rout_count", "temperature", "visibility" }))
                                          .Append(mlContext.MulticlassClassification.Trainers.LightGbm(labelColumnName: "distance", featureColumnName: "Features")
                                          .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel")));
            if(method == 6)
                return mlContext.Transforms.Conversion.MapValueToKey("distance", "distance")
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "week_day", "hour", "rout_count", "temperature", "visibility" })
                                      .Append(mlContext.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options() { NumberOfIterations = 100, LearningRate = 0.07110989f, NumberOfLeaves = 32, MinimumExampleCountPerLeaf = 1, UseCategoricalSplit = false, HandleMissingValue = true, MinimumExampleCountPerGroup = 100, MaximumCategoricalSplitPointCount = 16, CategoricalSmoothing = 20, L2CategoricalRegularization = 0.1, UseSoftmax = true, Booster = new GradientBooster.Options() { L2Regularization = 0.5, L1Regularization = 1 }, LabelColumnName = "distance", FeatureColumnName = "Features" })
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"))));

            //labai ilgai skaiciuoja, bet rezultatai gan tikslus
            //var trainer = mlContext.Regression.Trainers.Gam(labelColumnName: "distance", featureColumnName: "Features", learningRate: 0.002, numberOfIterations: 9500, maximumBinCountPerFeature: 255);

            //Kazkaip ne taip veikia, nesigauna rezultatai
            //var trainer = mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "distance", featureColumnName: "Features", optimizationTolerance: 1E-07f, historySize: 20, l2Regularization: 1, l1Regularization: 1);

            //siek tiek netikslu, gaunasi neigiamas ivertis
            //var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "distance", featureColumnName: "Features");

            return null;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============");
            return model;
        }

        private static void Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 10, labelColumnName: "fare_amount");
            PrintRegressionFoldsAverageMetrics(crossValidationResults);
        }
        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            //Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            //Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

        public static void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        public static void PrintRegressionFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine();
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            Console.WriteLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            Console.WriteLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine();
        }
    }
}
