using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;

namespace InferNetAR.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            // Parse command-line arguments
            if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
            {
                PrintUsage();
                return;
            }

            try
            {
                var config = ParseArguments(args);
                
                // Read CSV data
                var data = ReadCsvFile(config.InputFile);
                
                System.Console.WriteLine($"Loaded {data.Length} data points from {config.InputFile}");
                System.Console.WriteLine($"Sample values: [{string.Join(", ", data.Take(Math.Min(10, data.Length)).Select(x => x.ToString("F3")))}...]");
                System.Console.WriteLine();

                // Fit model based on type
                switch (config.ModelType.ToLower())
                {
                    case "ar":
                        FitARModel(data, config);
                        break;
                    case "arx":
                        FitARXModel(data, config);
                        break;
                    default:
                        System.Console.WriteLine($"Error: Unknown model type '{config.ModelType}'");
                        PrintUsage();
                        return;
                }
            }
            catch (Exception ex)
            {
                System.Console.WriteLine($"Error: {ex.Message}");
                if (ex.InnerException != null)
                {
                    System.Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                System.Console.WriteLine();
                PrintUsage();
            }
        }

        static void PrintUsage()
        {
            System.Console.WriteLine("Infer.NET Time Series Models");
            System.Console.WriteLine("=============================");
            System.Console.WriteLine();
            System.Console.WriteLine("Usage:");
            System.Console.WriteLine("  dotnet run --project InferNetAR.Console -- <options>");
            System.Console.WriteLine();
            System.Console.WriteLine("Required Arguments:");
            System.Console.WriteLine("  --input <file>         Path to CSV file containing time series data");
            System.Console.WriteLine("  --model <type>        Model type: 'ar' or 'arx'");
            System.Console.WriteLine();
            System.Console.WriteLine("AR Model Parameters:");
            System.Console.WriteLine("  --ar-order <n>        AR model order (p) [mandatory for AR]");
            System.Console.WriteLine();
            System.Console.WriteLine("ARX Model Parameters:");
            System.Console.WriteLine("  --ar-order <n>        AR model order (p) [mandatory for ARX]");
            System.Console.WriteLine("  --exog-order <n>      Exogenous lag order (q) [mandatory for ARX]");
            System.Console.WriteLine("  --exog-count <n>      Number of exogenous variables (k) [mandatory for ARX]");
            System.Console.WriteLine();
            System.Console.WriteLine("Optional Parameters:");
            System.Console.WriteLine("  --noise-var <val>     Prior for noise variance (default: 1.0)");
            System.Console.WriteLine("  --coeff-var <val>     Prior for coefficient variance (default: 10.0)");
            System.Console.WriteLine();
            System.Console.WriteLine("CSV File Format:");
            System.Console.WriteLine("  For AR model: One column with time series values");
            System.Console.WriteLine("  For ARX model: First column = endogenous series, subsequent columns = exogenous variables");
            System.Console.WriteLine("  First row can be a header row (will be skipped)");
            System.Console.WriteLine();
            System.Console.WriteLine("Examples:");
            System.Console.WriteLine("  # Fit AR(2) model:");
            System.Console.WriteLine("  dotnet run -- --input data.csv --model ar --ar-order 2");
            System.Console.WriteLine();
            System.Console.WriteLine("  # Fit ARX(2,1) model with 2 exogenous variables:");
            System.Console.WriteLine("  dotnet run -- --input data.csv --model arx --ar-order 2 --exog-order 1 --exog-count 2");
            System.Console.WriteLine();
        }

        static Configuration ParseArguments(string[] args)
        {
            var config = new Configuration();

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i].ToLower())
                {
                    case "--input":
                        if (i + 1 >= args.Length) throw new ArgumentException("--input requires a file path");
                        config.InputFile = args[++i];
                        break;
                    case "--model":
                        if (i + 1 >= args.Length) throw new ArgumentException("--model requires a model type");
                        config.ModelType = args[++i];
                        break;
                    case "--ar-order":
                        if (i + 1 >= args.Length) throw new ArgumentException("--ar-order requires a number");
                        config.ArOrder = int.Parse(args[++i]);
                        break;
                    case "--exog-order":
                        if (i + 1 >= args.Length) throw new ArgumentException("--exog-order requires a number");
                        config.ExogOrder = int.Parse(args[++i]);
                        break;
                    case "--exog-count":
                        if (i + 1 >= args.Length) throw new ArgumentException("--exog-count requires a number");
                        config.ExogCount = int.Parse(args[++i]);
                        break;
                    case "--noise-var":
                        if (i + 1 >= args.Length) throw new ArgumentException("--noise-var requires a number");
                        config.NoiseVariancePrior = double.Parse(args[++i]);
                        break;
                    case "--coeff-var":
                        if (i + 1 >= args.Length) throw new ArgumentException("--coeff-var requires a number");
                        config.CoefficientVariancePrior = double.Parse(args[++i]);
                        break;
                }
            }

            // Validate required arguments
            if (string.IsNullOrEmpty(config.InputFile))
                throw new ArgumentException("--input argument is required");
            
            if (string.IsNullOrEmpty(config.ModelType))
                throw new ArgumentException("--model argument is required");

            if (config.ModelType.ToLower() == "ar")
            {
                if (config.ArOrder == null)
                    throw new ArgumentException("--ar-order is required for AR model");
            }
            else if (config.ModelType.ToLower() == "arx")
            {
                if (config.ArOrder == null)
                    throw new ArgumentException("--ar-order is required for ARX model");
                if (config.ExogOrder == null)
                    throw new ArgumentException("--exog-order is required for ARX model");
                if (config.ExogCount == null)
                    throw new ArgumentException("--exog-count is required for ARX model");
            }

            return config;
        }

        static double[] ReadCsvFile(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"File not found: {filePath}");

            var values = new List<double>();
            var lines = File.ReadAllLines(filePath);
            
            int startIndex = 0;
            
            // Check if first line is a header (contains non-numeric data)
            if (lines.Length > 0)
            {
                var firstLine = lines[0].Split(',');
                if (!double.TryParse(firstLine[0].Trim(), out _))
                {
                    startIndex = 1; // Skip header row
                }
            }

            for (int i = startIndex; i < lines.Length; i++)
            {
                var parts = lines[i].Split(',');
                if (parts.Length == 0 || string.IsNullOrWhiteSpace(parts[0]))
                    continue;
                
                if (double.TryParse(parts[0].Trim(), out double value))
                {
                    values.Add(value);
                }
            }

            if (values.Count == 0)
                throw new InvalidDataException("No valid numeric data found in CSV file");

            return values.ToArray();
        }

        static double[][] ReadCsvFileMultiColumn(string filePath, int expectedColumns)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"File not found: {filePath}");

            var columns = new List<double>[expectedColumns];
            for (int i = 0; i < expectedColumns; i++)
            {
                columns[i] = new List<double>();
            }

            var lines = File.ReadAllLines(filePath);
            
            int startIndex = 0;
            
            // Check if first line is a header
            if (lines.Length > 0)
            {
                var firstLine = lines[0].Split(',');
                bool isHeader = true;
                foreach (var part in firstLine)
                {
                    if (double.TryParse(part.Trim(), out _))
                    {
                        isHeader = false;
                        break;
                    }
                }
                if (isHeader)
                    startIndex = 1;
            }

            for (int i = startIndex; i < lines.Length; i++)
            {
                var parts = lines[i].Split(',');
                if (parts.Length < expectedColumns)
                    throw new InvalidDataException($"Line {i + 1} has insufficient columns. Expected {expectedColumns}, got {parts.Length}");

                for (int col = 0; col < expectedColumns; col++)
                {
                    if (double.TryParse(parts[col].Trim(), out double value))
                    {
                        columns[col].Add(value);
                    }
                    else
                    {
                        throw new InvalidDataException($"Invalid numeric value in column {col + 1} at line {i + 1}");
                    }
                }
            }

            if (columns[0].Count == 0)
                throw new InvalidDataException("No valid numeric data found in CSV file");

            return columns.Select(col => col.ToArray()).ToArray();
        }

        static void FitARModel(double[] data, Configuration config)
        {
            if (!config.ArOrder.HasValue)
                throw new ArgumentException("AR model requires ar-order parameter");
            
            System.Console.WriteLine($"Fitting AR({config.ArOrder}) model using Infer.NET...");
            
            var model = new ARModel(order: config.ArOrder.Value);
            
            var posterior = model.Fit(data);
            System.Console.WriteLine("Model fitting completed!");
            System.Console.WriteLine();

            posterior.PrintSummary();
            System.Console.WriteLine();

            var coefficients = posterior.GetCoefficients();
            var noiseVariance = posterior.GetNoiseVariance();
            
            System.Console.WriteLine("Model Statistics:");
            System.Console.WriteLine($"- Intercept: {coefficients[0]:F4}");
            for (int i = 1; i < coefficients.Length; i++)
            {
                System.Console.WriteLine($"- AR({i}) coefficient: {coefficients[i]:F4}");
            }
            System.Console.WriteLine($"- Noise variance: {noiseVariance:F4}");
        }

        static void FitARXModel(double[] data, Configuration config)
        {
            if (!config.ArOrder.HasValue || !config.ExogOrder.HasValue || !config.ExogCount.HasValue)
                throw new ArgumentException("ARX model requires ar-order, exog-order, and exog-count parameters");
            
            System.Console.WriteLine($"Fitting ARX({config.ArOrder},{config.ExogOrder}) model using Infer.NET...");
            
            // Read multi-column CSV file
            int totalColumns = config.ExogCount.Value + 1; // endogenous + exogenous
            var allData = ReadCsvFileMultiColumn(config.InputFile, totalColumns);
            
            var endogenousData = allData[0];
            var exogenousData = new double[config.ExogCount.Value][];
            for (int i = 0; i < config.ExogCount.Value; i++)
            {
                exogenousData[i] = allData[i + 1];
            }

            System.Console.WriteLine($"Loaded {endogenousData.Length} endogenous observations");
            System.Console.WriteLine($"Loaded {config.ExogCount.Value} exogenous variables");
            System.Console.WriteLine();

            var model = new ARXModel(
                arOrder: config.ArOrder.Value,
                exogOrder: config.ExogOrder.Value,
                exogenousCount: config.ExogCount.Value,
                noiseVariancePrior: config.NoiseVariancePrior,
                coefficientVariancePrior: config.CoefficientVariancePrior
            );
            
            var posterior = model.Fit(endogenousData, exogenousData);
            System.Console.WriteLine("Model fitting completed!");
            System.Console.WriteLine();

            posterior.PrintSummary();
            System.Console.WriteLine();

            var arCoeffs = posterior.GetARCoefficients();
            var exogCoeffs = posterior.GetExogCoefficients();
            var intercept = posterior.GetIntercept();
            var noiseVariance = posterior.GetNoiseVariance();
            
            System.Console.WriteLine("Model Statistics:");
            System.Console.WriteLine($"- Intercept: {intercept:F4}");
            for (int i = 0; i < arCoeffs.Length; i++)
            {
                System.Console.WriteLine($"- AR({i + 1}) coefficient: {arCoeffs[i]:F4}");
            }
            for (int varIdx = 0; varIdx < config.ExogCount.Value; varIdx++)
            {
                var coeffs = posterior.GetExogCoefficientsForVariable(varIdx);
                for (int lag = 0; lag < config.ExogOrder.Value; lag++)
                {
                    System.Console.WriteLine($"- Exog{varIdx + 1}(lag{lag + 1}) coefficient: {coeffs[lag]:F4}");
                }
            }
            System.Console.WriteLine($"- Noise variance: {noiseVariance:F4}");
        }

        class Configuration
        {
            public string InputFile { get; set; } = string.Empty;
            public string ModelType { get; set; } = string.Empty;
            public int? ArOrder { get; set; }
            public int? ExogOrder { get; set; }
            public int? ExogCount { get; set; }
            public double NoiseVariancePrior { get; set; } = 1.0;
            public double CoefficientVariancePrior { get; set; } = 10.0;
        }
    }
}
