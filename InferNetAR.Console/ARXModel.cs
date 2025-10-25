using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Math;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace InferNetAR.Console
{
    /// <summary>
    /// ARX(p,q) Autoregressive with eXogenous inputs Model implementation using Infer.NET
    /// </summary>
    public class ARXModel
    {
        private readonly int _arOrder;
        private readonly int _exogOrder;
        private readonly int _exogenousCount;
        private readonly double _noiseVariancePrior;
        private readonly double _coefficientVariancePrior;

        /// <summary>
        /// Initializes a new instance of the ARX(p,q) model
        /// </summary>
        /// <param name="arOrder">Order of autoregressive lags (p)</param>
        /// <param name="exogOrder">Order of exogenous variable lags (q)</param>
        /// <param name="exogenousCount">Number of exogenous variables</param>
        /// <param name="noiseVariancePrior">Prior for noise variance</param>
        /// <param name="coefficientVariancePrior">Prior for coefficient variance</param>
        public ARXModel(int arOrder, int exogOrder, int exogenousCount, double noiseVariancePrior = 1.0, double coefficientVariancePrior = 10.0)
        {
            if (arOrder < 1)
                throw new ArgumentOutOfRangeException(nameof(arOrder), "AR model order must be at least 1.");
            if (exogOrder < 0)
                throw new ArgumentOutOfRangeException(nameof(exogOrder), "Exogenous lag order must be non-negative.");
            if (exogenousCount < 0)
                throw new ArgumentOutOfRangeException(nameof(exogenousCount), "Exogenous count must be non-negative.");
            
            _arOrder = arOrder;
            _exogOrder = exogOrder;
            _exogenousCount = exogenousCount;
            _noiseVariancePrior = noiseVariancePrior;
            _coefficientVariancePrior = coefficientVariancePrior;
        }

        /// <summary>
        /// Gets the AR order (p)
        /// </summary>
        public int ArOrder => _arOrder;

        /// <summary>
        /// Gets the exogenous lag order (q)
        /// </summary>
        public int ExogOrder => _exogOrder;

        /// <summary>
        /// Gets the number of exogenous variables
        /// </summary>
        public int ExogenousCount => _exogenousCount;

        /// <summary>
        /// Fits an ARX(p,q) model to the observed time series data with exogenous inputs using Infer.NET.
        /// </summary>
        /// <param name="observedSeries">The time series data.</param>
        /// <param name="exogenousData">The exogenous variables data (each row is a time series).</param>
        /// <returns>An ARXPosterior object containing the inferred distributions of the parameters.</returns>
        public ARXPosterior Fit(double[] observedSeries, double[][] exogenousData)
        {
            if (exogenousData == null || exogenousData.Length != _exogenousCount)
                throw new ArgumentException($"Expected {_exogenousCount} exogenous time series, but got {exogenousData?.Length ?? 0}.");
            
            if (exogenousData.Any(series => series.Length != observedSeries.Length))
                throw new ArgumentException("All time series must have the same length.");
            
            return BuildARXModel(observedSeries, exogenousData, _arOrder, _exogOrder);
        }

        /// <summary>
        /// Generates sample ARX(p) data for testing
        /// </summary>
        /// <param name="arCoefficients">AR coefficients (including intercept as first element)</param>
        /// <param name="exogCoefficients">Exogenous coefficients</param>
        /// <param name="exogenousData">Exogenous time series data</param>
        /// <param name="noiseVariance">Noise variance</param>
        /// <param name="length">Length of time series</param>
        /// <param name="seed">Random seed</param>
        /// <returns>Generated time series data</returns>
        public static double[] GenerateSampleARXData(double[] arCoefficients, double[] exogCoefficients, double[][] exogenousData, double noiseVariance, int length, int seed = 42)
        {
            var random = new Random(seed);
            var data = new double[length];
            int order = arCoefficients.Length - 1; // order is number of AR coefficients (excluding intercept)
            int exogCount = exogCoefficients.Length;
            
            // Initialize with zeros for the first p values
            for (int i = 0; i < order; i++)
            {
                data[i] = 0.0;
            }

            // Generate ARX(p) process
            for (int t = order; t < length; t++)
            {
                double value = arCoefficients[0]; // intercept
                
                // Add AR terms
                for (int i = 1; i < arCoefficients.Length; i++)
                {
                    value += arCoefficients[i] * data[t - i];
                }
                
                // Add exogenous terms
                for (int i = 0; i < exogCount; i++)
                {
                    value += exogCoefficients[i] * exogenousData[i][t];
                }
                
                // Add noise
                value += Math.Sqrt(noiseVariance) * random.NextGaussian();
                data[t] = value;
            }

            return data;
        }

        /// <summary>
        /// Generate sample ARX(p) data with default parameters
        /// </summary>
        public static double[] GenerateSampleARXData(int order, int exogCount, int length, int seed = 42)
        {
            // Default AR coefficients
            var arCoefficients = new double[order + 1];
            arCoefficients[0] = 0.0; // intercept
            if (order >= 1) arCoefficients[1] = 0.6;
            if (order >= 2) arCoefficients[2] = -0.3;
            if (order >= 3) arCoefficients[3] = 0.1;
            
            // Default exogenous coefficients
            var exogCoefficients = new double[exogCount];
            for (int i = 0; i < exogCount; i++)
            {
                exogCoefficients[i] = 0.2 + 0.1 * i; // Simple increasing pattern
            }
            
            // Generate random exogenous data
            var random = new Random(seed);
            var exogenousData = new double[exogCount][];
            for (int i = 0; i < exogCount; i++)
            {
                exogenousData[i] = new double[length];
                for (int t = 0; t < length; t++)
                {
                    exogenousData[i][t] = random.NextGaussian();
                }
            }
            
            return GenerateSampleARXData(arCoefficients, exogCoefficients, exogenousData, 1.0, length, seed);
        }

        /// <summary>
        /// Build ARX(p) model using Infer.NET
        /// </summary>
        private ARXPosterior BuildARXModel(double[] observedSeries, double[][] exogenousData, int p, int q)
        {
            int T = observedSeries.Length;
            int maxLag = Math.Max(p, q);
            int n = T - maxLag; // Number of data points for regression (after initial lags)
            int k = _exogenousCount; // Number of exogenous variables
            int totalCoeffs = 1 + p + (k * q); // intercept + AR coefficients + (exogenous variables × q lags)
            
            // Model using VectorGaussian for efficiency (including intercept and exogenous terms)
            var thetaMean = Vector.Zero(totalCoeffs);
            var thetaVariance = PositiveDefiniteMatrix.Identity(totalCoeffs);
            var theta = Variable.VectorGaussianFromMeanAndVariance(thetaMean, thetaVariance).Named("theta");
            var prec = Variable.GammaFromShapeAndRate(1.0, 1.0).Named("precision");
            
            // Create design matrix and response (including intercept, AR terms, and exogenous lags)
            var XData = new Vector[n];
            var yData = new double[n];
            
            for (int i = 0; i < n; i++)
            {
                var row = new double[totalCoeffs];
                int colIdx = 0;
                
                // Intercept term
                row[colIdx++] = 1.0;
                
                // AR terms: [x_{t-1}, x_{t-2}, ..., x_{t-p}]
                for (int j = 0; j < p; j++)
                    row[colIdx++] = observedSeries[maxLag - 1 - j + i];
                
                // Exogenous terms with q lags for each exogenous variable
                // For each exogenous variable k, include lags: [u_{k,t-1}, u_{k,t-2}, ..., u_{k,t-q}]
                for (int exogVar = 0; exogVar < k; exogVar++)
                {
                    for (int lag = 0; lag < q; lag++)
                    {
                        row[colIdx++] = exogenousData[exogVar][maxLag - lag - 1 + i];
                    }
                }
                
                XData[i] = Vector.FromArray(row);
                yData[i] = observedSeries[maxLag + i];
            }
            
            Range dataRange = new Range(n).Named("dataRange");
            var X = Variable.Observed(XData, dataRange).Named("X");
            var y = Variable.Observed(yData, dataRange).Named("y");
            
            // Model: y[i] = X[i] · theta + noise
            using (Variable.ForEach(dataRange))
            {
                var mean = Variable.InnerProduct(X[dataRange], theta).Named("mean");
                y[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, prec);
            }
            
            // Inference
            var engine = new InferenceEngine();
            engine.NumberOfIterations = 50; // Number of iterations for Expectation Propagation
            
            var thetaPost = engine.Infer<VectorGaussian>(theta);
            var precPost = engine.Infer<Gamma>(prec);
            
            // Convert to individual Gaussian distributions
            var thetaMeanPost = thetaPost.GetMean();
            var thetaVarPost = thetaPost.GetVariance();
            
            var allCoefficientDistributions = new Gaussian[totalCoeffs];
            
            // All coefficients (intercept, AR, exogenous) are estimated
            for (int i = 0; i < totalCoeffs; i++)
                allCoefficientDistributions[i] = new Gaussian(thetaMeanPost[i], thetaVarPost[i, i]);
            
            // Separate AR and exogenous coefficients
            var intercept = allCoefficientDistributions[0];
            var arCoefficients = new Gaussian[p];
            var exogCoefficients = new Gaussian[k * q]; // k variables × q lags
            
            for (int i = 0; i < p; i++)
                arCoefficients[i] = allCoefficientDistributions[i + 1];
            
            for (int i = 0; i < k * q; i++)
                exogCoefficients[i] = allCoefficientDistributions[p + 1 + i];
            
            var sigma2Post = new Gamma(precPost.Shape, 1.0 / precPost.Rate);
            
            return new ARXPosterior(arCoefficients, exogCoefficients, sigma2Post, intercept, p, q, k);
        }
    }

    /// <summary>
    /// Posterior distributions from ARX(p,q) model fitting
    /// </summary>
    public class ARXPosterior
    {
        private readonly Gaussian[] _arCoefficients;
        private readonly Gaussian[] _exogCoefficients;
        private readonly Gamma _noiseVariance;
        private readonly Gaussian _intercept;
        private readonly int _arOrder;
        private readonly int _exogOrder;
        private readonly int _exogCount;

        /// <summary>
        /// Initializes a new instance of ARXPosterior
        /// </summary>
        /// <param name="arCoefficients">AR coefficient distributions</param>
        /// <param name="exogCoefficients">Exogenous coefficient distributions (flattened: k variables × q lags)</param>
        /// <param name="noiseVariance">Noise variance distribution</param>
        /// <param name="intercept">Intercept distribution</param>
        /// <param name="arOrder">AR order (p)</param>
        /// <param name="exogOrder">Exogenous lag order (q)</param>
        /// <param name="exogCount">Number of exogenous variables (k)</param>
        public ARXPosterior(Gaussian[] arCoefficients, Gaussian[] exogCoefficients, Gamma noiseVariance, Gaussian intercept, int arOrder, int exogOrder, int exogCount)
        {
            _arCoefficients = arCoefficients;
            _exogCoefficients = exogCoefficients;
            _noiseVariance = noiseVariance;
            _intercept = intercept;
            _arOrder = arOrder;
            _exogOrder = exogOrder;
            _exogCount = exogCount;
        }

        /// <summary>
        /// Gets the AR order (p)
        /// </summary>
        public int ArOrder => _arOrder;

        /// <summary>
        /// Gets the exogenous lag order (q)
        /// </summary>
        public int ExogOrder => _exogOrder;

        /// <summary>
        /// Gets the number of exogenous variables (k)
        /// </summary>
        public int ExogCount => _exogCount;

        public Gaussian[] GetARCoefficientsDistributions() => _arCoefficients;
        public Gaussian[] GetExogCoefficientsDistributions() => _exogCoefficients;
        public Gamma GetNoiseVarianceDistribution() => _noiseVariance;
        public Gaussian GetInterceptDistribution() => _intercept;

        public double[] GetARCoefficients()
        {
            return _arCoefficients.Select(g => g.GetMean()).ToArray();
        }

        public double[] GetExogCoefficients()
        {
            return _exogCoefficients.Select(g => g.GetMean()).ToArray();
        }

        /// <summary>
        /// Gets exogenous coefficients for a specific variable by index
        /// </summary>
        /// <param name="variableIndex">Index of the exogenous variable (0-indexed)</param>
        /// <returns>Array of q coefficients for the specified variable</returns>
        public double[] GetExogCoefficientsForVariable(int variableIndex)
        {
            if (variableIndex < 0 || variableIndex >= _exogCount)
                throw new ArgumentOutOfRangeException(nameof(variableIndex), $"Variable index must be between 0 and {_exogCount - 1}.");
            
            var result = new double[_exogOrder];
            int startIdx = variableIndex * _exogOrder;
            
            for (int i = 0; i < _exogOrder; i++)
            {
                result[i] = _exogCoefficients[startIdx + i].GetMean();
            }
            
            return result;
        }

        public double GetIntercept()
        {
            return _intercept.GetMean();
        }

        public double GetNoiseVariance()
        {
            // Mean of InverseGamma(a,b) is b/(a-1)
            // Here, Gamma(shape, rate) for precision, so variance is 1/precision
            // Mean of 1/Gamma(shape, rate) is rate / (shape - 1) for shape > 1
            if (_noiseVariance.Shape > 1)
            {
                return _noiseVariance.Rate / (_noiseVariance.Shape - 1);
            }
            return double.NaN; // Or handle appropriately if shape <= 1
        }

        public (double lower, double upper)[] GetARCoefficientCredibleIntervals(double probability = 0.95)
        {
            var intervals = new (double lower, double upper)[_arCoefficients.Length];
            var z = 1.96; // For 95% confidence interval
            
            for (int i = 0; i < _arCoefficients.Length; i++)
            {
                var mean = _arCoefficients[i].GetMean();
                var std = Math.Sqrt(_arCoefficients[i].GetVariance());
                intervals[i] = (mean - z * std, mean + z * std);
            }
            return intervals;
        }

        public (double lower, double upper)[] GetExogCoefficientCredibleIntervals(double probability = 0.95)
        {
            var intervals = new (double lower, double upper)[_exogCoefficients.Length];
            var z = 1.96; // For 95% confidence interval
            
            for (int i = 0; i < _exogCoefficients.Length; i++)
            {
                var mean = _exogCoefficients[i].GetMean();
                var std = Math.Sqrt(_exogCoefficients[i].GetVariance());
                intervals[i] = (mean - z * std, mean + z * std);
            }
            return intervals;
        }

        public (double lower, double upper) GetInterceptCredibleInterval(double probability = 0.95)
        {
            var z = 1.96; // For 95% confidence interval
            var mean = _intercept.GetMean();
            var std = Math.Sqrt(_intercept.GetVariance());
            return (mean - z * std, mean + z * std);
        }

        /// <summary>
        /// Print a summary of the ARX model posterior
        /// </summary>
        public void PrintSummary()
        {
            System.Console.WriteLine($"ARX({_arOrder},{_exogOrder}) Model Posterior Summary (using Infer.NET)");
            System.Console.WriteLine("===============================================");
            
            var interceptInterval = GetInterceptCredibleInterval();
            System.Console.WriteLine($"Intercept: {GetIntercept():F4} [{interceptInterval.lower:F4}, {interceptInterval.upper:F4}]");
            
            var arIntervals = GetARCoefficientCredibleIntervals();
            for (int i = 0; i < _arOrder; i++)
            {
                System.Console.WriteLine($"φ{i + 1}: {GetARCoefficients()[i]:F4} [{arIntervals[i].lower:F4}, {arIntervals[i].upper:F4}]");
            }
            
            var exogIntervals = GetExogCoefficientCredibleIntervals();
            for (int varIdx = 0; varIdx < _exogCount; varIdx++)
            {
                var coeffs = GetExogCoefficientsForVariable(varIdx);
                int startIdx = varIdx * _exogOrder;
                
                for (int lag = 0; lag < _exogOrder; lag++)
                {
                    System.Console.WriteLine($"β{varIdx + 1},{lag + 1}: {coeffs[lag]:F4} [{exogIntervals[startIdx + lag].lower:F4}, {exogIntervals[startIdx + lag].upper:F4}]");
                }
            }
            
            System.Console.WriteLine($"Noise Variance: {GetNoiseVariance():F4}");
        }
    }
}
