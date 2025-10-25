using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;
using System;
using System.Linq;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace InferNetAR.Console
{
    /// <summary>
    /// AR(p) Autoregressive Model implementation using Infer.NET
    /// </summary>
    public class ARModel
    {
        private readonly int _order;

        public ARModel(int order)
        {
            _order = order;
        }

        /// <summary>
        /// Fit the AR(p) model to the given time series data using Infer.NET
        /// </summary>
        public ARPosterior Fit(double[] data)
        {
            if (data.Length <= _order)
                throw new ArgumentException($"Data length ({data.Length}) must be greater than AR order ({_order})");

            return BuildARModel(data, _order);
        }

        /// <summary>
        /// Generate sample AR(p) data for testing
        /// </summary>
        public static double[] GenerateSampleARData(double[] coefficients, double noiseVariance, int length, int seed = 42)
        {
            var random = new Random(seed);
            var data = new double[length];
            int order = coefficients.Length - 1;
            
            for (int i = 0; i < order; i++)
                data[i] = 0.0;

            for (int t = order; t < length; t++)
            {
                double value = coefficients[0];
                for (int i = 1; i < coefficients.Length; i++)
                    value += coefficients[i] * data[t - i];
                value += Math.Sqrt(noiseVariance) * random.NextGaussian();
                data[t] = value;
            }

            return data;
        }

        /// <summary>
        /// Generate sample AR(p) data with default parameters
        /// </summary>
        public static double[] GenerateSampleARData(int order, int length, int seed = 42)
        {
            var coefficients = new double[order + 1];
            coefficients[0] = 0.0;
            if (order >= 1) coefficients[1] = 0.6;
            if (order >= 2) coefficients[2] = -0.3;
            if (order >= 3) coefficients[3] = 0.1;
            
            return GenerateSampleARData(coefficients, 1.0, length, seed);
        }

        /// <summary>
        /// Build AR(p) model using Infer.NET - simple working version
        /// </summary>
        private ARPosterior BuildARModel(double[] observedSeries, int p)
        {
            int T = observedSeries.Length;
            int n = T - p;
            
            // Model using VectorGaussian for efficiency (including intercept)
            var phiMean = Vector.Zero(p + 1); // +1 for intercept
            var phiVariance = PositiveDefiniteMatrix.Identity(p + 1);
            var phi = Variable.VectorGaussianFromMeanAndVariance(phiMean, phiVariance).Named("phi");
            var prec = Variable.GammaFromShapeAndRate(1.0, 1.0).Named("precision");
            
            // Create design matrix and response (including intercept)
            var XData = new Vector[n];
            var yData = new double[n];
            
            for (int i = 0; i < n; i++)
            {
                var row = new double[p + 1]; // +1 for intercept
                row[0] = 1.0; // intercept term
                for (int j = 0; j < p; j++)
                    row[j + 1] = observedSeries[p - 1 - j + i];
                XData[i] = Vector.FromArray(row);
                yData[i] = observedSeries[p + i];
            }
            
            Range dataRange = new Range(n).Named("dataRange");
            var X = Variable.Observed(XData, dataRange).Named("X");
            var y = Variable.Observed(yData, dataRange).Named("y");
            
            // Model: y[i] = X[i] · phi + noise
            using (Variable.ForEach(dataRange))
            {
                var mean = Variable.InnerProduct(X[dataRange], phi).Named("mean");
                y[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, prec);
            }
            
            // Inference
            var engine = new InferenceEngine();
            engine.NumberOfIterations = 50;
            
            var phiPost = engine.Infer<VectorGaussian>(phi);
            var precPost = engine.Infer<Gamma>(prec);
            
            // Convert to individual Gaussian distributions
            var phiMeanPost = phiPost.GetMean();
            var phiVarPost = phiPost.GetVariance();
            
            var coefficientDistributions = new Gaussian[p + 1];
            
            // Now all coefficients (including intercept) are estimated
            for (int i = 0; i < p + 1; i++)
                coefficientDistributions[i] = new Gaussian(phiMeanPost[i], phiVarPost[i, i]);
            
            var sigma2Post = new Gamma(precPost.Shape, 1.0 / precPost.Rate);
            
            return new ARPosterior(coefficientDistributions, sigma2Post, coefficientDistributions[0], p);
        }
    }

    /// <summary>
    /// Posterior distributions from AR(p) model fitting
    /// </summary>
    public class ARPosterior
    {
        private readonly Gaussian[] _coefficientPosterior;
        private readonly Gamma _noiseVariancePosterior;
        private readonly Gaussian _interceptPosterior;
        private readonly int _order;

        public ARPosterior(Gaussian[] coefficientPosterior, Gamma noiseVariancePosterior, Gaussian interceptPosterior, int order)
        {
            _coefficientPosterior = coefficientPosterior;
            _noiseVariancePosterior = noiseVariancePosterior;
            _interceptPosterior = interceptPosterior;
            _order = order;
        }

        public double[] GetCoefficients()
        {
            var coefficients = new double[_order + 1];
            coefficients[0] = _interceptPosterior.GetMean();
            
            for (int i = 0; i < _order; i++)
                coefficients[i + 1] = _coefficientPosterior[i + 1].GetMean();
            
            return coefficients;
        }

        public double GetNoiseVariance() => _noiseVariancePosterior.GetMean();

        public double GetIntercept() => _interceptPosterior.GetMean();

        public (double lower, double upper)[] GetCoefficientCredibleIntervals(double probability = 0.95)
        {
            var intervals = new (double lower, double upper)[_order + 1];
            var z = 1.96;
            
            var interceptStd = Math.Sqrt(_interceptPosterior.GetVariance());
            intervals[0] = (_interceptPosterior.GetMean() - z * interceptStd, 
                           _interceptPosterior.GetMean() + z * interceptStd);
            
            for (int i = 0; i < _order; i++)
            {
                var coeffStd = Math.Sqrt(_coefficientPosterior[i + 1].GetVariance());
                intervals[i + 1] = (_coefficientPosterior[i + 1].GetMean() - z * coeffStd,
                                   _coefficientPosterior[i + 1].GetMean() + z * coeffStd);
            }
            
            return intervals;
        }

        public void PrintSummary()
        {
            System.Console.WriteLine("AR(p) Model Posterior Summary (using Infer.NET)");
            System.Console.WriteLine("================================================");
            
            var coefficients = GetCoefficients();
            var intervals = GetCoefficientCredibleIntervals();
            
            System.Console.WriteLine($"Intercept: {coefficients[0]:F4} [{intervals[0].lower:F4}, {intervals[0].upper:F4}]");
            
            for (int i = 0; i < _order; i++)
                System.Console.WriteLine($"φ{i + 1}: {coefficients[i + 1]:F4} [{intervals[i + 1].lower:F4}, {intervals[i + 1].upper:F4}]");
            
            System.Console.WriteLine($"Noise Variance: {GetNoiseVariance():F4}");
        }
    }

    public static class RandomExtensions
    {
        public static double NextGaussian(this Random random, double mean = 0.0, double stdDev = 1.0)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
    }
}