# Infer.NET Time Series Models

This project demonstrates how to fit AR(p) (autoregressive) and ARX(p,q) (autoregressive with exogenous inputs) models using Microsoft's Infer.NET probabilistic programming framework in .NET 8.

## Models Overview

### AR(p) Model

An autoregressive model of order p, denoted AR(p), is a time series model where the current value depends linearly on the previous p values plus a random error term. The mathematical form is:

```
X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φ_pX_{t-p} + ε_t
```

Where:
- `X_t` is the value at time t
- `c` is a constant (intercept)
- `φ₁, φ₂, ..., φ_p` are the autoregressive coefficients
- `ε_t` is white noise (random error term)
- `p` is the order of the autoregressive model

### ARX(p,q) Model

An autoregressive model with exogenous inputs of orders p and q, denoted ARX(p,q), extends the AR(p) model by including external variables with lagged effects. The mathematical form is:

```
X_t = c + φ₁X_{t-1} + φ₂X_{t-2} + ... + φ_pX_{t-p} + Σⱼ β₁ⱼU_{1,t-j} + Σⱼ β₂ⱼU_{2,t-j} + ... + Σⱼ βₖⱼU_{k,t-j} + ε_t
```

Or more explicitly:
```
X_t = c + φ₁X_{t-1} + ... + φ_pX_{t-p} + β₁₁U_{1,t-1} + ... + β₁ₚU_{1,t-q} + β₂₁U_{2,t-1} + ... + β₂ₚU_{2,t-q} + ... + ε_t
```

Where:
- `X_t` is the value at time t
- `c` is a constant (intercept)
- `φ₁, φ₂, ..., φ_p` are the autoregressive coefficients
- `βᵢⱼ` are the exogenous variable coefficients for variable i at lag j
- `U_{i,t-j}` are the exogenous variables at time t-j
- `ε_t` is white noise (random error term)
- `p` is the order of the autoregressive model
- `q` is the order of exogenous variable lags
- `k` is the number of exogenous variables

## Key Features

- **Dual Model Support**: Both AR(p) and ARX(p,q) model implementations
- **Flexible Lag Structure**: ARX(p,q) supports separate lag orders for AR and exogenous variables
- **CSV Input Support**: Read time series data from CSV files via command-line interface
- **Bayesian Inference**: Uses Infer.NET's probabilistic programming capabilities for robust parameter estimation
- **Uncertainty Quantification**: Provides credible intervals and posterior distributions for model parameters
- **Model Selection**: Supports different AR orders and model comparison
- **Exogenous Variables**: ARX model supports multiple external input variables with lagged effects
- **Modular Design**: Clean, well-structured code ready for unit testing
- **Time Series Analysis**: Designed specifically for temporal data analysis

## Quick Start

**Fit an AR(2) model from CSV:**
```bash
dotnet run --project InferNetAR.Console -- --input data.csv --model ar --ar-order 2
```

**Fit an ARX(2,1) model from CSV:**
```bash
dotnet run --project InferNetAR.Console -- --input data.csv --model arx --ar-order 2 --exog-order 1 --exog-count 2
```

**See help for all options:**
```bash
dotnet run --project InferNetAR.Console -- --help
```

## Project Structure

```
InferNetAR/
├── InferNetAR.Console/          # Main console application
│   ├── Program.cs               # Entry point with CSV input and CLI
│   ├── ARModel.cs               # AR(p) model implementation
│   ├── ARXModel.cs              # ARX(p,q) model implementation
│   └── InferNetAR.Console.csproj
├── InferNetAR.sln               # Solution file
├── sample_ar.csv                # Sample data for AR model
├── sample_arx.csv               # Sample data for ARX model
└── README.md                    # This file
```

## Getting Started

### Prerequisites

- .NET 8 SDK
- Visual Studio 2022 or VS Code with C# extension

### Installation

1. Clone the repository
2. Navigate to the project directory
3. Restore packages:
   ```bash
   dotnet restore
   ```

### Running the Application

The application can be run in two ways:

**1. From command-line with CSV input:**
```bash
# Fit AR(2) model
dotnet run --project InferNetAR.Console -- --input data.csv --model ar --ar-order 2

# Fit ARX(2,1) model
dotnet run --project InferNetAR.Console -- --input data.csv --model arx --ar-order 2 --exog-order 1 --exog-count 2
```

**2. As a library** - See "Programming API Usage" section above for code examples.

## AR(p) Model Implementation

The project includes a complete implementation of AR(p) model fitting using Infer.NET:

- **Parameter Estimation**: Bayesian inference for autoregressive coefficients using variational message passing
- **Noise Modeling**: Proper handling of the error term distribution using Gamma priors on precision
- **Vectorized Implementation**: Efficient computation using VectorGaussian distributions
- **Credible Intervals**: Full posterior distributions with uncertainty quantification

### Model Details

The implementation treats the AR(p) model as a **Bayesian linear regression problem**:

1. **Design Matrix Construction**: For each time point t > p, we create a feature vector containing the intercept and p previous observations:
   ```
   X[i] = [1, x_{t-1}, x_{t-2}, ..., x_{t-p}]
   ```

2. **Priors**:
   - All coefficients (including intercept): `θ ~ VectorGaussian(0, I)` - multivariate normal with identity covariance
   - Noise precision: `τ ~ Gamma(1, 1)` - converted to variance σ² = 1/τ

3. **Likelihood**:
   ```
   y_t ~ Gaussian(X_t · θ, τ⁻¹)
   ```
   where `X_t · θ` is the inner product of the feature vector and coefficients, giving us the full AR(p) model with intercept: `y_t = c + φ₁x_{t-1} + φ₂x_{t-2} + ... + φ_px_{t-p} + ε_t`

4. **Inference**: Uses Expectation Propagation (EP) algorithm to compute posterior distributions.

### Key Implementation Aspects

**Lagging Logic**: For an AR(2) model at time t:
```csharp
// Row i contains: [1, x_{t-1}, x_{t-2}]
row[0] = 1.0;                                    // intercept term
row[1] = observedSeries[p - 1 + i];              // lag-1: x_{t-1}
row[2] = observedSeries[p - 2 + i];              // lag-2: x_{t-2}
```

The model then fits: `x_t = c + φ₁ · x_{t-1} + φ₂ · x_{t-2} + ε_t`

**Vectorization**: Instead of loops, uses Infer.NET's `InnerProduct` operation:
```csharp
var mean = Variable.InnerProduct(X[dataRange], phi);
y[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, prec);
```

This approach is both more efficient and avoids unimplemented operations in Infer.NET.

## Usage

### Command-Line Interface

The application supports reading time series data from CSV files and fitting models via command-line arguments.

#### Getting Help

```bash
dotnet run --project InferNetAR.Console -- --help
```

#### Basic Usage Examples

**Fit AR(2) model:**
```bash
dotnet run --project InferNetAR.Console -- --input data.csv --model ar --ar-order 2
```

**Fit ARX(2,1) model with 2 exogenous variables:**
```bash
dotnet run --project InferNetAR.Console -- --input data.csv --model arx --ar-order 2 --exog-order 1 --exog-count 2
```

#### Command-Line Arguments

**Required Arguments:**
- `--input <file>` - Path to CSV file containing time series data
- `--model <type>` - Model type: `ar` or `arx`

**AR Model Parameters:**
- `--ar-order <n>` - AR model order (p) **[mandatory for AR]**

**ARX Model Parameters:**
- `--ar-order <n>` - AR model order (p) **[mandatory for ARX]**
- `--exog-order <n>` - Exogenous lag order (q) **[mandatory for ARX]**
- `--exog-count <n>` - Number of exogenous variables (k) **[mandatory for ARX]**

**Optional Parameters:**
- `--noise-var <val>` - Prior for noise variance (default: 1.0)
- `--coeff-var <val>` - Prior for coefficient variance (default: 10.0)

#### CSV File Format

**For AR Model** - Single column format:
```csv
value
0.000
0.000
-1.150
-0.616
-0.630
...
```

**For ARX Model** - Multi-column format:
```csv
endogenous,exog1,exog2
0.000,-1.150,-1.372
0.000,0.074,-0.798
-1.150,-0.605,1.459
...
```

**Notes:**
- First row can be a header row (will be automatically skipped)
- For AR model: Only the first column is used
- For ARX model: First column = endogenous series, subsequent columns = exogenous variables
- All series must have the same length

#### Example Output

**AR Model:**
```
Loaded 20 data points from sample_ar.csv
Sample values: [0.000, 0.000, -1.150, -0.616, -0.630, -0.063, 0.767, -0.251, -1.454, -1.774...]

Fitting AR(2) model using Infer.NET...
Compiling model...done.
Iterating: 
.........|.........|.........|.........|.........| 50
Model fitting completed!

AR(p) Model Posterior Summary (using Infer.NET)
================================================
Intercept: -0.2439 [-0.5728, 0.0849]
φ1: 0.6061 [0.1256, 1.0866]
φ2: -0.5383 [-1.0162, -0.0605]
Noise Variance: 2.6911

Model Statistics:
- Intercept: -0.2439
- AR(1) coefficient: 0.6061
- AR(2) coefficient: -0.5383
- Noise variance: 2.6911
```

**ARX Model:**
```
Loaded 10 data points from sample_arx.csv
Sample values: [0.000, 0.000, -1.150, -0.616, -0.630, -0.063, 0.767, -0.251, -1.454, -1.774...]

Fitting ARX(2,1) model using Infer.NET...
Loaded 10 endogenous observations
Loaded 2 exogenous variables

Compiling model...done.
Iterating: 
.........|.........|.........|.........|.........| 50
Model fitting completed!

ARX(2,1) Model Posterior Summary (using Infer.NET)
===============================================
Intercept: -0.5370 [-1.1885, 0.1145]
φ1: 0.7115 [-0.1962, 1.6191]
φ2: -0.6661 [-1.6642, 0.3321]
β1,1: -0.1947 [-1.6061, 1.2167]
β2,1: 0.3973 [-0.3855, 1.1802]
Noise Variance: 0.7867

Model Statistics:
- Intercept: -0.5370
- AR(1) coefficient: 0.7115
- AR(2) coefficient: -0.6661
- Exog1(lag1) coefficient: -0.1947
- Exog2(lag1) coefficient: 0.3973
- Noise variance: 0.7867
```

### Programming API Usage

#### AR(p) Model Usage

```csharp
using InferNetAR.Console;

// Generate sample AR(2) data
var data = ARModel.GenerateSampleARData(order: 2, length: 100, seed: 42);

// Fit AR(2) model
var arModel = new ARModel(order: 2);
var arPosterior = arModel.Fit(data);

// Print summary of results
arPosterior.PrintSummary();
// Output:
// AR(p) Model Posterior Summary (using Infer.NET)
// ================================================
// Intercept: 0.0131 [-0.1674, 0.1937]
// φ1: 0.5316 [0.3446, 0.7185]
// φ2: -0.3907 [-0.5827, -0.1988]
// Noise Variance: 1.2182

// Get parameter estimates
var coefficients = arPosterior.GetCoefficients();
var noiseVariance = arPosterior.GetNoiseVariance();

Console.WriteLine($"Intercept: {coefficients[0]:F4}");
Console.WriteLine($"AR(1) coefficient: {coefficients[1]:F4}");
Console.WriteLine($"AR(2) coefficient: {coefficients[2]:F4}");
Console.WriteLine($"Noise variance: {noiseVariance:F4}");
```

#### ARX(p,q) Model Usage

```csharp
using InferNetAR.Console;

// Generate sample ARX(2,1) data with 2 exogenous variables
var arxData = ARXModel.GenerateSampleARXData(order: 2, exogCount: 2, length: 100, seed: 42);

// Generate exogenous variables
var random = new Random(42);
var exogData = new double[2][];
for (int i = 0; i < 2; i++)
{
    exogData[i] = new double[100];
    for (int t = 0; t < 100; t++)
    {
        exogData[i][t] = random.NextGaussian();
    }
}

// Fit ARX(2,1) model - AR order 2, exogenous lag order 1
var arxModel = new ARXModel(arOrder: 2, exogOrder: 1, exogenousCount: 2);
var arxPosterior = arxModel.Fit(arxData, exogData);

// Print summary of results
arxPosterior.PrintSummary();
// Output:
// ARX(2,1) Model Posterior Summary (using Infer.NET)
// ================================================
// Intercept: 0.0100 [-0.1677, 0.1877]
// φ1: 0.5291 [0.3451, 0.7131]
// φ2: -0.3865 [-0.5753, -0.1977]
// β1,1: -0.0794 [-0.2779, 0.1190]
// β2,1: 0.1902 [0.0128, 0.3677]
// Noise Variance: 0.8094

// Get parameter estimates
var arCoeffs = arxPosterior.GetARCoefficients();
var exogCoeffs = arxPosterior.GetExogCoefficients();
var intercept = arxPosterior.GetIntercept();
var noiseVariance = arxPosterior.GetNoiseVariance();

Console.WriteLine($"Intercept: {intercept:F4}");
Console.WriteLine($"AR(1) coefficient: {arCoeffs[0]:F4}");
Console.WriteLine($"AR(2) coefficient: {arCoeffs[1]:F4}");
Console.WriteLine($"Exog1(lag1) coefficient: {exogCoeffs[0]:F4}");
Console.WriteLine($"Exog2(lag1) coefficient: {exogCoeffs[1]:F4}");
Console.WriteLine($"Noise variance: {noiseVariance:F4}");

// Access coefficients for specific variable
var exog1Coeffs = arxPosterior.GetExogCoefficientsForVariable(0);
Console.WriteLine($"Exog1 coefficients: [{string.Join(", ", exog1Coeffs.Select(x => x.ToString("F3")))}]");
```

## API Reference

### ARModel Class

#### Constructor
```csharp
ARModel(int order, double noiseVariancePrior = 1.0, double coefficientVariancePrior = 10.0)
```

#### Methods
- `ARPosterior Fit(double[] observedSeries)` - Fit AR model to data
- `static double[] GenerateSampleARData(int order, int length, int seed = 42)` - Generate sample data
- `static double[] GenerateSampleARData(double[] coefficients, double noiseVariance, int length, int seed = 42)` - Generate custom data

### ARPosterior Class

#### Methods
- `double[] GetCoefficients()` - Get coefficient estimates
- `double GetNoiseVariance()` - Get noise variance estimate
- `Gaussian[] GetCoefficientsDistributions()` - Get full posterior distributions
- `(double lower, double upper)[] GetCoefficientCredibleIntervals(double probability = 0.95)` - Get credible intervals
- `void PrintSummary()` - Print formatted summary

### ARXModel Class

#### Constructor
```csharp
ARXModel(int arOrder, int exogOrder, int exogenousCount, double noiseVariancePrior = 1.0, double coefficientVariancePrior = 10.0)
```

#### Properties
- `int ArOrder` - Gets the AR order (p)
- `int ExogOrder` - Gets the exogenous lag order (q)
- `int ExogenousCount` - Gets the number of exogenous variables (k)

#### Methods
- `ARXPosterior Fit(double[] observedSeries, double[][] exogenousData)` - Fit ARX(p,q) model to data
- `static double[] GenerateSampleARXData(int order, int exogCount, int length, int seed = 42)` - Generate sample data
- `static double[] GenerateSampleARXData(double[] arCoefficients, double[] exogCoefficients, double[][] exogenousData, double noiseVariance, int length, int seed = 42)` - Generate custom data

### ARXPosterior Class

#### Properties
- `int ArOrder` - Gets the AR order (p)
- `int ExogOrder` - Gets the exogenous lag order (q)
- `int ExogCount` - Gets the number of exogenous variables (k)

#### Methods
- `double[] GetARCoefficients()` - Get AR coefficient estimates
- `double[] GetExogCoefficients()` - Get all exogenous coefficient estimates (flattened: k variables × q lags)
- `double[] GetExogCoefficientsForVariable(int variableIndex)` - Get exogenous coefficients for a specific variable
- `double GetIntercept()` - Get intercept estimate
- `double GetNoiseVariance()` - Get noise variance estimate
- `Gaussian[] GetARCoefficientsDistributions()` - Get AR coefficient distributions
- `Gaussian[] GetExogCoefficientsDistributions()` - Get exogenous coefficient distributions
- `(double lower, double upper)[] GetARCoefficientCredibleIntervals(double probability = 0.95)` - Get AR coefficient intervals
- `(double lower, double upper)[] GetExogCoefficientCredibleIntervals(double probability = 0.95)` - Get exogenous coefficient intervals
- `void PrintSummary()` - Print formatted summary with ARX(p,q) notation

## Applications

AR(p) models are widely used in:

- **Economics**: GDP growth, inflation, unemployment rates
- **Finance**: Stock prices, exchange rates, volatility modeling
- **Engineering**: Signal processing, control systems
- **Climate Science**: Temperature, precipitation time series
- **Healthcare**: Patient monitoring, disease progression

## Advanced Features

- **Model Selection**: Automatic order selection using information criteria (AIC, BIC)
- **Seasonal AR Models**: Extension to SARIMA models
- **Multivariate AR**: Vector autoregressive (VAR) models
- **Non-linear Extensions**: Threshold AR, GARCH models

## References

1. Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). *Time Series Analysis: Forecasting and Control*
2. Hamilton, J.D. (1994). *Time Series Analysis*
3. Microsoft Research. (2024). *Infer.NET: A Framework for Probabilistic Programming*

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on GitHub.
