using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace HyperVectorDB.Embedder
{
    /// <summary>
    /// Implementation of IEmbedder that uses Google's Gemini API to convert text into vector representations.
    /// </summary>
    public class Gemini : IEmbedder
    {
        private readonly HttpClient _client;
        private readonly string _apiKey;
        private readonly string _model;
        private static readonly Stopwatch _rateLimitTimer = new Stopwatch();
        private static readonly object _rateLimitLock = new object();
        private static int _minimumRequestIntervalMs = 20000; // Default 20 seconds between requests

        /// <summary>
        /// Gets or sets the minimum interval in milliseconds between API requests.
        /// Default is 20000ms (20 seconds).
        /// </summary>
        public static int MinimumRequestIntervalMs
        {
            get => _minimumRequestIntervalMs;
            set
            {
                if (value < 0)
                {
                    throw new ArgumentException("Minimum request interval cannot be negative", nameof(value));
                }
                _minimumRequestIntervalMs = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the Gemini class.
        /// </summary>
        /// <param name="apiKey">The API key for accessing Google's Gemini API.</param>
        /// <param name="model">The model to use for generating embeddings. Defaults to "gemini-embedding-exp-03-07".</param>
        public Gemini(string apiKey, string model = "gemini-embedding-exp-03-07")
        {
            if (string.IsNullOrEmpty(apiKey))
            {
                throw new ArgumentException("API key cannot be null or empty", nameof(apiKey));
            }

            _apiKey = apiKey;
            _model = model;
            _client = new HttpClient();
            _client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
            
            // Start the rate limit timer if it hasn't been started
            if (!_rateLimitTimer.IsRunning)
            {
                _rateLimitTimer.Start();
            }
        }

        private static async Task WaitForRateLimit()
        {
            lock (_rateLimitLock)
            {
                if (_rateLimitTimer.ElapsedMilliseconds < _minimumRequestIntervalMs)
                {
                    var delayMs = _minimumRequestIntervalMs - _rateLimitTimer.ElapsedMilliseconds;
                    _rateLimitTimer.Restart();
                    Task.Delay((int)delayMs).Wait();
                }
                else
                {
                    _rateLimitTimer.Restart();
                }
            }
        }

        /// <inheritdoc/>
        public double[] GetVector(string Document)
        {
            if (string.IsNullOrEmpty(Document))
            {
                throw new ArgumentException("Document cannot be null or empty", nameof(Document));
            }

            try
            {
                WaitForRateLimit().Wait(); // Wait for rate limit before making request

                var request = new
                {
                    model = $"models/{_model}",
                    content = new
                    {
                        parts = new[]
                        {
                            new { text = Document }
                        }
                    },
                    taskType = "SEMANTIC_SIMILARITY"
                };

                var response = SendRequest(request).GetAwaiter().GetResult();
                return response.GetEmbeddingValues().ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in GetVector: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                throw;
            }
        }

        /// <inheritdoc/>
        public double[][] GetVectors(string[] Documents)
        {
            var vectors = new List<double[]>();
            foreach (var document in Documents)
            {
                vectors.Add(GetVector(document));
            }
            return vectors.ToArray();
        }

        private async Task<GeminiResponse> SendRequest(object request)
        {
            try
            {
                var url = $"https://generativelanguage.googleapis.com/v1beta/models/{_model}:embedContent?key={_apiKey}";
                Console.WriteLine($"Sending request to: {url}");
                
                var jsonRequest = JsonSerializer.Serialize(request);
                Console.WriteLine($"Request payload: {jsonRequest}");
                
                var content = new StringContent(jsonRequest, Encoding.UTF8, "application/json");
                
                var response = await _client.PostAsync(url, content);
                
                if (!response.IsSuccessStatusCode)
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"Error response: {errorContent}");
                    response.EnsureSuccessStatusCode();
                }
                
                var responseJson = await response.Content.ReadAsStringAsync();
                // The response JSON is quite large. Only uncomment these lines when necessary
                //Console.WriteLine($"Response received: {responseJson}");
                
                // Write full response to file for debugging
                //await File.WriteAllTextAsync("gemini-output.json", responseJson);
                
                return JsonSerializer.Deserialize<GeminiResponse>(responseJson)!;
            }
            catch (HttpRequestException ex)
            {
                Console.WriteLine($"HTTP Request Error: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                throw;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Unexpected error: {ex.Message}");
                throw;
            }
        }
    }

    /// <summary>
    /// Represents the response from the Gemini API embedding endpoint.
    /// </summary>
    class GeminiResponse
    {
        /// <summary>
        /// The embedding object containing the vector values.
        /// </summary>
        public EmbeddingObject embedding { get; set; } = new EmbeddingObject();

        /// <summary>
        /// Gets the embedding values as a list of doubles.
        /// </summary>
        public List<double> GetEmbeddingValues() => embedding.values;
    }

    /// <summary>
    /// Represents the embedding object in the Gemini API response.
    /// </summary>
    class EmbeddingObject
    {
        /// <summary>
        /// The embedding vector values.
        /// </summary>
        public List<double> values { get; set; } = new List<double>();
    }
}