using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

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

        /// <summary>
        /// Initializes a new instance of the Gemini class.
        /// </summary>
        /// <param name="apiKey">The API key for accessing Google's Gemini API.</param>
        /// <param name="model">The model to use for generating embeddings. Defaults to "gemini-embedding-exp-03-07".</param>
        public Gemini(string apiKey, string model = "gemini-embedding-exp-03-07")
        {
            _apiKey = apiKey;
            _model = model;
            _client = new HttpClient();
            _client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        }

        /// <inheritdoc/>
        public double[] GetVector(string Document)
        {
            var request = new
            {
                model = $"models/{_model}",
                content = new
                {
                    parts = new[]
                    {
                        new { text = Document }
                    }
                }
            };

            var response = SendRequest(request).GetAwaiter().GetResult();
            return response.embedding.ToArray();
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
            var url = $"https://generativelanguage.googleapis.com/v1beta/models/{_model}:embedContent?key={_apiKey}";
            var content = new StringContent(JsonSerializer.Serialize(request), Encoding.UTF8, "application/json");
            
            var response = await _client.PostAsync(url, content);
            response.EnsureSuccessStatusCode();
            
            var responseJson = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<GeminiResponse>(responseJson)!;
        }
    }

    /// <summary>
    /// Represents the response from the Gemini API embedding endpoint.
    /// </summary>
    class GeminiResponse
    {
        /// <summary>
        /// The embedding vector.
        /// </summary>
        public List<double> embedding { get; set; } = new List<double>();
    }
}