﻿using OpenAI.Files;
using System.Diagnostics;
using System.Text.Json;

namespace HyperVectorDBExample
{
    /// <summary>
    /// Example program demonstrating the usage of HyperVectorDB
    /// This example shows how to:
    /// 1. Initialize and manage a vector database
    /// 2. Index documents and files
    /// 3. Perform similarity searches
    /// 4. Use custom preprocessing for document indexing
    /// </summary>
    internal class Program
    {
        // Global database instance
        private static HyperVectorDB.HyperVectorDB? DB;

        // State variable for markdown preprocessing
        private static bool skippingBlock = false;

        private static string LoadApiKey()
        {
            try
            {
                string secretsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "secrets.json");
                if (!File.Exists(secretsPath))
                {
                    Console.WriteLine("secrets.json not found. Creating template file...");
                    var template = new Dictionary<string, string>
                    {
                        { "GeminiApiKey", "YOUR_GEMINI_API_KEY_HERE" }
                    };
                    string templateJson = JsonSerializer.Serialize(template, new JsonSerializerOptions { WriteIndented = true });
                    File.WriteAllText(secretsPath, templateJson);
                    Console.WriteLine($"Template secrets.json created at: {secretsPath}");
                    Console.WriteLine("Please update the GeminiApiKey in secrets.json with your actual API key.");
                    throw new FileNotFoundException("Please update secrets.json with your Gemini API key and restart the application.");
                }

                string jsonContent = File.ReadAllText(secretsPath);
                var secrets = JsonSerializer.Deserialize<Dictionary<string, string>>(jsonContent);

                if (secrets == null || !secrets.ContainsKey("GeminiApiKey"))
                {
                    throw new KeyNotFoundException("GeminiApiKey not found in secrets.json");
                }

                string apiKey = secrets["GeminiApiKey"];
                if (apiKey == "YOUR_GEMINI_API_KEY_HERE")
                {
                    throw new InvalidOperationException("Please update the GeminiApiKey in secrets.json with your actual API key.");
                }

                return apiKey;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading API key: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Custom preprocessor for document indexing that handles markdown-specific formatting
        /// Filters out:
        /// - Empty lines
        /// - YAML frontmatter
        /// - Code blocks
        /// - Metadata lines
        /// - Annotation lines
        /// - Index links
        /// </summary>
        private static string? CustomPreprocessor(string line, string? path, int? lineNumber)
        {
            if (string.IsNullOrWhiteSpace(line)) { return null; }

            if (path != null && path.ToUpperInvariant().EndsWith(".MD"))
            {
                // Skip YAML frontmatter
                if (line.Contains("---"))
                {
                    skippingBlock = false;
                    return null;
                }
                // Handle code blocks
                else if (line.Contains("```"))
                {
                    skippingBlock = !skippingBlock;
                    return null;
                }
                // Skip metadata and special markdown lines
                else if (line.EndsWith("aliases: ") ||
                        line.Contains("date created:") ||
                        line.Contains("date modified:") ||
                        (line.EndsWith(":") && !line.StartsWith("#")))
                {
                    return null;
                }

                // Skip annotation lines
                if (line.Contains("%%")) { return null; }

                // Skip index links
                if (line.Trim().StartsWith("[[") && line.Trim().EndsWith("]]")) { return null; }

                // Skip content within code blocks
                if (skippingBlock) { return null; }
            }

            return line.Trim();
        }

        /// <summary>
        /// Custom postprocessor that adds file path and line number information to each line
        /// </summary>
        private static string? CustomPostprocessor(string line, string? path, int? lineNumber)
        {
            if (path == null) { return line; }
            return $"{path!}|{lineNumber}";
        }

        /// <summary>
        /// Main entry point of the example program
        /// </summary>
        static void Main()
        {
            try
            {
                InitializeDatabase();
                if (DB != null)
                {
                    RunInteractiveSearch();
                }
                else
                {
                    Console.WriteLine("Failed to initialize database. Exiting...");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred: {ex.Message}");
            }
        }

        /// <summary>
        /// Initializes the database, either loading existing data or creating new test data
        /// </summary>
        private static void InitializeDatabase()
        {
            string apiKey = LoadApiKey();

            // Initialize database with Gemini embedder, 32 dimensions
            DB = new HyperVectorDB.HyperVectorDB(new HyperVectorDB.Embedder.Gemini(apiKey), "TestDatabase", 32);

            if (Directory.Exists("TestDatabase"))
            {
                Console.WriteLine("Loading existing database...");
                DB.Load();
            }
            else
            {
                Console.WriteLine("Creating new database...");
                CreateTestDatabase();
                IndexDocumentFiles();
            }
        }

        /// <summary>
        /// Creates initial test data in the database
        /// </summary>
        private static void CreateTestDatabase()
        {
            if (DB == null) throw new InvalidOperationException("Database not initialized");

            DB.CreateIndex("TestIndex");

            // Create test documents about different animals
            string[] testDocuments = new string[]
            {
                "This is a test document about dogs",
                "This is a test document about cats",
                "This is a test document about fish",
                "This is a test document about birds",
                "This is a test document about dogs and cats",
                "This is a test document about cats and fish",
                "This is a test document about fish and birds",
                "This is a test document about birds and dogs",
                "This is a test document about dogs and cats and fish",
                "This is a test document about cats and fish and birds",
                "This is a test document about fish and birds and dogs",
                "This is a test document about birds and dogs and cats",
                "This is a test document about dogs and cats and fish and birds",
                "This is a test document about cats and fish and birds and dogs",
                "This is a test document about fish and birds and dogs and cats",
                "This is a test document about birds and dogs and cats and fish"
            };

            foreach (var doc in testDocuments)
            {
                DB.IndexDocument(doc);
            }

            // Example: Remove all documents containing the word "test"
            int removed = db.Purge(doc => doc.Contains("cats"));

            // Example: Remove all empty documents
            int removed = db.Purge(doc => string.IsNullOrWhiteSpace(doc));

            // Example: Remove documents matching a regex pattern
            int removed = db.Purge(doc => System.Text.RegularExpressions.Regex.IsMatch(doc, @"\b\d{3}-\d{2}-\d{4}\b"));

            DB.Save();
        }

        /// <summary>
        /// Indexes all files in the TestDocuments directory
        /// </summary>
        private static void IndexDocumentFiles()
        {
            if (DB == null) throw new InvalidOperationException("Database not initialized");

            string[] files = Directory.GetFiles(@".\TestDocuments", "*.*", SearchOption.AllDirectories);
            Console.WriteLine($"Indexing {files.Length} files...");

            int processedCount = 0;
            foreach (string file in files)
            {
                Console.WriteLine($"Processing: {file}");
                DB.IndexDocumentFile(file, CustomPreprocessor, CustomPostprocessor);
                processedCount++;

                // Save progress every 10 files
                if (processedCount % 10 == 0)
                {
                    DB.Save();
                }
            }

            DB.Save();
        }

        /// <summary>
        /// Runs an interactive search loop allowing users to query the database
        /// </summary>
        private static void RunInteractiveSearch()
        {
            if (DB == null) throw new InvalidOperationException("Database not initialized");

            while (true)
            {
                Console.WriteLine("\nEnter a search term (or 'exit' to quit):");
                var searchTerm = Console.ReadLine();

                if (string.IsNullOrEmpty(searchTerm)) continue;
                if (searchTerm.ToLower() == "exit") break;

                var sw = Stopwatch.StartNew();
                var result = DB.QueryCosineSimilarity(searchTerm, 10);
                sw.Stop();

                Console.WriteLine("\nSearch Results:");
                for (var i = 0; i < result.Documents.Count; i++)
                {
                    Console.WriteLine($"{result.Documents[i].DocumentString} (Score: {result.Distances[i]:F4})");
                }
                Console.WriteLine($"\nSearch completed in {sw.ElapsedMilliseconds}ms");
            }

            Console.WriteLine("\nPress Enter to exit...");
            Console.ReadLine();
        }
    }
}