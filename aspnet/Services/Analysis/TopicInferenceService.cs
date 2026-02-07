using System.Security.Cryptography;

namespace DlpKeywordPolicyGenerator.Web.Services.Analysis;

public sealed class TopicInferenceService
{
    public List<TopicCluster> InferTopics(
        IReadOnlyList<string> docsText,
        Dictionary<string, TermStats> stats,
        double dfRatioMin,
        double dfRatioMax,
        int maxTopics = 8)
    {
        if (docsText.Count == 0)
        {
            return new List<TopicCluster>();
        }

        if (docsText.Count == 1)
        {
            var phrases = stats
                .OrderByDescending(kv => kv.Value.Df)
                .ThenByDescending(kv => kv.Value.Tf)
                .Take(20)
                .Select(kv => new TopicCandidate { Phrase = kv.Key, DfRatio = kv.Value.DfRatio })
                .ToList();

            return new List<TopicCluster>
            {
                new()
                {
                    TopicName = "Single Document Topic",
                    DocIds = new List<int> { 0 },
                    Candidates = phrases
                }
            };
        }

        var filteredTerms = stats
            .Where(kv => kv.Value.DfRatio >= dfRatioMin && kv.Value.DfRatio <= dfRatioMax)
            .Where(kv => kv.Key.Length >= 3 && kv.Key.Length <= 60)
            .OrderByDescending(kv => kv.Value.DfRatio)
            .ThenByDescending(kv => kv.Value.Tf)
            .Take(5000)
            .Select(kv => kv.Key)
            .ToList();

        if (filteredTerms.Count == 0)
        {
            var fallback = stats
                .OrderByDescending(kv => kv.Value.Df)
                .ThenByDescending(kv => kv.Value.Tf)
                .Take(25)
                .Select(kv => new TopicCandidate { Phrase = kv.Key, DfRatio = kv.Value.DfRatio })
                .ToList();

            return new List<TopicCluster>
            {
                new()
                {
                    TopicName = "General Topic",
                    DocIds = Enumerable.Range(0, docsText.Count).ToList(),
                    Candidates = fallback
                }
            };
        }

        var vocabIndex = filteredTerms
            .Select((term, index) => new { term, index })
            .ToDictionary(x => x.term, x => x.index, StringComparer.Ordinal);

        var vectors = new List<double[]>();
        foreach (var doc in docsText)
        {
            var terms = TextProcessing.BuildDocTerms(doc);
            var counts = new double[vocabIndex.Count];
            foreach (var term in terms)
            {
                if (vocabIndex.TryGetValue(term, out var idx))
                {
                    counts[idx] += 1;
                }
            }

            var vector = new double[vocabIndex.Count];
            for (var i = 0; i < counts.Length; i++)
            {
                var term = filteredTerms[i];
                var df = stats[term].Df;
                var idf = Math.Log((docsText.Count + 1.0) / (df + 1.0)) + 1.0;
                vector[i] = counts[i] * idf;
            }

            NormalizeVector(vector);
            vectors.Add(vector);
        }

        var k = Math.Max(2, Math.Min(maxTopics, (int)Math.Round(Math.Sqrt(docsText.Count)) + 1));
        var (labels, centroids) = RunKMeans(vectors, k, nInit: 5, iterations: 10);

        var clusters = new Dictionary<int, List<int>>();
        for (var i = 0; i < labels.Length; i++)
        {
            clusters.TryAdd(labels[i], new List<int>());
            clusters[labels[i]].Add(i);
        }

        var topics = new List<TopicCluster>();
        foreach (var (label, docIds) in clusters)
        {
            var centroid = centroids[label];
            var indices = centroid
                .Select((value, idx) => new { value, idx })
                .OrderByDescending(x => x.value)
                .Select(x => x.idx)
                .Take(250)
                .ToList();

            var candidates = new List<TopicCandidate>();
            foreach (var idx in indices)
            {
                var phrase = filteredTerms[idx];
                if (!stats.TryGetValue(phrase, out var stat))
                {
                    continue;
                }

                if (stat.DfRatio < dfRatioMin || stat.DfRatio > dfRatioMax)
                {
                    continue;
                }

                if (phrase.Length < 3 || phrase.All(char.IsDigit))
                {
                    continue;
                }

                candidates.Add(new TopicCandidate { Phrase = phrase, DfRatio = stat.DfRatio });
                if (candidates.Count >= 40)
                {
                    break;
                }
            }

            topics.Add(new TopicCluster
            {
                TopicName = $"Topic {label + 1}",
                DocIds = docIds,
                Candidates = candidates
            });
        }

        return topics.OrderByDescending(t => t.DocIds.Count).Take(maxTopics).ToList();
    }

    private static void NormalizeVector(double[] vector)
    {
        var norm = Math.Sqrt(vector.Sum(v => v * v));
        if (norm <= 0)
        {
            return;
        }

        for (var i = 0; i < vector.Length; i++)
        {
            vector[i] /= norm;
        }
    }

    private static (int[] labels, double[][] centroids) RunKMeans(
        IReadOnlyList<double[]> vectors,
        int k,
        int nInit,
        int iterations)
    {
        var bestLabels = new int[vectors.Count];
        var bestCentroids = new double[k][];
        var bestScore = double.MaxValue;

        for (var init = 0; init < nInit; init++)
        {
            var (labels, centroids) = InitializeCentroids(vectors, k);
            for (var iter = 0; iter < iterations; iter++)
            {
                labels = AssignClusters(vectors, centroids);
                centroids = RecomputeCentroids(vectors, labels, k);
            }

            var score = TotalDistance(vectors, labels, centroids);
            if (score < bestScore)
            {
                bestScore = score;
                bestLabels = labels;
                bestCentroids = centroids;
            }
        }

        return (bestLabels, bestCentroids);
    }

    private static (int[] labels, double[][] centroids) InitializeCentroids(IReadOnlyList<double[]> vectors, int k)
    {
        var labels = new int[vectors.Count];
        var centroids = new double[k][];
        var indices = Enumerable.Range(0, vectors.Count).OrderBy(_ => RandomNumberGenerator.GetInt32(int.MaxValue)).ToList();
        for (var i = 0; i < k; i++)
        {
            centroids[i] = (double[])vectors[indices[i % vectors.Count]].Clone();
        }

        return (labels, centroids);
    }

    private static int[] AssignClusters(IReadOnlyList<double[]> vectors, double[][] centroids)
    {
        var labels = new int[vectors.Count];
        for (var i = 0; i < vectors.Count; i++)
        {
            var best = 0;
            var bestDistance = double.MaxValue;
            for (var c = 0; c < centroids.Length; c++)
            {
                var distance = CosineDistance(vectors[i], centroids[c]);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    best = c;
                }
            }
            labels[i] = best;
        }
        return labels;
    }

    private static double[][] RecomputeCentroids(IReadOnlyList<double[]> vectors, int[] labels, int k)
    {
        var dim = vectors[0].Length;
        var centroids = Enumerable.Range(0, k).Select(_ => new double[dim]).ToArray();
        var counts = new int[k];

        for (var i = 0; i < vectors.Count; i++)
        {
            var label = labels[i];
            counts[label] += 1;
            var vector = vectors[i];
            for (var d = 0; d < dim; d++)
            {
                centroids[label][d] += vector[d];
            }
        }

        for (var c = 0; c < k; c++)
        {
            if (counts[c] == 0)
            {
                continue;
            }

            for (var d = 0; d < dim; d++)
            {
                centroids[c][d] /= counts[c];
            }

            NormalizeVector(centroids[c]);
        }

        return centroids;
    }

    private static double TotalDistance(IReadOnlyList<double[]> vectors, int[] labels, double[][] centroids)
    {
        var total = 0.0;
        for (var i = 0; i < vectors.Count; i++)
        {
            total += CosineDistance(vectors[i], centroids[labels[i]]);
        }
        return total;
    }

    private static double CosineDistance(double[] a, double[] b)
    {
        var dot = 0.0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
        }
        return 1.0 - dot;
    }
}
