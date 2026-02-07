using System.Text;
using DlpKeywordPolicyGenerator.Web.Models;

namespace DlpKeywordPolicyGenerator.Web.Services.Analysis;

public sealed class AnalysisService
{
    private readonly TopicInferenceService _topicInferenceService;

    public AnalysisService(TopicInferenceService topicInferenceService)
    {
        _topicInferenceService = topicInferenceService;
    }

    public AnalyzeResponse AnalyzeDocuments(
        IReadOnlyList<string> docsText,
        Dictionary<string, TermStats> stats,
        IReadOnlyList<TopicCluster> topics)
    {

        var topTerms = stats
            .OrderByDescending(kv => kv.Value.Df)
            .ThenByDescending(kv => kv.Value.Tf)
            .Take(40)
            .Select(kv => new TermRow
            {
                Term = kv.Key,
                Tf = kv.Value.Tf,
                Df = kv.Value.Df,
                DfRatio = kv.Value.DfRatio
            })
            .ToList();

        var topicsPreview = topics
            .Select(topic => new TopicPreview
            {
                Name = topic.TopicName,
                DocCount = topic.DocIds.Count,
                Phrases = topic.Candidates
                    .Take(12)
                    .Select(c => new PhraseRow { Text = c.Phrase, DfRatio = c.DfRatio })
                    .ToList()
            })
            .ToList();

        return new AnalyzeResponse
        {
            NDocs = docsText.Count,
            TopicsPreview = topicsPreview,
            TopTerms = topTerms,
            TopicsPreviewMd = BuildTopicsMarkdown(topicsPreview),
            TopTermsMd = BuildTopTermsMarkdown(topTerms)
        };
    }

    public List<TopicCluster> InferTopics(
        IReadOnlyList<string> docsText,
        Dictionary<string, TermStats> stats,
        double dfRatioMin,
        double dfRatioMax)
    {
        return _topicInferenceService.InferTopics(docsText, stats, dfRatioMin, dfRatioMax);
    }

    public Dictionary<string, TermStats> ComputeTfDf(IReadOnlyList<List<string>> docsTerms)
    {
        var tfMap = new Dictionary<string, int>(StringComparer.Ordinal);
        var dfMap = new Dictionary<string, int>(StringComparer.Ordinal);
        var nDocs = docsTerms.Count;

        foreach (var terms in docsTerms)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var term in terms)
            {
                tfMap[term] = tfMap.TryGetValue(term, out var tf) ? tf + 1 : 1;
                if (seen.Add(term))
                {
                    dfMap[term] = dfMap.TryGetValue(term, out var df) ? df + 1 : 1;
                }
            }
        }

        var stats = new Dictionary<string, TermStats>(StringComparer.Ordinal);
        foreach (var (term, tf) in tfMap)
        {
            var df = dfMap.TryGetValue(term, out var dfValue) ? dfValue : 0;
            stats[term] = new TermStats
            {
                Tf = tf,
                Df = df,
                DfRatio = nDocs == 0 ? 0 : (double)df / nDocs
            };
        }

        return stats;
    }

    private static string BuildTopTermsMarkdown(IEnumerable<TermRow> terms)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Term | TF | DF | DF%");
        sb.AppendLine("---|---:|---:|---:");
        foreach (var term in terms)
        {
            var dfPct = (int)Math.Round(term.DfRatio * 100);
            sb.AppendLine($"{term.Term} | {term.Tf} | {term.Df} | {dfPct}%");
        }
        return sb.ToString();
    }

    private static string BuildTopicsMarkdown(IEnumerable<TopicPreview> topics)
    {
        var sb = new StringBuilder();
        foreach (var topic in topics)
        {
            sb.AppendLine($"### {topic.Name} (docs={topic.DocCount})");
            foreach (var phrase in topic.Phrases)
            {
                var dfPct = (int)Math.Round(phrase.DfRatio * 100);
                sb.AppendLine($"- {phrase.Text} ({dfPct}%)");
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }
}
