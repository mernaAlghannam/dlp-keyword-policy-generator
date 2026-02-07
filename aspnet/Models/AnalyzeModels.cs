namespace DlpKeywordPolicyGenerator.Web.Models;

public sealed class PhraseRow
{
    public string Text { get; set; } = string.Empty;
    public double DfRatio { get; set; }
}

public sealed class TopicPreview
{
    public string Name { get; set; } = string.Empty;
    public int DocCount { get; set; }
    public List<PhraseRow> Phrases { get; set; } = new();
}

public sealed class TermRow
{
    public string Term { get; set; } = string.Empty;
    public int Tf { get; set; }
    public int Df { get; set; }
    public double DfRatio { get; set; }
}

public sealed class AnalyzeResponse
{
    public string SessionId { get; set; } = string.Empty;
    public int NDocs { get; set; }
    public string TopicsPreviewMd { get; set; } = string.Empty;
    public string TopTermsMd { get; set; } = string.Empty;
    public List<TopicPreview> TopicsPreview { get; set; } = new();
    public List<TermRow> TopTerms { get; set; } = new();
    public List<string> GenericTerms { get; set; } = new();
}

public sealed class GeneratePrettyRequest
{
    public string SessionId { get; set; } = string.Empty;
    public string PolicyTitle { get; set; } = "DLP Keyword Policy (Proposed)";
    public string CorpusHint { get; set; } = string.Empty;
    public int MaxRules { get; set; } = 5;
    public double DfRatioMin { get; set; } = 0.10;
    public double DfRatioMax { get; set; } = 0.85;
    public bool IncludeRegexSuggestions { get; set; } = true;
    public int MaxRegexSuggestions { get; set; } = 8;
}

public sealed class TestPolicyResponse
{
    public int NFiles { get; set; }
    public List<TestPolicyFileResult> Results { get; set; } = new();
}

public sealed class TestPolicyFileResult
{
    public string Filename { get; set; } = string.Empty;
    public bool Readable { get; set; }
    public bool MatchedAny { get; set; }
    public List<TestPolicyRuleMatch> MatchedRules { get; set; } = new();
}

public sealed class TestPolicyRuleMatch
{
    public string RuleName { get; set; } = string.Empty;
    public List<string> HitPhrases { get; set; } = new();
}
