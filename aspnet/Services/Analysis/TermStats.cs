namespace DlpKeywordPolicyGenerator.Web.Services.Analysis;

public sealed class TermStats
{
    public int Tf { get; set; }
    public int Df { get; set; }
    public double DfRatio { get; set; }
}
