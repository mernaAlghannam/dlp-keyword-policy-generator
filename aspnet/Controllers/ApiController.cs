using DlpKeywordPolicyGenerator.Web.Models;
using DlpKeywordPolicyGenerator.Web.Services;
using DlpKeywordPolicyGenerator.Web.Services.Analysis;
using Microsoft.AspNetCore.Mvc;

namespace DlpKeywordPolicyGenerator.Web.Controllers;

[ApiController]
public class ApiController : ControllerBase
{
    private readonly TextExtractionService _textExtraction;
    private readonly AnalysisService _analysisService;
    private readonly PolicyService _policyService;
    private readonly SessionStore _sessionStore;

    public ApiController(
        TextExtractionService textExtraction,
        AnalysisService analysisService,
        PolicyService policyService,
        SessionStore sessionStore)
    {
        _textExtraction = textExtraction;
        _analysisService = analysisService;
        _policyService = policyService;
        _sessionStore = sessionStore;
    }

    [HttpGet("/api/health")]
    public IActionResult Health()
    {
        return Ok(new { ok = true });
    }

    [HttpPost("/api/analyze")]
    [DisableRequestSizeLimit]
    public async Task<ActionResult<AnalyzeResponse>> Analyze([FromForm] List<IFormFile> files)
    {
        if (files.Count == 0)
        {
            return BadRequest("No files uploaded.");
        }

        var docsText = new List<string>();
        foreach (var file in files)
        {
            await using var stream = file.OpenReadStream();
            if (stream.Length == 0)
            {
                continue;
            }

            var text = _textExtraction.ExtractText(file);
            text = TextProcessing.NormalizeWhitespace(text);
            if (!string.IsNullOrWhiteSpace(text))
            {
                docsText.Add(text);
            }
        }

        if (docsText.Count == 0)
        {
            return BadRequest("No readable text extracted from uploaded files.");
        }

        var docsTerms = docsText.Select(TextProcessing.BuildDocTerms).ToList();
        var stats = _analysisService.ComputeTfDf(docsTerms);
        var topics = _analysisService.InferTopics(docsText, stats, dfRatioMin: 0.10, dfRatioMax: 0.85);
        var response = _analysisService.AnalyzeDocuments(docsText, stats, topics);
        var session = new SessionState
        {
            DocsText = docsText,
            Stats = stats,
            Topics = topics,
            GenericTerms = new List<string>()
        };

        _sessionStore.Create(session);

        response.SessionId = session.SessionId;
        response.NDocs = docsText.Count;

        return Ok(response);
    }

    [HttpPost("/api/generate_pretty")]
    public IActionResult GeneratePretty([FromBody] GeneratePrettyRequest request)
    {
        var session = _sessionStore.Get(request.SessionId);
        if (session == null)
        {
            return NotFound("Unknown session_id. Run /api/analyze first.");
        }

        var options = new GeneratePrettyOptions
        {
            PolicyTitle = request.PolicyTitle,
            MaxRules = request.MaxRules,
            DfRatioMin = request.DfRatioMin,
            DfRatioMax = request.DfRatioMax
        };

        var output = _policyService.GeneratePrettyPolicy(session, options);
        return Content(output, "text/plain");
    }

    [HttpPost("/api/test_policy")]
    [DisableRequestSizeLimit]
    public IActionResult TestPolicy([FromForm] string policyText, [FromForm] List<IFormFile> files)
    {
        var rules = RuleEngine.ParsePrettyPolicyRules(policyText);
        if (rules.Count == 0)
        {
            return BadRequest("No rules found. Expected either: **Rule Name**: `( ... )` OR '## Rule X: Name' followed by `( ... )`.");
        }

        var results = new List<TestPolicyFileResult>();
        foreach (var file in files)
        {
            var text = _textExtraction.ExtractText(file);
            text = TextProcessing.NormalizeWhitespace(text);
            if (string.IsNullOrWhiteSpace(text))
            {
                results.Add(new TestPolicyFileResult
                {
                    Filename = file.FileName,
                    Readable = false,
                    MatchedAny = false
                });
                continue;
            }

            var matchedRules = new List<TestPolicyRuleMatch>();
            foreach (var rule in rules)
            {
                var (ok, hits) = RuleEngine.TestRule(rule.Expression, text);
                if (!ok)
                {
                    continue;
                }

                var topHits = hits.OrderByDescending(kv => kv.Value).Take(12).Select(kv => kv.Key).ToList();
                matchedRules.Add(new TestPolicyRuleMatch
                {
                    RuleName = rule.Name,
                    HitPhrases = topHits
                });
            }

            results.Add(new TestPolicyFileResult
            {
                Filename = file.FileName,
                Readable = true,
                MatchedAny = matchedRules.Count > 0,
                MatchedRules = matchedRules
            });
        }

        return Ok(new TestPolicyResponse
        {
            NFiles = results.Count,
            Results = results
        });
    }
}
