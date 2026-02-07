using System.Text;

namespace DlpKeywordPolicyGenerator.Web.Services.Analysis;

public sealed class PolicyService
{
    public string GeneratePrettyPolicy(
        SessionState session,
        GeneratePrettyOptions options)
    {
        var stats = session.Stats;
        var docsAll = session.DocsText;
        var topics = session.Topics.Take(Math.Min(session.Topics.Count, options.MaxRules + 8)).ToList();

        var cleanedRules = new List<GeneratedRule>();
        foreach (var topic in topics)
        {
            var candidates = topic.Candidates
                .Select(c => TextProcessing.NormalizeWhitespace(c.Phrase))
                .Where(p => !string.IsNullOrWhiteSpace(p))
                .Where(p => stats.ContainsKey(p))
                .Where(p => !IsLetterGarbage(p))
                .Where(p => TextProcessing.NormalizeForMatch(p).Length >= 3)
                .Where(p => !p.All(char.IsDigit))
                .Where(p => stats[p].DfRatio >= options.DfRatioMin && stats[p].DfRatio <= options.DfRatioMax)
                .Take(40)
                .ToList();

            var topicDocs = topic.DocIds.Where(i => i >= 0 && i < docsAll.Count).Select(i => docsAll[i]).ToList();
            var proposedGroups = new List<List<string>>();
            if (topicDocs.Count >= 2 && candidates.Count >= 8)
            {
                var (presence, normalized) = BuildPresenceMatrix(topicDocs, candidates);
                var (dfLocal, pairDf) = CoocStatsForPhrases(presence);
                proposedGroups = ProposeAndGroupsFromCooc(
                    normalized,
                    dfLocal,
                    pairDf,
                    topicDocs.Count,
                    session.GenericTerms,
                    maxGroups: 4,
                    groupSize: 6,
                    minDfDocs: 2,
                    minPairDocs: 2);
            }

            if (proposedGroups.Count == 0)
            {
                proposedGroups = new List<List<string>>
                {
                    candidates.Take(6).ToList(),
                    candidates.Skip(6).Take(6).ToList()
                };
            }

            var cleanGroups = EnforceSymantecShape(proposedGroups);
            if (cleanGroups.Count == 0)
            {
                continue;
            }

            var expr = GroupsToExpr(cleanGroups);
            cleanedRules.Add(new GeneratedRule
            {
                Title = topic.TopicName,
                Groups = cleanGroups,
                Expression = expr
            });
        }

        if (cleanedRules.Count == 0)
        {
            return "# DLP Keyword Policy (Proposed)\n\nNo rules generated.";
        }

        var selected = SelectCoverageAwareRules(cleanedRules, docsAll, stats, options.MaxRules);

        var sb = new StringBuilder();
        sb.AppendLine($"# {options.PolicyTitle}");
        sb.AppendLine();
        sb.AppendLine("Format: Symantec-style boolean rules (AND groups of OR phrases). No proximity logic.");
        sb.AppendLine();
        sb.AppendLine("## Rules");
        sb.AppendLine();

        var ruleIndex = 1;
        foreach (var rule in selected)
        {
            sb.AppendLine($"## Rule {ruleIndex}: {rule.Title}");
            sb.AppendLine($"`{rule.Expression}`");
            sb.AppendLine();
            sb.AppendLine("**Why chosen:**");
            sb.AppendLine("- Built from co-occurrence and whitelisted topic phrases to balance recall and precision.");
            sb.AppendLine();
            sb.AppendLine("---");
            sb.AppendLine();
            ruleIndex++;
        }

        return sb.ToString();
    }

    private static List<GeneratedRule> SelectCoverageAwareRules(
        List<GeneratedRule> rules,
        IReadOnlyList<string> docsAll,
        Dictionary<string, TermStats> stats,
        int maxRules)
    {
        var baseMax = Math.Clamp(maxRules, 1, 20);
        var hardCap = 20;
        var desiredCoverage = 0.85;
        var targetDocs = (int)Math.Ceiling(desiredCoverage * Math.Max(1, docsAll.Count));

        foreach (var rule in rules)
        {
            rule.CoveredDocs = DocsCoveredByExpr(rule.Expression, docsAll);
        }

        var selected = new List<GeneratedRule>();
        var covered = new HashSet<int>();
        var remaining = new List<GeneratedRule>(rules);

        while (remaining.Count > 0 && covered.Count < targetDocs && selected.Count < hardCap)
        {
            var bestIndex = -1;
            var bestScore = double.MinValue;
            for (var i = 0; i < remaining.Count; i++)
            {
                var rule = remaining[i];
                var gain = rule.CoveredDocs.Except(covered).Count();
                var penalty = RulePenalty(rule.Groups, stats);
                var score = gain - penalty;
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                break;
            }

            var pick = remaining[bestIndex];
            remaining.RemoveAt(bestIndex);
            var gainDocs = pick.CoveredDocs.Except(covered).Count();
            if (gainDocs <= 0)
            {
                break;
            }

            selected.Add(pick);
            foreach (var doc in pick.CoveredDocs)
            {
                covered.Add(doc);
            }

            if (selected.Count >= baseMax && covered.Count >= targetDocs)
            {
                break;
            }
        }

        if (selected.Count < baseMax)
        {
            foreach (var rule in rules)
            {
                if (selected.Contains(rule))
                {
                    continue;
                }
                selected.Add(rule);
                if (selected.Count >= baseMax || selected.Count >= hardCap)
                {
                    break;
                }
            }
        }

        return selected;
    }

    private static HashSet<int> DocsCoveredByExpr(string expr, IReadOnlyList<string> docs)
    {
        var covered = new HashSet<int>();
        for (var i = 0; i < docs.Count; i++)
        {
            var (ok, _) = RuleEngine.TestRule(expr, docs[i]);
            if (ok)
            {
                covered.Add(i);
            }
        }
        return covered;
    }

    private static double RulePenalty(List<List<string>> groups, Dictionary<string, TermStats> stats)
    {
        if (groups.Count < 2)
        {
            return 999.0;
        }

        var penalty = 0.0;
        foreach (var group in groups)
        {
            if (group.Count < 2)
            {
                penalty += 2.0;
            }
            else if (group.Count < 3)
            {
                penalty += 1.0;
            }
        }

        foreach (var group in groups)
        {
            foreach (var phrase in group)
            {
                if (!stats.TryGetValue(phrase, out var stat))
                {
                    continue;
                }

                if (stat.DfRatio >= 0.85)
                {
                    penalty += 2.0;
                }
                else if (stat.DfRatio >= 0.70)
                {
                    penalty += 1.0;
                }
            }
        }

        return penalty;
    }

    private static string GroupsToExpr(List<List<string>> groups)
    {
        var parts = groups
            .Select(group => OrGroup(group))
            .Where(expr => !string.IsNullOrWhiteSpace(expr))
            .ToList();

        if (parts.Count == 0)
        {
            return string.Empty;
        }

        if (parts.Count == 1)
        {
            return parts[0];
        }

        return "(" + string.Join(" AND ", parts) + ")";
    }

    private static string OrGroup(List<string> items)
    {
        var clean = items
            .Select(TextProcessing.NormalizeWhitespace)
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Select(Quote)
            .ToList();

        if (clean.Count == 0)
        {
            return string.Empty;
        }

        if (clean.Count == 1)
        {
            return clean[0];
        }

        return "(" + string.Join(" OR ", clean) + ")";
    }

    private static string Quote(string value)
    {
        var cleaned = TextProcessing.NormalizeWhitespace(value).Trim('"', '\'');
        return $"\"{cleaned}\"";
    }

    private static List<List<string>> EnforceSymantecShape(List<List<string>> groups)
    {
        var fixedGroups = new List<List<string>>();
        foreach (var group in groups.Where(g => g.Count > 0))
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            var deduped = new List<string>();
            foreach (var phrase in group)
            {
                var key = TextProcessing.NormalizeForMatch(phrase);
                if (seen.Add(key))
                {
                    deduped.Add(phrase);
                }
            }

            deduped = deduped.Take(8).ToList();
            if (deduped.Count >= 2)
            {
                fixedGroups.Add(deduped);
            }
        }

        fixedGroups = fixedGroups.Take(4).ToList();
        if (fixedGroups.Count < 2)
        {
            return new List<List<string>>();
        }

        return fixedGroups;
    }

    private static bool IsLetterGarbage(string phrase)
    {
        var tokens = TextProcessing.TokenizeForMatch(phrase);
        if (tokens.Count >= 4)
        {
            var one = tokens.Count(t => t.Length == 1);
            if ((double)one / tokens.Count >= 0.5)
            {
                return true;
            }
        }
        return false;
    }

    private static (List<List<bool>> presence, List<string> normalized) BuildPresenceMatrix(
        IReadOnlyList<string> docsText,
        IReadOnlyList<string> phrases)
    {
        var normalized = phrases.Select(TextProcessing.NormalizeWhitespace).Where(p => !string.IsNullOrWhiteSpace(p)).ToList();
        var presence = new List<List<bool>>();
        foreach (var doc in docsText)
        {
            var textNormalized = TextProcessing.NormalizeForMatch(doc);
            var row = normalized.Select(p => TextProcessing.PhrasePresent(p, textNormalized)).ToList();
            presence.Add(row);
        }

        return (presence, normalized);
    }

    private static (List<int> df, Dictionary<(int, int), int> pairDf) CoocStatsForPhrases(List<List<bool>> presence)
    {
        var df = new List<int>();
        if (presence.Count == 0)
        {
            return (df, new Dictionary<(int, int), int>());
        }

        var m = presence[0].Count;
        df = Enumerable.Repeat(0, m).ToList();
        var pairDf = new Dictionary<(int, int), int>();

        foreach (var row in presence)
        {
            var idxs = row.Select((value, index) => new { value, index })
                .Where(x => x.value)
                .Select(x => x.index)
                .ToList();

            foreach (var idx in idxs)
            {
                df[idx] += 1;
            }

            for (var a = 0; a < idxs.Count; a++)
            {
                for (var b = a + 1; b < idxs.Count; b++)
                {
                    var key = (idxs[a], idxs[b]);
                    pairDf[key] = pairDf.TryGetValue(key, out var count) ? count + 1 : 1;
                }
            }
        }

        return (df, pairDf);
    }

    private static double LiftScore(double pAb, double pA, double pB)
    {
        if (pA <= 0 || pB <= 0)
        {
            return 0.0;
        }
        return pAb / (pA * pB);
    }

    private static List<List<string>> ProposeAndGroupsFromCooc(
        IReadOnlyList<string> phrases,
        IReadOnlyList<int> df,
        IReadOnlyDictionary<(int, int), int> pairDf,
        int nDocs,
        IReadOnlyList<string> genericTerms,
        int maxGroups,
        int groupSize,
        int minDfDocs,
        int minPairDocs)
    {
        var genericSet = new HashSet<string>(genericTerms.Select(TextProcessing.NormalizeWhitespace).Where(s => !string.IsNullOrWhiteSpace(s)).Select(s => s.ToLowerInvariant()));
        var candidates = new List<(double midness, int df, int index)>();

        for (var i = 0; i < phrases.Count; i++)
        {
            var phrase = phrases[i].ToLowerInvariant();
            if (string.IsNullOrWhiteSpace(phrase) || genericSet.Contains(phrase))
            {
                continue;
            }
            if (df[i] < minDfDocs)
            {
                continue;
            }
            if (phrase.All(char.IsDigit))
            {
                continue;
            }

            var ratio = df[i] / (double)Math.Max(1, nDocs);
            var midness = ratio * (1.0 - ratio);
            candidates.Add((midness, df[i], i));
        }

        if (candidates.Count == 0)
        {
            return new List<List<string>> { phrases.Take(Math.Min(groupSize, phrases.Count)).ToList() };
        }

        candidates = candidates.OrderByDescending(c => c.midness).ToList();
        var anchorIdxs = candidates.Take(3).Select(c => c.index).ToList();

        var contextScores = new List<(double lift, int pairDocs, int df, int index)>();
        for (var j = 0; j < phrases.Count; j++)
        {
            if (anchorIdxs.Contains(j))
            {
                continue;
            }

            if (df[j] < minDfDocs)
            {
                continue;
            }

            var phrase = phrases[j].ToLowerInvariant();
            if (string.IsNullOrWhiteSpace(phrase) || genericSet.Contains(phrase))
            {
                continue;
            }

            var bestLift = 0.0;
            var bestPairDocs = 0;
            foreach (var anchor in anchorIdxs)
            {
                var key = anchor < j ? (anchor, j) : (j, anchor);
                if (!pairDf.TryGetValue(key, out var pairDocs) || pairDocs < minPairDocs)
                {
                    continue;
                }

                var pAb = pairDocs / (double)Math.Max(1, nDocs);
                var pA = df[anchor] / (double)Math.Max(1, nDocs);
                var pB = df[j] / (double)Math.Max(1, nDocs);
                var lift = LiftScore(pAb, pA, pB);
                if (lift > bestLift)
                {
                    bestLift = lift;
                    bestPairDocs = pairDocs;
                }
            }

            if (bestLift > 0)
            {
                contextScores.Add((bestLift, bestPairDocs, df[j], j));
            }
        }

        contextScores = contextScores.OrderByDescending(c => c.lift).ThenByDescending(c => c.pairDocs).ToList();

        var anchors = anchorIdxs.Select(i => phrases[i]).Take(groupSize).ToList();
        var context = contextScores.Select(c => phrases[c.index]).Take(groupSize).ToList();

        var groups = new List<List<string>>();
        if (anchors.Count > 0)
        {
            groups.Add(anchors);
        }
        if (context.Count > 0)
        {
            groups.Add(context);
        }

        var controls = new List<string>();
        foreach (var item in contextScores.Skip(groupSize).Take(20))
        {
            var ratio = df[item.index] / (double)Math.Max(1, nDocs);
            if (ratio >= 0.35 && ratio <= 0.85)
            {
                controls.Add(phrases[item.index]);
            }
            if (controls.Count >= groupSize)
            {
                break;
            }
        }

        if (controls.Count > 0 && groups.Count < maxGroups)
        {
            groups.Add(controls);
        }

        return groups.Take(maxGroups).ToList();
    }

    private sealed class GeneratedRule
    {
        public string Title { get; set; } = string.Empty;
        public List<List<string>> Groups { get; set; } = new();
        public string Expression { get; set; } = string.Empty;
        public HashSet<int> CoveredDocs { get; set; } = new();
    }
}

public sealed class GeneratePrettyOptions
{
    public string PolicyTitle { get; set; } = "DLP Keyword Policy (Proposed)";
    public int MaxRules { get; set; } = 5;
    public double DfRatioMin { get; set; } = 0.10;
    public double DfRatioMax { get; set; } = 0.85;
}
