using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace DlpKeywordPolicyGenerator.Web.Services.Analysis;

public static class TextProcessing
{
    private static readonly Regex TokenRegex = new("[A-Za-z0-9\u0600-\u06FF]+", RegexOptions.Compiled);
    private static readonly Regex TokenMatchRegex = new("[A-Za-z0-9\u0600-\u06FF]+", RegexOptions.Compiled);
    private static readonly Regex ArabicDiacritics = new("[\u064B-\u065F\u0670\u06D6-\u06ED]", RegexOptions.Compiled);
    private const string ArabicTatweel = "\u0640";

    public static string NormalizeWhitespace(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        return Regex.Replace(value, "\\s+", " ").Trim();
    }

    public static List<string> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return new List<string>();
        }

        var lower = text.ToLowerInvariant();
        return TokenRegex.Matches(lower).Select(m => m.Value).ToList();
    }

    public static IEnumerable<string> Ngrams(IReadOnlyList<string> tokens, int n)
    {
        if (n <= 1)
        {
            foreach (var token in tokens)
            {
                yield return token;
            }
            yield break;
        }

        for (var i = 0; i <= tokens.Count - n; i++)
        {
            yield return string.Join(" ", tokens.Skip(i).Take(n));
        }
    }

    public static List<string> BuildDocTerms(string text, int maxTokens = 120_000)
    {
        var tokens = Tokenize(text);
        if (tokens.Count > maxTokens)
        {
            tokens = tokens.Take(maxTokens).ToList();
        }

        var terms = new List<string>();
        terms.AddRange(Ngrams(tokens, 1));
        terms.AddRange(Ngrams(tokens, 2));
        terms.AddRange(Ngrams(tokens, 3));
        return terms;
    }

    public static string NormalizeForMatch(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var normalized = text.ToLowerInvariant()
            .Replace("\u201c", "\"")
            .Replace("\u201d", "\"")
            .Replace("\u2019", "'")
            .Replace(ArabicTatweel, string.Empty);

        normalized = ArabicDiacritics.Replace(normalized, string.Empty);
        normalized = normalized.Replace("أ", "ا").Replace("إ", "ا").Replace("آ", "ا");
        normalized = normalized.Normalize(NormalizationForm.FormKC);

        return NormalizeWhitespace(normalized);
    }

    public static List<string> TokenizeForMatch(string text)
    {
        var normalized = NormalizeForMatch(text);
        return TokenMatchRegex.Matches(normalized).Select(m => m.Value).ToList();
    }

    public static bool PhraseInTokens(string phrase, IReadOnlyList<string> tokens, HashSet<string> tokenSet)
    {
        var normalized = NormalizeForMatch(phrase);
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return false;
        }

        var phraseTokens = TokenizeForMatch(normalized);
        if (phraseTokens.Count == 0)
        {
            return false;
        }

        if (phraseTokens.Count == 1)
        {
            return tokenSet.Contains(phraseTokens[0]);
        }

        var n = phraseTokens.Count;
        for (var i = 0; i <= tokens.Count - n; i++)
        {
            var matches = true;
            for (var j = 0; j < n; j++)
            {
                if (!string.Equals(tokens[i + j], phraseTokens[j], StringComparison.Ordinal))
                {
                    matches = false;
                    break;
                }
            }

            if (matches)
            {
                return true;
            }
        }

        return false;
    }

    public static bool PhrasePresent(string phrase, string normalizedText)
    {
        var normalized = NormalizeForMatch(phrase);
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return false;
        }

        var escaped = Regex.Escape(normalized);
        var pattern = new Regex($"(?<![A-Za-z0-9\u0600-\u06FF]){escaped}(?![A-Za-z0-9\u0600-\u06FF])", RegexOptions.Compiled | RegexOptions.CultureInvariant);
        return pattern.IsMatch(normalizedText);
    }
}
