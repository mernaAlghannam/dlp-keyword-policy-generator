using System.Text.RegularExpressions;

namespace DlpKeywordPolicyGenerator.Web.Services.Analysis;

public static class RuleEngine
{
    private static readonly Regex TokenRegex = new("\"([^\"]+)\"|\\bAND\\b|\\bOR\\b|\\(|\\)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex RuleHeadingRegex = new("^##\\s*Rule\\s*\\d+\\s*:\\s*(.+?)\\s*$", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex ExpressionCodeRegex = new("`([^`]+)`", RegexOptions.Compiled);
    private static readonly Regex ExpressionParenRegex = new("\\((.+)\\)", RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex NamedRuleRegex = new("\\*\\*(.+?)\\*\\*\\s*:\\s*`([^`]+)`", RegexOptions.Compiled);

    public static List<ParsedRule> ParsePrettyPolicyRules(string policyText)
    {
        var text = StripCodeFences(policyText);
        var rules = new List<ParsedRule>();

        var blocks = Regex.Split(text, "\\n(?=##\\s*Rule\\s*\\d+\\s*:)\");
        foreach (var block in blocks)
        {
            var headingMatch = RuleHeadingRegex.Match(block.Trim());
            if (!headingMatch.Success)
            {
                continue;
            }

            var name = headingMatch.Groups[1].Value.Trim();
            var exprMatch = ExpressionCodeRegex.Match(block);
            if (!exprMatch.Success)
            {
                exprMatch = ExpressionParenRegex.Match(block);
            }

            if (exprMatch.Success)
            {
                var expr = exprMatch.Groups[1].Value.Trim();
                rules.Add(new ParsedRule { Name = name, Expression = expr });
            }
        }

        if (rules.Count == 0)
        {
            foreach (Match match in NamedRuleRegex.Matches(text))
            {
                rules.Add(new ParsedRule
                {
                    Name = match.Groups[1].Value.Trim(),
                    Expression = match.Groups[2].Value.Trim()
                });
            }
        }

        return rules;
    }

    public static (bool ok, Dictionary<string, int> hits) TestRule(string expr, string text)
    {
        var tokens = TokenizeExpr(expr);
        var rpn = ToRpn(tokens);
        var hits = new Dictionary<string, int>(StringComparer.Ordinal);
        var ok = EvalRpn(rpn, text, hits);
        return (ok, hits);
    }

    private static string StripCodeFences(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return string.Empty;
        }

        var trimmed = input.Trim();
        trimmed = Regex.Replace(trimmed, "^```[a-zA-Z0-9_-]*\\s*", string.Empty);
        trimmed = Regex.Replace(trimmed, "\\s*```$", string.Empty);
        return trimmed.Trim();
    }

    private static List<Token> TokenizeExpr(string expr)
    {
        var tokens = new List<Token>();
        foreach (Match match in TokenRegex.Matches(expr))
        {
            if (match.Groups[1].Success)
            {
                tokens.Add(new Token(TokenType.Phrase, match.Groups[1].Value));
                continue;
            }

            var value = match.Value.ToUpperInvariant();
            tokens.Add(value switch
            {
                "AND" => new Token(TokenType.And, value),
                "OR" => new Token(TokenType.Or, value),
                "(" => new Token(TokenType.LeftParen, value),
                ")" => new Token(TokenType.RightParen, value),
                _ => new Token(TokenType.Unknown, value)
            });
        }

        return tokens;
    }

    private static List<Token> ToRpn(List<Token> tokens)
    {
        var output = new List<Token>();
        var stack = new Stack<Token>();
        var precedence = new Dictionary<TokenType, int>
        {
            [TokenType.And] = 2,
            [TokenType.Or] = 1
        };

        foreach (var token in tokens)
        {
            if (token.Type == TokenType.Phrase)
            {
                output.Add(token);
            }
            else if (token.Type == TokenType.And || token.Type == TokenType.Or)
            {
                while (stack.Count > 0 && (stack.Peek().Type == TokenType.And || stack.Peek().Type == TokenType.Or))
                {
                    if (precedence[stack.Peek().Type] >= precedence[token.Type])
                    {
                        output.Add(stack.Pop());
                    }
                    else
                    {
                        break;
                    }
                }
                stack.Push(token);
            }
            else if (token.Type == TokenType.LeftParen)
            {
                stack.Push(token);
            }
            else if (token.Type == TokenType.RightParen)
            {
                while (stack.Count > 0 && stack.Peek().Type != TokenType.LeftParen)
                {
                    output.Add(stack.Pop());
                }
                if (stack.Count > 0 && stack.Peek().Type == TokenType.LeftParen)
                {
                    stack.Pop();
                }
            }
        }

        while (stack.Count > 0)
        {
            var item = stack.Pop();
            if (item.Type != TokenType.LeftParen)
            {
                output.Add(item);
            }
        }

        return output;
    }

    private static bool EvalRpn(List<Token> rpn, string rawText, Dictionary<string, int> hitCounts)
    {
        var tokens = TextProcessing.TokenizeForMatch(rawText);
        var tokenSet = new HashSet<string>(tokens, StringComparer.Ordinal);
        var stack = new Stack<(bool ok, HashSet<string> hits)>();

        foreach (var token in rpn)
        {
            if (token.Type == TokenType.Phrase)
            {
                var needle = token.Value.Trim();
                var ok = TextProcessing.PhraseInTokens(needle, tokens, tokenSet);
                var hits = new HashSet<string>(StringComparer.Ordinal);
                if (ok)
                {
                    hits.Add(TextProcessing.NormalizeForMatch(needle));
                }
                stack.Push((ok, hits));
                continue;
            }

            if (token.Type == TokenType.And || token.Type == TokenType.Or)
            {
                var right = stack.Count > 0 ? stack.Pop() : (false, new HashSet<string>());
                var left = stack.Count > 0 ? stack.Pop() : (false, new HashSet<string>());
                if (token.Type == TokenType.And)
                {
                    var ok = left.ok && right.ok;
                    stack.Push((ok, ok ? new HashSet<string>(left.hits.Union(right.hits)) : new HashSet<string>()));
                }
                else
                {
                    var ok = left.ok || right.ok;
                    if (left.ok && right.ok)
                    {
                        stack.Push((true, new HashSet<string>(left.hits.Union(right.hits))));
                    }
                    else if (left.ok)
                    {
                        stack.Push((true, new HashSet<string>(left.hits)));
                    }
                    else if (right.ok)
                    {
                        stack.Push((true, new HashSet<string>(right.hits)));
                    }
                    else
                    {
                        stack.Push((false, new HashSet<string>()));
                    }
                }
            }
        }

        if (stack.Count == 0)
        {
            return false;
        }

        var result = stack.Pop();
        if (result.ok)
        {
            foreach (var key in result.hits)
            {
                hitCounts[key] = hitCounts.TryGetValue(key, out var count) ? count + 1 : 1;
            }
        }

        return result.ok;
    }

    public sealed class ParsedRule
    {
        public string Name { get; set; } = string.Empty;
        public string Expression { get; set; } = string.Empty;
    }

    private enum TokenType
    {
        Phrase,
        And,
        Or,
        LeftParen,
        RightParen,
        Unknown
    }

    private sealed record Token(TokenType Type, string Value);
}
