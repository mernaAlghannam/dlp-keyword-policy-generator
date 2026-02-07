using System.Collections.Concurrent;

namespace DlpKeywordPolicyGenerator.Web.Services.Analysis;

public sealed class SessionStore
{
    private readonly ConcurrentDictionary<string, SessionState> _sessions = new();

    public SessionState Create(SessionState state)
    {
        _sessions[state.SessionId] = state;
        return state;
    }

    public SessionState? Get(string sessionId)
    {
        _sessions.TryGetValue(sessionId, out var state);
        return state;
    }
}

public sealed class SessionState
{
    public string SessionId { get; init; } = Guid.NewGuid().ToString("N");
    public List<string> DocsText { get; init; } = new();
    public Dictionary<string, TermStats> Stats { get; init; } = new();
    public List<TopicCluster> Topics { get; init; } = new();
    public List<string> GenericTerms { get; init; } = new();
}

public sealed class TopicCluster
{
    public string TopicName { get; set; } = string.Empty;
    public List<int> DocIds { get; set; } = new();
    public List<TopicCandidate> Candidates { get; set; } = new();
}

public sealed class TopicCandidate
{
    public string Phrase { get; set; } = string.Empty;
    public double DfRatio { get; set; }
}
