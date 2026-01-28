// frontend/src/components/Upload.jsx
import React, { useMemo, useState } from "react";
import { analyzeDocs } from "../api";

export default function Upload({ onSession }) {
    const [files, setFiles] = useState([]);
    const [busy, setBusy] = useState(false);
    const [msg, setMsg] = useState("");
    const [sessionId, setSessionId] = useState("");
    const [analysis, setAnalysis] = useState(null);

    const canAnalyze = useMemo(() => files.length > 0 && !busy, [files, busy]);

    function onPick(e) {
        const list = Array.from(e.target.files || []);
        setFiles(list);
        setMsg("");
    }

    async function onAnalyze() {
        if (!files.length) {
            setMsg("Choose files first.");
            return;
        }
        setBusy(true);
        setMsg("");
        setAnalysis(null);

        try {
            const data = await analyzeDocs(files);

            const sid = data?.session_id || "";
            setSessionId(sid);
            setAnalysis(data);

            if (onSession && sid) onSession(sid);

            setMsg(`Analyze complete. session_id=${sid || "—"}`);
        } catch (e) {
            const detail = e?.response?.data?.detail || e?.message || "Analyze failed.";
            setMsg(String(detail));
        } finally {
            setBusy(false);
        }
    }

    // robust: accept either naming
    const topics =
        analysis?.topics ||
        analysis?.topics_preview ||
        analysis?.topic_preview ||
        [];

    const topTerms =
        analysis?.top_terms ||
        analysis?.top_terms_table ||
        analysis?.terms ||
        [];

    return (
        <div style={styles.card}>
            <div style={styles.headerRow}>
                <div>
                    <div style={styles.hTitle}>1) Upload confidential docs</div>
                    <div style={styles.hSub}>
                        Upload representative documents (txt / pdf / docx) then run analysis to see topics + TF/DF.
                    </div>
                </div>

                <div style={styles.badge}>
                    Backend <span style={styles.kbd}>:8000</span>
                </div>
            </div>

            <div style={{ marginTop: 10 }}>
                <input
                    type="file"
                    multiple
                    onChange={onPick}
                    style={styles.file}
                    accept=".txt,.pdf,.docx"
                />
                <div style={styles.small}>
                    {files.length} file(s) selected
                </div>
            </div>

            <div style={styles.rowBetween}>
                <button
                    style={{ ...styles.btn, ...(canAnalyze ? styles.btnPrimary : styles.btnDisabled) }}
                    onClick={onAnalyze}
                    disabled={!canAnalyze}
                >
                    {busy ? "Analyzing…" : "Analyze"}
                </button>

                <div style={styles.badge}>
                    Session{" "}
                    <span style={styles.kbd}>
                        {sessionId ? sessionId.slice(0, 8) + "…" : "—"}
                    </span>
                </div>
            </div>

            {msg && (
                <div
                    style={{
                        ...styles.toast,
                        ...(msg.toLowerCase().includes("fail") || msg.toLowerCase().includes("error")
                            ? styles.toastErr
                            : styles.toastOk),
                    }}
                >
                    {msg}
                </div>
            )}

            <div style={{ marginTop: 16 }}>
                <div style={styles.sectionTitle}>Topics (preview)</div>
                <div style={styles.sectionHint}>
                    High-signal phrases per topic (DF% hints generalization).
                </div>

                {!analysis ? (
                    <div style={styles.muted}>Run Analyze to see topics.</div>
                ) : topics.length === 0 ? (
                    <div style={styles.muted}>
                        No topics returned from backend. (Either extraction failed or backend returned different keys.)
                    </div>
                ) : (
                    <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
                        {topics.map((t, i) => {
                            const name = t?.name || t?.topic || `Topic ${i + 1}`;
                            const docCount = t?.doc_count ?? t?.docs ?? null;
                            const phrases = t?.phrases || t?.terms || [];
                            return (
                                <div key={i} style={styles.topicCard}>
                                    <div style={styles.topicHeader}>
                                        <div style={styles.topicName}>{name}</div>
                                        {docCount !== null && (
                                            <div style={styles.pill}>{docCount} docs</div>
                                        )}
                                    </div>

                                    {phrases.length === 0 ? (
                                        <div style={styles.muted}>No phrases.</div>
                                    ) : (
                                        <div style={styles.phraseGrid}>
                                            {phrases.slice(0, 18).map((p, j) => {
                                                const text = p?.text || p?.term || String(p);
                                                const dfp =
                                                    p?.df_ratio != null
                                                        ? Math.round(p.df_ratio * 100)
                                                        : p?.df_percent != null
                                                            ? Math.round(p.df_percent)
                                                            : null;
                                                return (
                                                    <div key={j} style={styles.phraseChip}>
                                                        <span>{text}</span>
                                                        {dfp !== null && (
                                                            <span style={styles.dfp}>{dfp}%</span>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            <div style={{ marginTop: 18 }}>
                <div style={styles.sectionTitle}>Top terms (TF / DF / DF%)</div>

                {!analysis ? (
                    <div style={styles.muted}>Run Analyze to see TF/DF.</div>
                ) : topTerms.length === 0 ? (
                    <div style={styles.muted}>
                        No top terms returned from backend.
                    </div>
                ) : (
                    <div style={styles.tableWrap}>
                        <table style={styles.table}>
                            <thead>
                                <tr>
                                    <th style={styles.th}>Term</th>
                                    <th style={styles.th}>TF</th>
                                    <th style={styles.th}>DF</th>
                                    <th style={styles.th}>DF%</th>
                                </tr>
                            </thead>
                            <tbody>
                                {topTerms.slice(0, 80).map((r, idx) => {
                                    const term = r?.term || r?.text || "";
                                    const tf = r?.tf ?? r?.TF ?? "";
                                    const df = r?.df ?? r?.DF ?? "";
                                    const dfp =
                                        r?.df_ratio != null
                                            ? Math.round(r.df_ratio * 100)
                                            : r?.df_percent != null
                                                ? Math.round(r.df_percent)
                                                : "";
                                    return (
                                        <tr key={idx}>
                                            <td style={styles.tdTerm}>{term}</td>
                                            <td style={styles.td}>{tf}</td>
                                            <td style={styles.td}>{df}</td>
                                            <td style={styles.td}>{dfp !== "" ? `${dfp}%` : ""}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}

const styles = {
    card: {
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: 16,
        padding: 16,
        boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
        backdropFilter: "blur(10px)",
    },
    headerRow: { display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start" },
    hTitle: { fontSize: 16, fontWeight: 800 },
    hSub: { marginTop: 4, fontSize: 12, opacity: 0.8, lineHeight: 1.35 },
    badge: {
        padding: "8px 10px",
        borderRadius: 999,
        border: "1px solid rgba(255,255,255,0.15)",
        background: "rgba(255,255,255,0.06)",
        fontSize: 12,
        whiteSpace: "nowrap",
    },
    kbd: {
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        padding: "2px 6px",
        borderRadius: 8,
        border: "1px solid rgba(255,255,255,0.16)",
        background: "rgba(0,0,0,0.25)",
        marginLeft: 6,
    },
    file: { width: "100%" },
    small: { marginTop: 6, fontSize: 12, opacity: 0.75 },
    rowBetween: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, marginTop: 12, flexWrap: "wrap" },
    btn: { borderRadius: 12, padding: "10px 12px", fontWeight: 800, border: "1px solid rgba(255,255,255,0.14)", cursor: "pointer" },
    btnPrimary: { background: "rgba(255,255,255,0.16)", color: "white" },
    btnDisabled: { background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.45)", cursor: "not-allowed" },
    toast: { marginTop: 12, padding: "10px 12px", borderRadius: 12, border: "1px solid rgba(255,255,255,0.12)", fontSize: 12 },
    toastOk: { background: "rgba(46, 204, 113, 0.12)" },
    toastErr: { background: "rgba(231, 76, 60, 0.14)" },
    sectionTitle: { fontSize: 14, fontWeight: 800 },
    sectionHint: { fontSize: 12, opacity: 0.75, marginTop: 4 },
    muted: { fontSize: 12, opacity: 0.7, marginTop: 10 },
    topicCard: { borderRadius: 14, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(0,0,0,0.22)", padding: 12 },
    topicHeader: { display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" },
    topicName: { fontWeight: 800 },
    pill: { fontSize: 12, opacity: 0.85, border: "1px solid rgba(255,255,255,0.12)", borderRadius: 999, padding: "4px 8px", background: "rgba(255,255,255,0.06)" },
    phraseGrid: { marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" },
    phraseChip: { display: "inline-flex", gap: 8, alignItems: "center", padding: "6px 10px", borderRadius: 999, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.10)", fontSize: 12 },
    dfp: { fontSize: 11, opacity: 0.75, borderLeft: "1px solid rgba(255,255,255,0.12)", paddingLeft: 8 },
    tableWrap: { marginTop: 10, borderRadius: 14, border: "1px solid rgba(255,255,255,0.12)", overflow: "hidden" },
    table: { width: "100%", borderCollapse: "collapse" },
    th: { textAlign: "left", fontSize: 12, padding: "10px 12px", background: "rgba(255,255,255,0.06)", borderBottom: "1px solid rgba(255,255,255,0.10)" },
    td: { fontSize: 12, padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.06)" },
    tdTerm: { fontSize: 12, padding: "8px 12px", borderBottom: "1px solid rgba(255,255,255,0.06)", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" },
};
