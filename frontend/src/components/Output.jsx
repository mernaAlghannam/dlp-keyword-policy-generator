// frontend/src/components/Output.jsx
import React, { useMemo, useState } from "react";
import { generatePretty } from "../api";

export default function Output({ sessionId, onPolicy }) {
    const [policyTitle, setPolicyTitle] = useState("DLP Keyword Policy (Proposed)");
    const [maxRules, setMaxRules] = useState(6);
    const [hint, setHint] = useState("");
    const [dfMin, setDfMin] = useState(0.1);
    const [dfMax, setDfMax] = useState(0.85);
    const [llmRequired, setLlmRequired] = useState(true);
    const [includeRegex, setIncludeRegex] = useState(true);
    const [maxRegex, setMaxRegex] = useState(8);

    const [busy, setBusy] = useState(false);
    const [out, setOut] = useState("");
    const [msg, setMsg] = useState("");

    const canGenerate = useMemo(() => !!sessionId && !busy, [sessionId, busy]);

    async function onGenerate() {
        if (!sessionId) {
            setMsg("Analyze first to get a session_id.");
            return;
        }
        setBusy(true);
        setMsg("");
        setOut("");

        try {
            const text = await generatePretty({
                session_id: sessionId,
                policy_title: policyTitle,
                max_rules: Number(maxRules) || 6,
                corpus_hint: hint || "",
                df_ratio_min: Number(dfMin) || 0.1,
                df_ratio_max: Number(dfMax) || 0.85,
                llm_required: !!llmRequired,
                include_regex_suggestions: !!includeRegex,
                max_regex_suggestions: Number(maxRegex) || 8,
            });

            // ✅ FIX: if backend returned a JSON-stringified string, unescape it
            let cleaned = text;
            try {
                cleaned = JSON.parse(text);
            } catch {
                // not JSON, keep as-is
            }

            setOut(cleaned);
            if (typeof onPolicy === "function") onPolicy(cleaned);
            setMsg("Generated.");
        } catch (e) {
            const detail = e?.response?.data?.detail || e?.message || "Generate failed.";
            setMsg(String(detail));
        } finally {
            setBusy(false);
        }
    }

    async function onCopy() {
        try {
            await navigator.clipboard.writeText(out || "");
            setMsg("Copied to clipboard.");
        } catch {
            setMsg("Copy failed.");
        }
    }

    return (
        <div style={styles.card}>
            <div style={styles.headerRow}>
                <div>
                    <div style={styles.hTitle}>2) Generate policy</div>
                    <div style={styles.hSub}>
                        Produces readable rules (AND-gated groups). Endpoint:{" "}
                        <span style={styles.kbd}>/api/generate_pretty</span>
                    </div>
                </div>

                <div style={styles.badge}>
                    Session{" "}
                    <span style={styles.kbd}>{sessionId ? sessionId.slice(0, 8) + "…" : "—"}</span>
                </div>
            </div>

            <div style={styles.grid2}>
                <div>
                    <div style={styles.label}>Policy title</div>
                    <input
                        style={styles.input}
                        value={policyTitle}
                        onChange={(e) => setPolicyTitle(e.target.value)}
                    />
                </div>
                <div>
                    <div style={styles.label}>Max rules</div>
                    <input
                        style={styles.input}
                        type="number"
                        min={1}
                        max={20}
                        value={maxRules}
                        onChange={(e) => setMaxRules(e.target.value)}
                    />
                </div>
            </div>

            <div style={{ marginTop: 12 }}>
                <div style={styles.label}>Department hint (recommended)</div>
                <input
                    style={styles.input}
                    value={hint}
                    onChange={(e) => setHint(e.target.value)}
                    placeholder='e.g., "Work permits + PPE inspection + HSE monitoring"'
                />
            </div>

            <div style={styles.grid2}>
                <div>
                    <div style={styles.label}>DF% min</div>
                    <input
                        style={styles.input}
                        type="number"
                        step="0.01"
                        value={dfMin}
                        onChange={(e) => setDfMin(e.target.value)}
                    />
                </div>
                <div>
                    <div style={styles.label}>DF% max</div>
                    <input
                        style={styles.input}
                        type="number"
                        step="0.01"
                        value={dfMax}
                        onChange={(e) => setDfMax(e.target.value)}
                    />
                </div>
            </div>

            <div style={styles.rowBetween}>
                <label style={styles.check}>
                    <input
                        type="checkbox"
                        checked={llmRequired}
                        onChange={(e) => setLlmRequired(e.target.checked)}
                    />
                    <span style={{ marginLeft: 8 }}>LLM required</span>
                </label>

                <label style={styles.check}>
                    <input
                        type="checkbox"
                        checked={includeRegex}
                        onChange={(e) => setIncludeRegex(e.target.checked)}
                    />
                    <span style={{ marginLeft: 8 }}>Include optional regex section</span>
                </label>
            </div>

            {includeRegex && (
                <div style={{ marginTop: 12, width: 260 }}>
                    <div style={styles.label}>Max regex suggestions</div>
                    <input
                        style={styles.input}
                        type="number"
                        min={0}
                        max={50}
                        value={maxRegex}
                        onChange={(e) => setMaxRegex(e.target.value)}
                    />
                </div>
            )}

            <div style={styles.rowBetween}>
                <button
                    style={{ ...styles.btn, ...(canGenerate ? styles.btnPrimary : styles.btnDisabled) }}
                    onClick={onGenerate}
                    disabled={!canGenerate}
                >
                    {busy ? "Generating…" : "Generate policy"}
                </button>

                <button
                    style={{ ...styles.btn, ...(out ? styles.btnGhost : styles.btnDisabled) }}
                    onClick={onCopy}
                    disabled={!out}
                >
                    Copy
                </button>
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

            <div style={{ marginTop: 14 }}>
                <div style={styles.hTitle}>Generated policy</div>
                <div style={styles.codeBox}>
                    <pre style={styles.pre}>{out || "—"}</pre>
                </div>
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
    headerRow: {
        display: "flex",
        justifyContent: "space-between",
        gap: 12,
        alignItems: "flex-start",
        marginBottom: 12,
    },
    hTitle: { fontSize: 16, fontWeight: 700 },
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
        marginLeft: 4,
    },
    label: { fontSize: 12, opacity: 0.85, marginBottom: 6 },
    input: {
        width: "100%",
        borderRadius: 12,
        border: "1px solid rgba(255,255,255,0.14)",
        background: "rgba(0,0,0,0.25)",
        color: "rgba(255,255,255,0.92)",
        padding: "10px 12px",
        outline: "none",
    },
    grid2: {
        display: "grid",
        gridTemplateColumns: "1fr 160px",
        gap: 12,
        marginTop: 12,
    },
    rowBetween: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 12,
        marginTop: 14,
        flexWrap: "wrap",
    },
    check: {
        display: "inline-flex",
        alignItems: "center",
        fontSize: 12,
        opacity: 0.92,
        border: "1px solid rgba(255,255,255,0.12)",
        background: "rgba(255,255,255,0.05)",
        borderRadius: 999,
        padding: "8px 10px",
    },
    btn: {
        borderRadius: 12,
        padding: "10px 12px",
        fontWeight: 700,
        border: "1px solid rgba(255,255,255,0.14)",
        cursor: "pointer",
    },
    btnPrimary: { background: "rgba(255,255,255,0.16)", color: "white" },
    btnGhost: { background: "transparent", color: "rgba(255,255,255,0.92)" },
    btnDisabled: {
        background: "rgba(255,255,255,0.06)",
        color: "rgba(255,255,255,0.45)",
        cursor: "not-allowed",
    },
    toast: {
        marginTop: 12,
        padding: "10px 12px",
        borderRadius: 12,
        border: "1px solid rgba(255,255,255,0.12)",
        fontSize: 12,
    },
    toastOk: { background: "rgba(46, 204, 113, 0.12)" },
    toastErr: { background: "rgba(231, 76, 60, 0.14)" },
    codeBox: {
        marginTop: 10,
        borderRadius: 14,
        border: "1px solid rgba(255,255,255,0.12)",
        background: "rgba(0,0,0,0.35)",
        padding: 12,
        maxHeight: 520,
        overflow: "auto",
    },
    pre: {
        margin: 0,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        fontSize: 12,
        lineHeight: 1.45,
    },
};
