import React, { useMemo, useState } from "react";

export default function TermReview({ topTerms, onSelectionChange }) {
    const [include, setInclude] = useState(new Set());
    const [exclude, setExclude] = useState(new Set());

    const sorted = useMemo(() => topTerms || [], [topTerms]);

    function toggle(setter, setObj, term) {
        const next = new Set(setObj);
        if (next.has(term)) next.delete(term);
        else next.add(term);
        setter(next);
    }

    function emit(nextInclude, nextExclude) {
        onSelectionChange({
            include: Array.from(nextInclude),
            exclude: Array.from(nextExclude)
        });
    }

    return (
        <div className="card">
            <h2>2) Review candidate terms</h2>
            <small>Select high-signal terms to INCLUDE, and generic ones to EXCLUDE.</small>

            <div style={{ marginTop: 14 }}>
                <div className="badge">Top extracted terms</div>
                <div style={{ marginTop: 8 }}>
                    {sorted.map((t) => (
                        <div key={t.term} className="term" title={`score=${t.score.toFixed(4)} df=${t.doc_freq}`}>
                            <input
                                type="checkbox"
                                checked={include.has(t.term)}
                                onChange={() => {
                                    const ni = new Set(include);
                                    const ne = new Set(exclude);
                                    if (ni.has(t.term)) ni.delete(t.term); else ni.add(t.term);
                                    // if included, remove from exclude
                                    ne.delete(t.term);
                                    setInclude(ni); setExclude(ne);
                                    emit(ni, ne);
                                }}
                            />
                            <span>{t.term}</span>
                            <span className="badge">{t.kind}</span>
                            <button
                                style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.16)", padding: "5px 10px" }}
                                onClick={() => {
                                    const ni = new Set(include);
                                    const ne = new Set(exclude);
                                    if (ne.has(t.term)) ne.delete(t.term); else ne.add(t.term);
                                    // if excluded, remove from include
                                    ni.delete(t.term);
                                    setInclude(ni); setExclude(ne);
                                    emit(ni, ne);
                                }}
                            >
                                {exclude.has(t.term) ? "Excluded" : "Exclude"}
                            </button>
                        </div>
                    ))}
                </div>

                <div style={{ marginTop: 16 }}>
                    <div className="badge">Include selected: {include.size}</div>{" "}
                    <div className="badge">Exclude selected: {exclude.size}</div>
                </div>
            </div>
        </div>
    );
}
