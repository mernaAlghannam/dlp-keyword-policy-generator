import React, { useState } from "react";
import Upload from "./components/Upload";
import Output from "./components/Output";
import TestPolicy from "./components/TestPolicy";

export default function App() {
    const [sessionId, setSessionId] = useState("");
    const [page, setPage] = useState("generate");
    const [policyText, setPolicyText] = useState("");

    return (
        <div className="container">
            <div className="hero">
                <h1>DLP Keyword Policy Generator</h1>
                <p>Analyze → Generate policy → Test on another set of docs.</p>

                <div className="row" style={{ marginTop: 10, gap: 10, alignItems: "center" }}>
                    <button
                        className={`btn ${page === "generate" ? "btnPrimary" : "btnGhost"}`}
                        onClick={() => setPage("generate")}
                    >
                        Generate
                    </button>

                    <button
                        className={`btn ${page === "test" ? "btnPrimary" : "btnGhost"}`}
                        onClick={() => setPage("test")}
                    >
                        Test
                    </button>

                    <span className="badge">
                        Session <span className="kbd">{sessionId ? sessionId.slice(0, 8) + "…" : "—"}</span>
                    </span>
                </div>
            </div>

            {page === "generate" ? (
                <div className="grid">
                    <Upload onSession={(sid) => setSessionId(sid)} />
                    <Output sessionId={sessionId} onPolicy={(txt) => setPolicyText(txt)} />
                </div>
            ) : (
                <div className="grid">
                    <TestPolicy sessionId={sessionId} generatedPolicyText={policyText} />
                </div>
            )}
        </div>
    );
}
