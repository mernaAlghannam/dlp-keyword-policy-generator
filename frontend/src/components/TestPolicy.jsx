import React, { useEffect, useMemo, useState } from "react";
import { testPolicy } from "../api";

export default function TestPolicy({ sessionId, generatedPolicyText }) {
    const [policyText, setPolicyText] = useState(generatedPolicyText || "");
    const [files, setFiles] = useState([]);
    const [busy, setBusy] = useState(false);
    const [msg, setMsg] = useState("");
    const [res, setRes] = useState(null);

    // ✅ Put useEffect here (after useState, before useMemo)
    useEffect(() => {
        // If you regenerate a policy on the Generate page, refresh it here automatically
        setPolicyText(generatedPolicyText || "");
    }, [generatedPolicyText]);

    const canRun = useMemo(
        () => (policyText || "").trim() && files.length > 0 && !busy,
        [policyText, files, busy]
    );

    async function onRun() {
        setBusy(true);
        setMsg("");
        setRes(null);
        try {
            const data = await testPolicy({ policy_text: policyText, files });
            setRes(data);
            setMsg("Test complete.");
        } catch (e) {
            setMsg(e?.response?.data?.detail || e?.message || "Test failed.");
        } finally {
            setBusy(false);
        }
    }

    return (
        <div className="card">
            <div className="cardHeader">
                <h3 className="sectionTitle">3) Test policy on files</h3>
                <p className="sectionHint">Upload a folder of documents and see which rules match each file.</p>
            </div>

            <div className="cardBody">
                <input
                    className="input"
                    type="file"
                    multiple
                    accept=".txt,.pdf,.docx"
                    onChange={(e) => setFiles(Array.from(e.target.files || []))}
                />
                <div className="small" style={{ marginTop: 8 }}>
                    {files.length} file(s) selected
                </div>

                <div className="row" style={{ justifyContent: "space-between", marginTop: 12 }}>
                    <button className="btn btnPrimary" onClick={onRun} disabled={!canRun}>
                        {busy ? "Testing…" : "Run test"}
                    </button>
                </div>

                {msg && (
                    <div className={`toast ${msg.toLowerCase().includes("fail") ? "err" : "ok"}`} style={{ marginTop: 12 }}>
                        {msg}
                    </div>
                )}

                {res && (
                    <>
                        <div className="divider" />
                        <h3 className="sectionTitle">Results</h3>
                        <div className="codeBox" style={{ marginTop: 10 }}>
                            <pre>{JSON.stringify(res, null, 2)}</pre>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
