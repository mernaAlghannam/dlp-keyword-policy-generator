// frontend/src/api.js
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const http = axios.create({
    baseURL: API_BASE,
    timeout: 120000,
});

export async function analyzeDocs(files) {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);

    const res = await http.post("/api/analyze", fd, {
        headers: { "Content-Type": "multipart/form-data" },
    });

    return res.data; // IMPORTANT: return full JSON (topics + top_terms + etc.)
}

export async function generatePretty(payload) {
    const res = await http.post("/api/generate_pretty", payload, {
        responseType: "text", // IMPORTANT: backend should return plain text
        headers: { "Content-Type": "application/json" },
    });

    return res.data; // plain text
}


export async function testPolicy({ policy_text, files }) {
    const fd = new FormData();
    fd.append("policy_text", policy_text);
    for (const f of files) fd.append("files", f);

    const res = await http.post("/api/test_policy", fd, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
}

