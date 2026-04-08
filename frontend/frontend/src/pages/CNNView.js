import { useState } from "react";
import API from "../api/api";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip
} from "recharts";

function CNNView() {
  const [sequence, setSequence] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleCheck = async () => {
    if (!sequence.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await API.post("/predict/dl", { sequence });
      setResult(res.data);
    } catch (err) {
      const message = err.response?.data?.error || "Failed to analyze sequence with CNN.";
      setError(message);
      setResult(null);
    }
    setLoading(false);
  };

  const getSequenceStats = () => {
    if (!result?.sequence) return null;
    const seq = result.sequence.toUpperCase();
    const length = seq.length;
    const counts = { A: 0, C: 0, G: 0, T: 0 };
    for (const ch of seq) {
      if (counts[ch] !== undefined) counts[ch] += 1;
    }
    const gcContent = length > 0 ? ((counts.G + counts.C) / length) * 100 : 0;
    return { length, ...counts, gcContent };
  };

  const probabilityRows = result
    ? Object.entries(result.probabilities || {}).map(([label, probability]) => ({
        label,
        probability
      }))
    : [];

  const stats = getSequenceStats();

  return (
    <div className="card">
      <div className="card-header">
        <h2>🤖 CNN-Only Junction Prediction</h2>
      </div>

      <div className="card-body">
        <div className="input-group">
          <input
            className="styled-input"
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Enter DNA sequence (e.g. ATGC...)"
          />
          <button className="primary-btn" onClick={handleCheck} disabled={loading}>
            {loading ? "Predicting..." : "Predict with CNN"}
          </button>
        </div>

        {error && <p style={{ color: "#dc2626", marginTop: "-0.5rem" }}>{error}</p>}

        {result && (
          <div className="result-box dual-model-result">
            <p>
              <strong>Sequence:</strong> <span className="mono-text">{result.sequence}</span>
            </p>

            {stats && (
              <div className="analysis-summary-grid">
                <div className="analysis-summary-card">
                  <span className="analysis-summary-label">Length</span>
                  <span className="analysis-summary-value">{stats.length}</span>
                </div>
                <div className="analysis-summary-card">
                  <span className="analysis-summary-label">GC Content</span>
                  <span className="analysis-summary-value">{stats.gcContent.toFixed(1)}%</span>
                </div>
                <div className="analysis-summary-card">
                  <span className="analysis-summary-label">A / C / G / T</span>
                  <span className="analysis-summary-value">
                    {stats.A} / {stats.C} / {stats.G} / {stats.T}
                  </span>
                </div>
              </div>
            )}

            <div className="metrics-chart-card">
              <h3>CNN Class Probabilities</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={probabilityRows}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                  <Bar dataKey="probability" name="CNN Probability" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}

export default CNNView;
