import { useState } from "react";
import API from "../api/api";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend
} from "recharts";

function ProbabilityBars({ probabilities }) {
  return (
    <div className="probability-list">
      {Object.entries(probabilities).map(([label, value]) => (
        <div key={label} className="probability-row">
          <div className="probability-head">
            <span>{label}</span>
            <span>{(value * 100).toFixed(1)}%</span>
          </div>
          <div className="probability-track">
            <div
              className="probability-fill"
              style={{ width: `${(value * 100).toFixed(1)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function KBANNView() {
  const [sequence, setSequence] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleCheck = async () => {
    if (!sequence.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await API.post("/predict/compare", { sequence });
      setResult(res.data);
    } catch (err) {
      const message = err.response?.data?.error || "Failed to analyze sequence.";
      setError(message);
      setResult(null);
    }
    setLoading(false);
  };

  const buildComparativeProbabilities = () => {
    if (!result) return [];
    const labels = Object.keys(result.KBANN?.probabilities || {});
    return labels.map((label) => ({
      label,
      kbann: result.KBANN.probabilities[label] ?? 0,
      dl: result.DeepLearning.probabilities[label] ?? 0
    }));
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

  const probabilityRows = buildComparativeProbabilities();
  const stats = getSequenceStats();

  return (
    <div className="card">
      <div className="card-header">
        <h2>🧬 Dual-Model Junction Prediction</h2>
      </div>
      
      <div className="card-body">
        <div className="input-group">
          <input
            className="styled-input"
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Enter DNA sequence (e.g. ATGC...)"
          />
          <button 
            className="primary-btn" 
            onClick={handleCheck} 
            disabled={loading}
          >
            {loading ? "Predicting..." : "Predict in Both Models"}
          </button>
        </div>

        {error && <p style={{ color: "#dc2626", marginTop: "-0.5rem" }}>{error}</p>}

        {result && (
          <div className="result-box dual-model-result">
            <p>
              <strong>Sequence:</strong>{" "}
              <span className="mono-text">{result.sequence}</span>
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
              <h3>Class Probability Comparison (KBANN vs Deep Learning)</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={probabilityRows}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                  <Legend />
                  <Bar dataKey="kbann" name="KBANN" fill="#3b82f6" />
                  <Bar dataKey="dl" name="Deep Learning" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="metrics-chart-card">
              <h3>Confidence Gap</h3>
              <p style={{ margin: 0 }}>
                KBANN: <strong>{(result.KBANN.confidence * 100).toFixed(1)}%</strong> | Deep Learning:{" "}
                <strong>{(result.DeepLearning.confidence * 100).toFixed(1)}%</strong> | Difference:{" "}
                <strong>{Math.abs((result.KBANN.confidence - result.DeepLearning.confidence) * 100).toFixed(1)}%</strong>
              </p>
            </div>

            <div className="model-grid">
              <div className="model-card">
                <h4>KBANN</h4>
                <p>
                  Predicted Class:{" "}
                  <span className="prediction-badge">{result.KBANN.prediction}</span>
                </p>
                <p>
                  Confidence: {(result.KBANN.confidence * 100).toFixed(1)}%
                </p>
                <ProbabilityBars probabilities={result.KBANN.probabilities} />
                <h5>Rule Features</h5>
                <pre className="code-block">
                  {JSON.stringify(result.KBANN.features, null, 2)}
                </pre>
              </div>

              <div className="model-card">
                <h4>Deep Learning</h4>
                <p>
                  Predicted Class:{" "}
                  <span className="prediction-badge">{result.DeepLearning.prediction}</span>
                </p>
                <p>
                  Confidence: {(result.DeepLearning.confidence * 100).toFixed(1)}%
                </p>
                <ProbabilityBars probabilities={result.DeepLearning.probabilities} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default KBANNView;
