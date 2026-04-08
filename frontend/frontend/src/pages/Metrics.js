import { useEffect, useState } from "react";
import API from "../api/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from "recharts";

function Metrics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    API.get("/metrics/compare")
      .then((res) => setData(res.data))
      .catch(() => setError("Failed to load model metrics."));
  }, []);

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <p style={{ color: "#dc2626" }}>{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card">
        <div className="card-body">
          <p style={{ color: "var(--text-muted)" }}>Loading metrics...</p>
        </div>
      </div>
    );
  }

  const kbann = data.KBANN;
  const dl = data.DeepLearning;
  const formatPercent = (val) => `${(val * 100).toFixed(1)}%`;

  const metricRows = [
    { metric: "Accuracy", kbann: kbann.accuracy, dl: dl.accuracy },
    { metric: "Precision", kbann: kbann.precision, dl: dl.precision },
    { metric: "Recall", kbann: kbann.recall, dl: dl.recall },
    { metric: "F1 Score", kbann: kbann.f1_score, dl: dl.f1_score }
  ];

  const renderMatrix = (matrix) => (
    <table className="matrix-table">
      <thead>
        <tr>
          <th>Actual \ Pred</th>
          <th>EI</th>
          <th>IE</th>
          <th>N</th>
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, rowIdx) => (
          <tr key={rowIdx}>
            <td>{["EI", "IE", "N"][rowIdx]}</td>
            {row.map((value, colIdx) => (
              <td key={colIdx}>{value}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );

  return (
    <div className="card">
      <div className="card-header">
        <h2>📈 Model Metrics Dashboard (KBANN + Deep Learning)</h2>
      </div>

      <div className="card-body">
        <div className="stats-grid">
          {metricRows.map((item) => (
            <div key={item.metric} className="stat-box">
              <span className="stat-label">{item.metric}</span>
              <span className="stat-value">{formatPercent(Math.max(item.kbann, item.dl))}</span>
              <span style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.4rem" }}>
                KBANN: {formatPercent(item.kbann)} | DL: {formatPercent(item.dl)}
              </span>
            </div>
          ))}
        </div>

        <div className="metrics-chart-grid">
          <div className="metrics-chart-card">
            <h3>Grouped Metric Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={metricRows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis domain={[0.9, 1]} />
                <Tooltip formatter={(v) => formatPercent(v)} />
                <Legend />
                <Bar dataKey="kbann" name="KBANN" fill="#3b82f6" />
                <Bar dataKey="dl" name="Deep Learning" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="metrics-chart-card">
            <h3>Performance Shape (Radar)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={metricRows}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis domain={[0.9, 1]} />
                <Radar name="KBANN" dataKey="kbann" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Radar name="Deep Learning" dataKey="dl" stroke="#10b981" fill="#10b981" fillOpacity={0.25} />
                <Legend />
                <Tooltip formatter={(v) => formatPercent(v)} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="metrics-chart-grid">
          <div className="metrics-chart-card">
            <h3>KBANN Confusion Matrix</h3>
            {renderMatrix(kbann.confusion_matrix)}
          </div>

          <div className="metrics-chart-card">
            <h3>Deep Learning Confusion Matrix</h3>
            {renderMatrix(dl.confusion_matrix)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Metrics;
