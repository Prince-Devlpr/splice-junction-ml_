import { useEffect, useState } from "react";
import API from "../api/api";

function Comparison() {
  const [data, setData] = useState(null);

  useEffect(() => {
    API.get("/metrics/compare").then(res => setData(res.data));
  }, []);

  if (!data) {
    return (
      <div className="card">
        <div className="card-body">
          <p style={{ color: "var(--text-muted)" }}>Loading model comparison...</p>
        </div>
      </div>
    );
  }

  // Helper to format decimals to percentages
  const formatPercent = (val) => `${(val * 100).toFixed(1)}%`;

  return (
    <div className="card">
      <div className="card-header">
        <h2>⚖️ Model Comparison</h2>
      </div>
      
      <div className="card-body" style={{ overflowX: "auto" }}>
        <table className="styled-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>KBANN Performance</th>
              <th>Deep Learning Performance</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Accuracy</strong></td>
              <td>{formatPercent(data.KBANN.accuracy)}</td>
              <td>{formatPercent(data.DeepLearning.accuracy)}</td>
            </tr>
            <tr>
              <td><strong>Precision</strong></td>
              <td>{formatPercent(data.KBANN.precision)}</td>
              <td>{formatPercent(data.DeepLearning.precision)}</td>
            </tr>
            <tr>
              <td><strong>Recall</strong></td>
              <td>{formatPercent(data.KBANN.recall)}</td>
              <td>{formatPercent(data.DeepLearning.recall)}</td>
            </tr>
            <tr>
              <td><strong>F1 Score</strong></td>
              <td>{formatPercent(data.KBANN.f1_score)}</td>
              <td>{formatPercent(data.DeepLearning.f1_score)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Comparison;