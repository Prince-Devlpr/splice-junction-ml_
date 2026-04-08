import { useState } from "react";
import API from "../api/api";

function Preprocess() {
  const [sequence, setSequence] = useState("");
  const [result, setResult] = useState(null);

  const handleProcess = async () => {
    const res = await API.post("/preprocess", { sequence });
    setResult(res.data);
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Preprocessing Visualization</h2>
      </div>
      
      <div className="card-body">
        <div className="input-group">
          <input
            type="text"
            className="styled-input"
            placeholder="Enter DNA sequence (e.g., ATGC...)"
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
          />
          <button className="primary-btn" onClick={handleProcess}>Process Sequence</button>
        </div>

        {result && (
          <div className="result-box">
            <p><strong>Original:</strong> <span className="mono-text">{result.sequence}</span></p>
            <p><strong>Encoded Length:</strong> {result.length}</p>
            <div className="code-block">
              <strong>Encoded:</strong> {JSON.stringify(result.encoded)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Preprocess;