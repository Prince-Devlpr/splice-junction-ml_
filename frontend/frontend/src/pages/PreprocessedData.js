import { useEffect, useState } from "react";
import API from "../api/api";
import PreprocessedTable from "../components/PreprocessedTable";

function PreprocessedData() {
  const [data, setData] = useState([]);

  useEffect(() => {
    API.get("/data/preprocessed").then(res => setData(res.data));
  }, []);

  return (
    <div className="card">
      <div className="card-header">
        <h2>⚙️ Preprocessed Dataset</h2>
      </div>
      <div className="card-body" style={{ overflowX: "auto" }}>
        {data.length === 0 ? (
          <p style={{ color: "var(--text-muted)" }}>Loading preprocessed data...</p>
        ) : (
          <PreprocessedTable data={data} />
        )}
      </div>
    </div>
  );
}

export default PreprocessedData;