import { useEffect, useState } from "react";
import API from "../api/api";
import DataTable from "../components/DataTable";

function DataView() {
  const [data, setData] = useState([]);

  useEffect(() => {
    API.get("/data").then(res => setData(res.data));
  }, []);

  return (
    <div className="card">
      <div className="card-header">
        <h2>📁 Raw Dataset</h2>
      </div>
      <div className="card-body" style={{ overflowX: "auto" }}>
        {data.length === 0 ? (
          <p style={{ color: "var(--text-muted)" }}>Loading dataset...</p>
        ) : (
          <DataTable data={data} />
        )}
      </div>
    </div>
  );
}

export default DataView;