import { useEffect, useMemo, useState } from "react";
import API from "../api/api";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  Legend,
  LineChart,
  Line
} from "recharts";

const CHART_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

function Analytics() {
  const [classData, setClassData] = useState(null);
  const [nucleotideData, setNucleotideData] = useState(null);
  const [positionwiseData, setPositionwiseData] = useState(null);
  const [pcaData, setPcaData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadAnalytics = async () => {
      try {
        const [classesRes, nucleotidesRes, positionRes, pcaRes] = await Promise.all([
          API.get("/analytics/classes"),
          API.get("/analytics/nucleotides"),
          API.get("/analytics/positionwise"),
          API.get("/analytics/pca")
        ]);

        setClassData(classesRes.data);
        setNucleotideData(nucleotidesRes.data);
        setPositionwiseData(positionRes.data);
        setPcaData(pcaRes.data);
      } catch {
        setError("Failed to load analytics graphs.");
      }
    };

    loadAnalytics();
  }, []);

  const classRows = useMemo(
    () =>
      classData
        ? Object.entries(classData).map(([label, count]) => ({ label, count }))
        : [],
    [classData]
  );

  const nucleotideRows = useMemo(
    () =>
      nucleotideData
        ? Object.entries(nucleotideData).map(([nucleotide, count]) => ({
            nucleotide,
            count
          }))
        : [],
    [nucleotideData]
  );

  const pcaPoints = useMemo(
    () => (pcaData && Array.isArray(pcaData.points) ? pcaData.points : []),
    [pcaData]
  );

  const pcaLabels = useMemo(
    () => [...new Set(pcaPoints.map((point) => point.label))],
    [pcaPoints]
  );

  const pcaScatterSets = useMemo(
    () =>
      pcaLabels.map((label) => ({
        label,
        points: pcaPoints.filter((point) => point.label === label)
      })),
    [pcaLabels, pcaPoints]
  );

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <p style={{ color: "#dc2626" }}>{error}</p>
        </div>
      </div>
    );
  }

  const loading = !classData || !nucleotideData || !positionwiseData || !pcaData;

  return (
    <div className="card">
      <div className="card-header">
        <h2>📊 Dataset Analytics Graphs</h2>
      </div>
      <div className="card-body">
        {loading ? (
          <p style={{ color: "var(--text-muted)" }}>Loading analytics...</p>
        ) : (
          <div className="analytics-chart-grid">
            <div className="analytics-chart-card">
              <h3>Class Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={classRows}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="analytics-chart-card">
              <h3>Nucleotide Frequency</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={nucleotideRows} dataKey="count" nameKey="nucleotide" outerRadius={110} label>
                    {nucleotideRows.map((entry, index) => (
                      <Cell key={`cell-${entry.nucleotide}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="analytics-chart-card analytics-chart-card-wide">
              <h3>PCA Visualization</h3>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="pc1" name="PC1" />
                  <YAxis type="number" dataKey="pc2" name="PC2" />
                  <Tooltip />
                  <Legend />
                  {pcaScatterSets.map((entry, index) => (
                    <Scatter
                      key={entry.label}
                      name={entry.label}
                      data={entry.points}
                      fill={CHART_COLORS[index % CHART_COLORS.length]}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="analytics-chart-card analytics-chart-card-wide">
              <h3>Position-wise Nucleotide Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={positionwiseData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="position" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="A" stroke="#3b82f6" dot={false} />
                  <Line type="monotone" dataKey="C" stroke="#10b981" dot={false} />
                  <Line type="monotone" dataKey="G" stroke="#f59e0b" dot={false} />
                  <Line type="monotone" dataKey="T" stroke="#ef4444" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Analytics;
