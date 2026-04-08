import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

function ChartBox({ data }) {
  const formatted = Object.keys(data).map(key => ({
    name: key,
    value: data[key]
  }));

  return (
    <BarChart width={400} height={300} data={formatted}>
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Bar dataKey="value" />
    </BarChart>
  );
}

export default ChartBox;