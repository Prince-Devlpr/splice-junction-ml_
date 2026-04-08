function DataTable({ data }) {
  return (
    <table border="1">
      <thead>
        <tr>
          <th>Label</th>
          <th>ID</th>
          <th>Sequence</th>
        </tr>
      </thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={i}>
            <td>{row.label}</td>
            <td>{row.id}</td>
            <td>{row.sequence}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default DataTable;