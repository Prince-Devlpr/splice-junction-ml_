function PreprocessedTable({ data }) {
  return (
    <table border="1">
      <thead>
        <tr>
          <th>Label</th>
          <th>Sequence</th>
          <th>Encoded (partial)</th>
        </tr>
      </thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={i}>
            <td>{row.label}</td>
            <td>{row.sequence}</td>
            <td>{row.encoded.join(", ")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default PreprocessedTable;