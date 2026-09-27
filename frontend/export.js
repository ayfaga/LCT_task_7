/* Pure export serializers plus one browser download path; no UI styling. */
(function (root) {
  function soloJson(payload) {
    return `${JSON.stringify(payload, null, 2)}\n`;
  }

  function csvCell(value) {
    return `"${String(value ?? '').replace(/"/g, '""')}"`;
  }

  function manyCsv(results) {
    const lines = [['query_file', 'status', 'rank', 'gallery_id', 'confidence_cosine', 'accepted'].map(csvCell).join(',')];
    for (const { item, payload, error } of results) {
      if (error || !payload) {
        lines.push([item.name, 'error', '', '', '', ''].map(csvCell).join(','));
        continue;
      }
      const ranked = Array.isArray(payload.ranked) ? payload.ranked.slice(0, 10) : [];
      const accepted = new Set((Array.isArray(payload.accepted) ? payload.accepted : []).map((row) => row.gallery_id));
      if (!ranked.length) lines.push([item.name, payload.status, '', '', '', ''].map(csvCell).join(','));
      ranked.forEach((candidate, index) => lines.push([
        item.name, payload.status, index + 1, candidate.gallery_id, candidate.confidence,
        accepted.has(candidate.gallery_id) ? 'true' : 'false',
      ].map(csvCell).join(',')));
    }
    return `${lines.join('\n')}\n`;
  }

  function download(filename, content, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    try { link.click(); } finally {
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    }
  }

  const api = { soloJson, manyCsv, download };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReidExport = api;
}(typeof window === 'undefined' ? globalThis : window));
