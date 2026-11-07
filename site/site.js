(() => {
  const d = JSON.parse(document.getElementById('data').textContent);
  const fmt = (x, n = 1) => Number(x).toFixed(n);

  /* ------------------------------------------------------------- the rungs */
  const rungs = d.rungs || [];
  const control = rungs.find(r => r.tag === 'muon_fp32');
  const safe = rungs.filter(r => r.safe && r.tag !== 'muon_fp32');
  const bestSafe = safe.reduce((a, b) => (!a || b.kbs > a.kbs ? b : a), null);
  const speedup = control && bestSafe ? bestSafe.kbs / control.kbs : null;

  document.getElementById('rungs').innerHTML = rungs.map(r => {
    const rel = control ? r.kbs / control.kbs : 1;
    const note = r.safe ? '' : ' <span class="bad">[rejected]</span>';
    const d2 = r.dbpb;
    const q = d2 == null ? '<td class="dim">—</td>'
      : `<td class="${r.safe ? 'ok' : 'bad'}">${d2 > 0 ? '+' : ''}${fmt(d2, 2)}</td>`;
    return `<tr>
      <td>${r.label}${note}</td>
      <td>${fmt(r.kbs, 1)}</td>
      <td>${fmt(r.ms_per_step, 0)}</td>
      <td class="${r.tag === 'muon_fp32' ? 'note' : (r.safe ? 'ok' : 'bad')}">${r.tag === 'muon_fp32' ? 'baseline' : fmt(rel, 2) + '×'}</td>
      ${q}</tr>`;
  }).join('');

  if (speedup) document.getElementById('sSpeed').textContent = fmt(speedup, 2) + '×';
  document.getElementById('sRuns').textContent = d.stats.runs_on_record;
  document.getElementById('sRunsSub').textContent = d.stats.runs_with_bpb + ' with a held-out BPB';

  /* ------------------------------------------------------------- the curve */
  const pts = d.curve || [];
  if (pts.length) {
    const last = pts[pts.length - 1];
    document.getElementById('curveMeta').textContent =
      `${pts.length} held-out points over ${fmt(last.mb, 0)} MB of training bytes.`;
    document.getElementById('curveFinal').innerHTML =
      `<span class="mono">${fmt(last.bpb, 3)}</span> bits per byte at ${fmt(last.mb, 0)} MB`;
  }

  /* ------------------------------------------------------------- the runs */
  document.getElementById('runsBody').innerHTML = (d.runs || []).map(r => `<tr>
    <td>${r.run}</td><td>${r.source || '—'}</td><td>${r.opt || '—'}</td>
    <td>${r.batch}×${r.seq}</td><td>${fmt(r.mb, 0)}</td><td class="ok">${fmt(r.bpb, 3)}</td>
  </tr>`).join('');

  /* --------------------------------------------------------- the portrait */
  const sj = JSON.parse(document.getElementById('surprise').textContent || 'null');
  if (sj && sj.bits && sj.text) {
    const cols = sj.cols || 64;
    const bits = sj.bits;
    const band = (b) => b < 0.6 ? 0 : b < 1.6 ? 1 : b < 2.6 ? 2 : b < 4.2 ? 3 : 4;
    const col = ['#3a3a40', '#565660', '#8a8a96', '#c9c9d2', '#ffffff'];
    const esc = (c) => c === '&' ? '&amp;' : c === '<' ? '&lt;' : c === '>' ? '&gt;' : c;
    let out = '';
    for (let i = 0; i < bits.length; i++) {
      out += `<b style="color:${col[band(bits[i])]};font-weight:400">${esc(sj.text[i])}</b>`;
      if ((i + 1) % cols === 0) out += '\n';
    }
    document.getElementById('bytes').innerHTML = out.trimEnd();
    const mean = bits.reduce((a, b) => a + b, 0) / bits.length;
    document.getElementById('sMean').textContent = `mean ${mean.toFixed(3)} bits / byte`;
    document.getElementById('sMeta').textContent = `off=${sj.offset}`;
  }

  document.getElementById('built').textContent = `${d.stats.runs_on_record} log files`;
})();
