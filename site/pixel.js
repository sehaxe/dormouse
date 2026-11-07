(() => {
  const d = JSON.parse(document.getElementById('data').textContent);
  const fmt = (x, n = 1) => Number(x).toFixed(n);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ACCENT = '#00e08a';
  const HOT = '#ff3b5c';

  /* ---------------------------------------------------------------- clock */
  const clock = document.getElementById('clock');
  if (clock) {
    const tick = () => {
      const t = new Date();
      clock.textContent = [t.getHours(), t.getMinutes(), t.getSeconds()]
        .map(v => String(v).padStart(2, '0')).join(':');
    };
    tick();
    setInterval(tick, 1000);
  }

  /* ------------------------------------------------------- pixel headline */
  // Draw at 8 px, threshold to 1 bit, then let CSS scale it with
  // image-rendering:pixelated. That is what makes it a bitmap rather than
  // blurry text — no font file, no font licence, no webfont request.
  const title = document.getElementById('pixTitle');
  const word = 'dormouse.';
  if (title) {
    const off = document.createElement('canvas');
    const octx = off.getContext('2d', { willReadFrequently: true });
    const FONT = 'bold 14px "DejaVu Sans Mono", "Liberation Mono", monospace';
    octx.font = FONT;
    const w = Math.max(8, Math.ceil(octx.measureText(word).width));
    off.width = w;
    off.height = 16;
    octx.font = FONT;
    octx.textBaseline = 'alphabetic';
    octx.fillStyle = '#fff';
    octx.fillText(word, 0, 12);

    const img = octx.getImageData(0, 0, w, 16);
    for (let i = 0; i < img.data.length; i += 4) {
      const on = img.data[i + 3] > 128;
      img.data[i] = 255;
      img.data[i + 1] = 255;
      img.data[i + 2] = 255;
      img.data[i + 3] = on ? 255 : 0;
    }
    octx.putImageData(img, 0, 0);

    title.width = w;
    title.height = 16;
    const ctx = title.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    const paint = (cols) => {
      ctx.clearRect(0, 0, w, 16);
      ctx.drawImage(off, 0, 0, Math.max(1, cols), 16, 0, 0, Math.max(1, cols), 16);
    };
    paint(w);   // no wipe: a headline must never be caught half-drawn
  }

  /* ---------------------------------------------------------- binary rain */
  const cv = document.getElementById('rain');
  if (cv && !reduced) {
    const gl = cv.getContext('webgl', { antialias: false, alpha: true, premultipliedAlpha: true });
    if (gl) {
      const at = document.createElement('canvas');
      at.width = 64; at.height = 64;
      const c2 = at.getContext('2d');
      c2.fillStyle = '#000'; c2.fillRect(0, 0, 64, 64);
      c2.fillStyle = '#fff';
      c2.font = 'bold 54px "DejaVu Sans Mono", monospace';
      c2.textAlign = 'center'; c2.textBaseline = 'middle';
      c2.fillText('0', 16, 34); c2.fillText('1', 48, 34);

      const vs = 'attribute vec2 p; void main(){ gl_Position = vec4(p,0.0,1.0); }';
      const fs = `precision highp float;
uniform vec2 u_res; uniform float u_t; uniform sampler2D u_atlas;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
void main(){
  float cellW = 15.0, cellH = 19.0;
  vec2 g = gl_FragCoord.xy / vec2(cellW, cellH);
  float col = floor(g.x);
  float rowAll = g.y;
  float fx = fract(g.x), fy = 1.0 - fract(g.y);
  float speed = 0.9 + 2.4 * hash(vec2(col, 11.0));
  float phase = hash(vec2(col, 23.0));
  float rows = u_res.y / cellH;
  float headRow = mod(u_t * speed + phase * rows, rows + 26.0) - 13.0;
  float d = headRow - rowAll;
  float trail = d >= 0.0 ? exp(-d * 0.16) : 0.0;
  float cell = floor(rowAll);
  float rate = floor(u_t * (2.0 + 5.0 * hash(vec2(col, 7.0))));
  float pick = hash(vec2(col, cell) + rate);
  float glyph = texture2D(u_atlas, vec2(step(0.5, pick) * 0.5 + fx * 0.5, fy)).r;
  float a = trail * glyph;
  float head = smoothstep(1.6, 0.0, abs(d));
  vec3 green = vec3(0.0, 0.88, 0.54);
  vec3 c = mix(green * 0.45, vec3(0.75, 1.0, 0.88), head);
  a = step(0.16, a) * clamp(a * 1.7, 0.0, 1.0);
  gl_FragColor = vec4(c * a, a);
}`;
      const sh = (t, src) => { const o = gl.createShader(t); gl.shaderSource(o, src); gl.compileShader(o); return o; };
      const prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(prog);
      if (gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        gl.useProgram(prog);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(prog, 'p');
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, at);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_atlas'), 0);
        const uRes = gl.getUniformLocation(prog, 'u_res');
        const uT = gl.getUniformLocation(prog, 'u_t');
        const resize = () => {
          const dpr = Math.min(devicePixelRatio || 1, 1.5);
          const w = Math.max(1, Math.floor(cv.clientWidth * dpr));
          const h = Math.max(1, Math.floor(cv.clientHeight * dpr));
          if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
          gl.viewport(0, 0, w, h);
          gl.uniform2f(uRes, w, h);
        };
        const draw = (t) => { gl.uniform1f(uT, t); gl.drawArrays(gl.TRIANGLES, 0, 3); };
        resize(); draw(6.0);
        let raf = 0, t0 = performance.now(), vis = true;
        const loop = (now) => { if (!vis) return; raf = requestAnimationFrame(loop); resize(); draw((now - t0) / 1000); };
        new IntersectionObserver(([e]) => {
          vis = e.isIntersecting;
          if (vis) { cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); }
        }, { threshold: 0 }).observe(cv);
        addEventListener('resize', resize);
        raf = requestAnimationFrame(loop);
      } else {
        cv.style.display = 'none';
      }
    }
  }

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
      <td class="${r.tag === 'muon_fp32' ? 'dim' : (r.safe ? 'ok' : 'bad')}">${r.tag === 'muon_fp32' ? 'baseline' : fmt(rel, 2) + '×'}</td>
      ${q}</tr>`;
  }).join('');

  /* ------------------------------------------------------------- the curve */
  const pts = d.curve || [];
  const svg = document.getElementById('curve');
  if (pts.length > 1) {
    const W = 720, H = 240, L = 58, R = 18, T = 14, B = 34;
    const xs = pts.map(p => p.mb), ys = pts.map(p => p.bpb);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys) - 0.04, y1 = Math.max(...ys) + 0.04;
    const X = v => L + (v - x0) / Math.max(x1 - x0, 1e-9) * (W - L - R);
    const Y = v => T + (1 - (v - y0) / Math.max(y1 - y0, 1e-9)) * (H - T - B);
    const grid = [0, .25, .5, .75, 1].map(f => {
      const y = T + f * (H - T - B);
      return `<line class="grid" x1="${L}" y1="${y.toFixed(1)}" x2="${W - R}" y2="${y.toFixed(1)}"/>` +
        `<text x="0" y="${(y + 3).toFixed(1)}">${fmt(y1 - f * (y1 - y0), 2)}</text>`;
    }).join('');
    const path = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.mb).toFixed(1)},${Y(p.bpb).toFixed(1)}`).join(' ');
    const dots = pts.map(p => `<circle class="pt" cx="${X(p.mb).toFixed(1)}" cy="${Y(p.bpb).toFixed(1)}" r="2.2"/>`).join('');
    const last = pts[pts.length - 1];
    svg.innerHTML = grid + `<path class="line" d="${path}"/>${dots}` +
      `<text x="${L}" y="${H - 10}">${fmt(x0, 0)} MB</text>` +
      `<text x="${W - R}" y="${H - 10}" text-anchor="end">${fmt(x1, 0)} MB</text>` +
      `<text x="${(X(last.mb) - 8).toFixed(1)}" y="${(Y(last.bpb) - 9).toFixed(1)}" text-anchor="end">${fmt(last.bpb, 3)} BPB</text>`;
    document.getElementById('curveMeta').textContent =
      `${pts.length} held-out points · ${fmt(x1, 0)} MB seen · last ${fmt(last.bpb, 3)} BPB`;
    document.getElementById('hBpb').textContent = fmt(last.bpb, 3);
    document.getElementById('hMb').textContent = fmt(last.mb, 0);
  }

  if (speedup) {
    document.getElementById('hSpeed').textContent = fmt(speedup, 2) + '×';
    const cs = document.getElementById('cSpeed');
    if (cs) cs.textContent = fmt(speedup, 2) + '×';
  }
  document.getElementById('cRuns').textContent = d.stats.runs_on_record;
  document.getElementById('cRunsSub').textContent = d.stats.runs_with_bpb + ' with a held-out BPB';

  /* ------------------------------------------------------------- the runs */
  document.getElementById('runs').innerHTML = (d.runs || []).map(r => `<tr>
    <td>${r.run}</td><td class="dim">${r.source || '—'}</td><td class="dim">${r.opt || '—'}</td>
    <td class="dim">${r.batch}×${r.seq}</td><td>${fmt(r.mb, 0)}</td><td class="ok">${fmt(r.bpb, 3)}</td>
  </tr>`).join('');

  /* --------------------------------------------------------- the portrait */
  const sj = JSON.parse(document.getElementById('surprise').textContent || 'null');
  if (sj && sj.bits && sj.text) {
    const cols = sj.cols || 64;
    const bits = sj.bits;
    const band = (b) => b < 0.6 ? 0 : b < 1.6 ? 1 : b < 2.6 ? 2 : b < 4.2 ? 3 : 4;
    const col = ['#2b3742', '#3f5147', '#2f7a5c', '#00b273', ACCENT];
    const esc = (c) => c === '&' ? '&amp;' : c === '<' ? '&lt;' : c === '>' ? '&gt;' : c;
    let out = '';
    for (let i = 0; i < bits.length; i++) {
      const bd = band(bits[i]);
      out += `<b style="color:${col[bd]}">${esc(sj.text[i])}</b>`;
      if ((i + 1) % cols === 0) out += '\n';
    }
    document.getElementById('bytes').innerHTML = out.trimEnd();

    const hc = document.getElementById('heat');
    if (hc) {
      hc.width = bits.length;
      const ctx = hc.getContext('2d');
      const maxB = Math.max(...bits, 1);
      for (let i = 0; i < bits.length; i++) {
        const v = bits[i] / maxB;
        const h = Math.max(2, Math.round(v * 52));
        ctx.fillStyle = col[band(bits[i])];
        ctx.globalAlpha = 0.2 + 0.8 * v;
        ctx.fillRect(i, 52 - h, 1, h);
      }
      ctx.globalAlpha = 1;
    }
    const mean = bits.reduce((a, b) => a + b, 0) / bits.length;
    document.getElementById('sMean').textContent = `mean ${mean.toFixed(3)} bits / byte over ${bits.length} held-out bytes`;
    document.getElementById('sMeta').textContent = `${sj.ckpt} · off=${sj.offset}`;
  }

  /* --------------------------------------------------------------- the tape */
  const facts = [
    'flagship3 · 1.520 bits per byte at 246 MB',
    '95,035,200 parameters · 12 unique layers · d_model 832',
    '1M context · 34 MB of state',
    'bf16 Newton–Schulz · 1.92× the step for +0.01 BPB',
    '3-step Newton–Schulz rejected · +0.27 BPB at [832, 3072]',
    'one RTX 5070 Ti · 16 GB · sm_120',
    'every number here is computed from the run logs'
  ];
  const tape = document.getElementById('tape');
  if (tape) {
    const line = facts.map(t => `<b>·</b> ${t} `).join('');
    tape.innerHTML = line + line;
  }

  document.getElementById('built').textContent =
    `built from ${d.stats.runs_on_record} log files`;
})();
