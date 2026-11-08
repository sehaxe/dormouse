(() => {
  /* ---------------------------------------------------------------- theme */
  // auto -> follows prefers-color-scheme; light/dark pin it. Persisted, and the
  // attribute is set before the page paints (the script is at the end of body,
  // so the browser's own color-scheme picks the first paint and this only
  // confirms it).
  const html = document.documentElement;
  const sys = matchMedia('(prefers-color-scheme: dark)');
  const saved = () => { try { return localStorage.getItem('dormouse-theme') || 'auto'; } catch { return 'auto'; } };
  const resolve = (m) => (m === 'auto' ? (sys.matches ? 'dark' : 'light') : m);
  const apply = () => {
    const mode = saved();
    html.dataset.theme = resolve(mode);
    const btn = document.getElementById('themeBtn');
    if (btn) btn.textContent = mode;
  };
  apply();
  sys.addEventListener('change', apply);
  addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('themeBtn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const order = ['auto', 'light', 'dark'];
      const next = order[(order.indexOf(saved()) + 1) % order.length];
      try { localStorage.setItem('dormouse-theme', next); } catch {}
      apply();
    });
  });

  /* ------------------------------------------------------------ hero shader */
  // The photo, printed: a halftone whose dot grid slides under the pointer.
  // One texture, one pass; the dots are the picture, not an overlay on it.
  const cv = document.getElementById('heroCanvas');
  if (cv) {
    const gl = cv.getContext('webgl', { antialias: false, alpha: false, premultipliedAlpha: false });
    if (!gl) { cv.style.display = 'none'; }
    else {
      const vs = 'attribute vec2 p; void main(){ gl_Position = vec4(p,0.0,1.0); }';
      const fs = `precision highp float;
uniform vec2 u_res; uniform vec2 u_mouse; uniform float u_time; uniform float u_theme;
uniform sampler2D u_tex; uniform vec2 u_img;
void main(){
  vec2 uv = gl_FragCoord.xy / u_res;
  // cover-fit the photo
  float ra = u_res.x / u_res.y, ia = u_img.x / u_img.y;
  vec2 sc = ra > ia ? vec2(1.0, ia / ra) : vec2(ra / ia, 1.0);
  vec2 tuv = (uv - 0.5) * sc + 0.5;
  // the pointer slides the dot grid, and the texture with it a little
  vec2 par = (u_mouse - 0.5) * vec2(0.035, 0.025);
  vec3 c = texture2D(u_tex, clamp(tuv + par * 0.6, 0.001, 0.999)).rgb;
  float l = dot(c, vec3(0.299, 0.587, 0.114));

  vec2 gp = (uv + par) * vec2(300.0, 300.0 / (u_res.x / u_res.y));
  vec2 f = fract(gp) - 0.5;
  float r = 0.08 + 0.42 * pow(clamp(l, 0.0, 1.0), 0.7);
  float dots = smoothstep(r, r - 0.05, length(f));

  vec3 paper = mix(vec3(0.965, 0.953, 0.925), vec3(0.039, 0.043, 0.051), u_theme);
  vec3 ink   = mix(vec3(0.10, 0.11, 0.13), vec3(0.93, 0.92, 0.90), u_theme);
  vec3 photo = mix(c, vec3(l), 0.25);
  vec3 wash  = mix(photo * 1.02 + 0.32, photo * 0.55, u_theme);
  vec3 col = mix(wash, ink, dots * 0.92);
  // a whisper of paper grain so the flat areas are not flat
  col += (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233))) * 43758.5453) - 0.5) * 0.015;
  gl_FragColor = vec4(col, 1.0);
}`;
      const sh = (t, src) => { const o = gl.createShader(t); gl.shaderSource(o, src); gl.compileShader(o); return o; };
      const prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        cv.style.display = 'none';
      } else {
        gl.useProgram(prog);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(prog, 'p');
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

        const uRes = gl.getUniformLocation(prog, 'u_res');
        const uMouse = gl.getUniformLocation(prog, 'u_mouse');
        const uTime = gl.getUniformLocation(prog, 'u_time');
        const uTheme = gl.getUniformLocation(prog, 'u_theme');
        const uImg = gl.getUniformLocation(prog, 'u_img');
        const uTex = gl.getUniformLocation(prog, 'u_tex');
        gl.uniform1i(uTex, 0);

        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([16,18,22,255]));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.uniform2f(uImg, 1600, 900);

        const img = new Image();
        img.onload = () => {
          try {
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
            gl.uniform2f(uImg, img.naturalWidth, img.naturalHeight);
          } catch (e) {
            // file:// images are cross-origin to the canvas: no texture, no
            // shader. Hide the canvas so the CSS photo shows instead of grey.
            cv.style.display = 'none';
          }
        };
        img.onerror = () => { cv.style.display = 'none'; };
        img.src = 'media/hero.jpg';

        let mx = 0.5, my = 0.5, tx = 0.5, ty = 0.5;
        const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
        addEventListener('pointermove', (e) => {
          const r = cv.getBoundingClientRect();
          tx = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
          ty = Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1);
        }, { passive: true });

        const resize = () => {
          const dpr = Math.min(devicePixelRatio || 1, 1.6);
          const w = Math.max(1, Math.floor(cv.clientWidth * dpr));
          const h = Math.max(1, Math.floor(cv.clientHeight * dpr));
          if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
          gl.viewport(0, 0, w, h);
          gl.uniform2f(uRes, w, h);
        };
        const draw = (t) => {
          resize();
          mx += (tx - mx) * 0.06; my += (ty - my) * 0.06;
          gl.uniform2f(uMouse, mx, my);
          gl.uniform1f(uTime, t);
          gl.uniform1f(uTheme, html.dataset.theme === 'dark' ? 1 : 0);
          gl.drawArrays(gl.TRIANGLES, 0, 3);
        };
        draw(0);
        if (!reduced) {
          const loop = (now) => { draw(now / 1000); requestAnimationFrame(loop); };
          requestAnimationFrame(loop);
        } else {
          addEventListener('pointermove', () => draw(0), { passive: true });
        }
        addEventListener('resize', () => draw(0));
      }
    }
  }

  // NOTE: this element's id must never collide with a section anchor. It did
  // twice today (id="bytes" on a section and a <pre>, id="data" on a section and
  // this script): getElementById returns the FIRST match, JSON.parse throws on
  // the section's text, and the whole behaviour script dies silently.
  const d = JSON.parse(document.getElementById('data').textContent);
  const fmt = (x, n = 1) => Number(x).toFixed(n);

  /* ------------------------------------------------------------- the rungs */
  const rungs = d.rungs || [];
  const control = rungs.find(r => r.tag === 'muon_fp32');
  const safe = rungs.filter(r => r.safe && r.tag !== 'muon_fp32');
  const bestSafe = safe.reduce((a, b) => (!a || b.kbs > a.kbs ? b : a), null);
  const speedup = control && bestSafe ? bestSafe.kbs / control.kbs : null;

  const rungsBody = document.getElementById('rungs');
  if (rungsBody) rungsBody.innerHTML = rungs.map(r => {
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

  const put = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };
  if (speedup) put('sSpeed', fmt(speedup, 2) + '×');
  put('sRuns', d.stats.runs_on_record);
  put('sRunsSub', d.stats.runs_with_bpb + ' with a held-out BPB');

  /* ------------------------------------------------------------- the curve */
  const pts = d.curve || [];
  if (pts.length) {
    const last = pts[pts.length - 1];
    const sBpb = document.getElementById('sBpb');
    if (sBpb) sBpb.innerHTML = fmt(last.bpb, 2);
    const cf = document.getElementById('curveFinal');
    if (cf) cf.textContent = fmt(last.bpb, 3);
    const cm = document.getElementById('curveMeta');
    if (cm) cm.textContent = `${pts.length} held-out points over ${fmt(last.mb, 0)} MB of training bytes.`;
  }

  /* ------------------------------------------------------------- the runs */
  const runsBody = document.getElementById('runsBody');
  if (runsBody) runsBody.innerHTML = (d.runs || []).map(r => `<tr>
    <td>${r.run}</td><td>${r.source || '—'}</td><td>${r.opt || '—'}</td>
    <td>${r.batch}×${r.seq}</td><td>${fmt(r.mb, 0)}</td><td class="ok">${fmt(r.bpb, 3)}</td>
  </tr>`).join('');

  /* --------------------------------------------------------- the portrait */
  const sj = JSON.parse(document.getElementById('surprise').textContent || 'null');
  if (sj && sj.bits && sj.text) {
    const cols = sj.cols || 64;
    const bits = sj.bits;
    const band = (b) => b < 0.6 ? 0 : b < 1.6 ? 1 : b < 2.6 ? 2 : b < 4.2 ? 3 : 4;
    // classes, not inline colours: the ramp is a CSS variable per theme,
    // so a theme switch repaints the strip without a re-render.
    const cls = ['b0', 'b1', 'b2', 'b3', 'b4'];
    const esc = (c) => c === '&' ? '&amp;' : c === '<' ? '&lt;' : c === '>' ? '&gt;' : c;
    let out = '';
    for (let i = 0; i < bits.length; i++) {
      out += `<b class="${cls[band(bits[i])]}">${esc(sj.text[i])}</b>`;
      if ((i + 1) % cols === 0) out += '\n';
    }
    document.getElementById('byteStrip').innerHTML = out.trimEnd();
    const mean = bits.reduce((a, b) => a + b, 0) / bits.length;
    document.getElementById('sMean').textContent = `mean ${mean.toFixed(3)} bits / byte`;
    document.getElementById('sMeta').textContent = `off=${sj.offset}`;
  }

  document.getElementById('built').textContent = `${d.stats.runs_on_record} log files`;

  /* ------------------------------------------------------------- motion */
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduce) {
    const items = document.querySelectorAll('.chapter, #curve > .wrap, #next > .wrap');
    items.forEach((el) => el.classList.add('reveal'));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -12% 0px', threshold: 0.12 });
    items.forEach((el) => io.observe(el));
    // Progressive enhancement must not hide content: an element already in (or
    // above) the viewport reveals at once, and a 2 s fallback reveals the rest —
    // a screenshot, a crawler or a missed callback can never leave a blank page.
    const revealNow = () => items.forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.top < innerHeight * 0.95) el.classList.add('in');
    });
    revealNow();
    setTimeout(() => items.forEach((el) => el.classList.add('in')), 2000);

    // count-up for the plain integers: 95M, 12, 256, 1, 16, 8h
    const counters = document.querySelectorAll('.band .v, .stat .v');
    const seen = new WeakSet();
    const run = (el) => {
      const raw = el.textContent.trim();
      const m = raw.match(/^(\d+)(.*)$/);
      if (!m) return;
      const target = +m[1];
      const suffix = m[2];
      const t0 = performance.now();
      const tick = (t) => {
        const k = Math.min(1, (t - t0) / 900);
        const eased = 1 - Math.pow(1 - k, 3);
        el.textContent = Math.round(target * eased) + suffix;
        if (k < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    };
    const io2 = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting && !seen.has(e.target)) { seen.add(e.target); run(e.target); } });
    }, { threshold: 0.5 });
    counters.forEach((el) => { if (/^\d+$/.test(el.textContent.trim())) io2.observe(el); });

    // the progress line
    const prog = document.getElementById('prog');
    const paintProg = () => {
      if (!prog) return;
      const h = document.documentElement.scrollHeight - innerHeight;
      prog.style.width = `${h > 0 ? Math.min(100, (scrollY / h) * 100) : 0}%`;
    };
    addEventListener('scroll', paintProg, { passive: true });
    addEventListener('resize', paintProg);
    paintProg();

    // data bars fill when they arrive
    const barIo = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); barIo.unobserve(e.target); } });
    }, { threshold: 0.4 });
    document.querySelectorAll('.bar').forEach((b) => { b.classList.remove('in'); barIo.observe(b); });

    // stagger the cards in a band
    document.querySelectorAll('.band .cell').forEach((el, i) => {
      el.style.transitionDelay = `${i * 70}ms`;
    });

    // the hero drifts a little as you scroll: 0.18x, capped
    const hero = document.querySelector('header');
    let raf = 0;
    const drift = () => {
      raf = 0;
      const y = Math.min(window.scrollY, 900);
      hero.style.setProperty('--drift', `${y * 0.18}px`);
    };
    addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(drift); }, { passive: true });
  }
})();
