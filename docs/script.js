const copyButton = document.querySelector('#copy-citation');
if (navigator.clipboard && window.isSecureContext) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    const status = document.querySelector('#copy-status');
    try {
      await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent.trim());
      status.textContent = 'BibTeX copied.';
    } catch {
      status.textContent = 'Copy unavailable. Please select and copy the citation above.';
    }
  });
}

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

// Official OGBench dataset clips; the original photographs remain underneath.
function setupEnvironmentDemos() {
  const videos = [...document.querySelectorAll('.environment-gallery video')];
  const button = document.querySelector('#toggle-environments');
  const visible = new Set();
  let playing = !reducedMotion.matches;

  function sync() {
    button.textContent = playing ? 'Pause demos' : 'Play demos';
    for (const video of videos) {
      if (playing && visible.has(video) && !document.hidden && !video.dataset.failed) {
        if (!video.hasAttribute('src')) video.src = video.dataset.src;
        if (video.paused) {
          video.play().catch(error => {
            if (error.name === 'NotAllowedError') {
              playing = false;
              sync();
            }
          });
        }
      } else {
        video.pause();
      }
    }
  }

  for (const video of videos) {
    video.addEventListener('playing', () => video.classList.add('is-ready'));
    video.addEventListener('error', () => {
      video.dataset.failed = 'true';
      video.classList.remove('is-ready');
    });
  }
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) visible.add(entry.target);
      else visible.delete(entry.target);
    }
    sync();
  }, { threshold: 0.1 });
  videos.forEach(video => observer.observe(video));
  button.hidden = false;
  button.addEventListener('click', () => { playing = !playing; sync(); });
  document.addEventListener('visibilitychange', sync);
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) { playing = false; sync(); }
  });
  sync();
}

// Reveal the PDF's numbered subgoals. Equal step timing is illustrative only.
function setupPlanningAnimation() {
  const section = document.querySelector('.planning');
  const links = [...section.querySelectorAll('[data-planning-src]')];
  const toggle = document.querySelector('#toggle-planning');
  let figures = [];
  let loading = false;
  let visible = false;
  let playing = !reducedMotion.matches;
  let elapsed = 0;
  let lastTime = null;
  let frame = null;
  let renderedStep = null;
  const stepMs = 240;
  let cycleMs = 0;

  function showStep(step) {
    if (step === renderedStep) return;
    renderedStep = step;
    for (const { svg, layers, count } of figures) {
      svg.dataset.currentStep = String(Math.min(step, count - 1));
      for (const layer of layers) {
        layer.style.visibility = Number(layer.dataset.step) <= step ? 'visible' : 'hidden';
      }
    }
  }

  function tick(time) {
    if (lastTime !== null) elapsed = (elapsed + time - lastTime) % cycleMs;
    lastTime = time;
    showStep(Math.floor(elapsed / stepMs));
    frame = requestAnimationFrame(tick);
  }

  function sync() {
    toggle.textContent = playing ? 'Pause' : 'Play';
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    lastTime = null;
    if (playing && visible && !document.hidden && figures.length) {
      frame = requestAnimationFrame(tick);
    }
  }

  async function load() {
    if (loading) return;
    loading = true;
    try {
      const svgs = await Promise.all(links.map(async link => {
        const response = await fetch(link.dataset.planningSrc);
        if (!response.ok) throw new Error('Planning figure unavailable');
        const doc = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
        const svg = doc.documentElement;
        if (svg.localName !== 'svg' || !svg.dataset.subgoalCount) throw new Error('Invalid planning figure');
        return document.importNode(svg, true);
      }));
      figures = svgs.map(svg => ({ svg, layers: [...svg.querySelectorAll('[data-step]')],
                                  count: Number(svg.dataset.subgoalCount) }));
      renderedStep = null;
      cycleMs = Math.max(...figures.map(figure => figure.count)) * stepMs + 1800;
      svgs.forEach((svg, index) => links[index].replaceChildren(svg));
      document.querySelector('#planning-controls').hidden = false;
      showStep(playing ? 0 : Infinity);
      sync();
    } catch {
      // A failed SVG request leaves both original paper images available.
    }
  }

  const observer = new IntersectionObserver(entries => {
    visible = entries[0].isIntersecting;
    if (visible) load();
    sync();
  }, { threshold: 0.05 });
  observer.observe(section);
  toggle.addEventListener('click', () => { playing = !playing; sync(); });
  document.querySelector('#replay-planning').addEventListener('click', () => {
    elapsed = 0;
    playing = true;
    showStep(0);
    sync();
  });
  document.querySelector('#show-full-planning').addEventListener('click', () => {
    playing = false;
    showStep(Infinity);
    sync();
  });
  document.addEventListener('visibilitychange', sync);
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) {
      playing = false;
      showStep(Infinity);
      sync();
    }
  });
}

// Page UV/PV, scoped to this project; previews do not increment production counts.
async function loadVisitStats() {
  if (location.hostname !== 'henry0132.github.io' || !/^\/DSP\/(?:index\.html)?$/.test(location.pathname)) return;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const headers = { 'x-bsz-referer': 'https://henry0132.github.io/DSP/' };
    try {
      const identity = localStorage.getItem('bsz-id');
      if (identity) headers.Authorization = `Bearer ${identity}`;
    } catch { /* Storage may be disabled; the counter can still respond. */ }
    const response = await fetch('https://bsz.iirose.cn/api', {
      method: 'POST', headers, signal: controller.signal, credentials: 'omit'
    });
    if (!response.ok) return;
    const result = await response.json();
    if (!result.success) return;
    const { page_uv: visitors, page_pv: views } = result.data || {};
    if (!Number.isSafeInteger(visitors) || visitors < 0 || !Number.isSafeInteger(views) || views < 0) return;
    document.querySelector('#page-visitors').textContent = visitors.toLocaleString('en-US');
    document.querySelector('#page-views').textContent = views.toLocaleString('en-US');
    const identity = response.headers.get('Set-Bsz-Identity');
    if (identity) {
      try { localStorage.setItem('bsz-id', identity); } catch { /* Optional storage. */ }
    }
  } catch {
    // Leave an em dash when the external counter is unavailable; never invent a count.
  } finally {
    clearTimeout(timeout);
  }
}

if ('IntersectionObserver' in window) {
  setupEnvironmentDemos();
  setupPlanningAnimation();
}
loadVisitStats();
