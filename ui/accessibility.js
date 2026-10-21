// Streamlit 1.55 adapter. Trusted local code only; no input or model data is read.
// Native semantics are supplemented, never removed or hidden from accessibility tools.
(() => {
  if (window.__creditScoreA11y) { window.__creditScoreA11y.refresh(); return; }
  const set = (element, name, value) => {
    if (element && element.getAttribute(name) !== value) element.setAttribute(name, value);
  };
  let scheduled = false;
  let lastFocus = '';
  const refresh = () => {
    const main = document.querySelector('[data-testid="stMain"]');
    set(main, 'role', 'main');
    set(main, 'aria-label', 'Simulator kategori kredit');
    set(document.documentElement, 'lang', 'id');
    for (const id of ['isi-profil', 'hasil-prediksi', 'evaluasi-model']) {
      set(document.getElementById(id), 'tabindex', '-1');
    }
    document.querySelectorAll('[data-testid="stTextInput"]').forEach(widget => {
      const input = widget.querySelector('input');
      const label = input?.getAttribute('aria-label') || 'nilai';
      set(input, 'inputmode', label === 'Usia (tahun)' ? 'numeric' : 'decimal');
      if (label === 'Usia (tahun)') set(input, 'aria-required', 'true');
    });
    const request = document.querySelector('[data-focus-request]');
    if (request && request.dataset.focusRequest !== lastFocus) {
      const target = document.getElementById(request.dataset.focusTarget);
      if (target) {
        lastFocus = request.dataset.focusRequest;
        set(target, 'tabindex', '-1');
        target.focus({preventScroll: true});
        target.scrollIntoView({block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
      }
    }
  };
  const observer = new MutationObserver(() => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { scheduled = false; refresh(); });
  });
  observer.observe(document.body, {childList: true, subtree: true});
  window.__creditScoreA11y = {refresh};
  refresh();
})();
