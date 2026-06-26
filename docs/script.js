/* Reveal-on-scroll + tap-to-enlarge lightbox. No slide/deck behaviour. */
(function () {
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // fade sections in as they enter the viewport
  var reveals = document.querySelectorAll('.reveal');
  if (reduce || !('IntersectionObserver' in window)) {
    reveals.forEach(function (e) { e.classList.add('in'); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { threshold: 0.12 });
    reveals.forEach(function (e) { io.observe(e); });
  }

  // lightbox: click any figure to see it full-size
  var lb = document.getElementById('lb');
  var lbImg = document.getElementById('lbImg');
  function openLb(src, alt) {
    lbImg.src = src; lbImg.alt = alt || '';
    lb.classList.add('open'); lb.setAttribute('aria-hidden', 'false');
  }
  function closeLb() {
    lb.classList.remove('open'); lb.setAttribute('aria-hidden', 'true'); lbImg.src = '';
  }
  document.querySelectorAll('button.imglink').forEach(function (b) {
    b.addEventListener('click', function () {
      var img = b.querySelector('img');
      openLb(b.getAttribute('data-full'), img ? img.alt : '');
    });
  });
  if (lb) {
    lb.addEventListener('click', closeLb);
    document.getElementById('lbClose').addEventListener('click', closeLb);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && lb.classList.contains('open')) closeLb();
    });
  }
})();
