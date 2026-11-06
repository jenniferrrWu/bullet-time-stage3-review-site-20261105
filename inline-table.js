const videos = [...document.querySelectorAll('video[data-src]')];
const button = document.querySelector('#global-play');
const active = new Set();
const globalFps = 15;
let playing = false;
let elapsedBeforePause = 0;
let startedAt = 0;

function elapsedSeconds() {
  return (playing ? performance.now() - startedAt : elapsedBeforePause) / 1000;
}

function activate(video) {
  if (video.src) return;
  video.poster = video.dataset.poster;
  video.src = video.dataset.src;
  video.preload = 'metadata';
  video.loop = true;
}

function ready(video) {
  activate(video);
  if (video.readyState >= 1) return Promise.resolve();
  return new Promise((resolve, reject) => {
    video.addEventListener('loadedmetadata', resolve, {once: true});
    video.addEventListener('error', reject, {once: true});
    video.load();
  });
}

function align(video) {
  if (!Number.isFinite(video.duration) || video.duration <= 0) return;
  const nativeFps = Number(video.dataset.fps) || globalFps;
  const rate = globalFps / nativeFps;
  video.playbackRate = rate;
  const target = (elapsedSeconds() * rate) % video.duration;
  if (Math.abs(video.currentTime - target) > 0.3) video.currentTime = target;
}

async function playVisible(video) {
  try {
    await ready(video);
    if (!playing || !active.has(video) || document.hidden) return;
    align(video);
    await video.play();
  } catch (error) {
    console.error('视频加载或播放失败：', video.dataset.src, error);
  }
}

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const video = entry.target;
      if (entry.isIntersecting) {
        active.add(video);
        activate(video);
        if (playing) playVisible(video);
      } else {
        active.delete(video);
        video.pause();
      }
    }
  }, {rootMargin: '120px'});
  videos.forEach(video => observer.observe(video));
} else {
  const refreshVisible = () => {
    for (const video of videos) {
      const rect = video.getBoundingClientRect();
      const visible = rect.bottom >= -120 && rect.top <= innerHeight + 120 && rect.right >= -120 && rect.left <= innerWidth + 120;
      if (visible) {
        if (!active.has(video)) {
          active.add(video);
          activate(video);
          if (playing) playVisible(video);
        }
      } else {
        active.delete(video);
        video.pause();
      }
    }
  };
  addEventListener('scroll', refreshVisible, {passive: true});
  addEventListener('resize', refreshVisible);
  refreshVisible();
}

button.addEventListener('click', () => {
  if (playing) {
    elapsedBeforePause = performance.now() - startedAt;
    playing = false;
    videos.forEach(video => video.pause());
    button.textContent = '▶ 全站一起播';
    button.setAttribute('aria-pressed', 'false');
  } else {
    startedAt = performance.now() - elapsedBeforePause;
    playing = true;
    button.textContent = 'Ⅱ 暂停全部';
    button.setAttribute('aria-pressed', 'true');
    active.forEach(playVisible);
  }
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) active.forEach(video => video.pause());
  else if (playing) active.forEach(playVisible);
});

setInterval(() => {
  if (playing && !document.hidden) active.forEach(video => {
    if (video.readyState >= 1) align(video);
  });
}, 2000);
