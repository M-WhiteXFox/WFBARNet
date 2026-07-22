(() => {
  "use strict";

  const DURATION_SECONDS = 30;
  const app = document.querySelector("#inferenceApp");
  const scrubber = document.querySelector("#timelineScrubber");
  const currentTimeOutput = document.querySelector("#currentTime");
  const playButton = document.querySelector("#playButton");
  const stopButton = document.querySelector("#stopInference");
  const recordButton = document.querySelector("#recordButton");
  const replayButton = document.querySelector("#replayButton");
  const bookmarkButton = document.querySelector("#bookmarkButton");
  const menuButton = document.querySelector("#menuButton");
  const insightToggle = document.querySelector("#insightToggle");
  const panelClose = document.querySelector("#panelClose");
  const fullscreenButton = document.querySelector("#fullscreenButton");
  const toast = document.querySelector("#toast");
  const toolButtons = [...document.querySelectorAll(".tool-button[data-tool]")];
  const seekButtons = [...document.querySelectorAll("[data-seek]")];

  let currentSeconds = Number(scrubber.value);
  let isPlaying = false;
  let lastFrameTime = null;
  let toastTimer = null;

  const formatTime = (seconds) => {
    const normalized = Math.max(0, Math.min(DURATION_SECONDS, Math.round(seconds)));
    return `00:${String(normalized).padStart(2, "0")}`;
  };

  const setPlaybackIcon = (symbolId) => {
    const iconUse = playButton.querySelector("use");
    iconUse.setAttribute("href", symbolId);
  };

  const updateTimeline = (seconds, announce = false) => {
    currentSeconds = Math.max(0, Math.min(DURATION_SECONDS, Number(seconds)));
    const percent = (currentSeconds / DURATION_SECONDS) * 100;
    scrubber.value = String(currentSeconds);
    scrubber.style.setProperty("--progress", `${percent}%`);
    scrubber.setAttribute("aria-valuetext", `${currentSeconds.toFixed(1)} 秒，共 30 秒`);
    currentTimeOutput.textContent = formatTime(currentSeconds);

    if (announce) {
      showToast(`已定位到 ${formatTime(currentSeconds)}`);
    }
  };

  const setPlaying = (nextPlaying) => {
    isPlaying = nextPlaying;
    playButton.setAttribute("aria-pressed", String(isPlaying));
    playButton.setAttribute("aria-label", isPlaying ? "暂停" : "播放");
    setPlaybackIcon(isPlaying ? "#icon-pause" : "#icon-play");
    lastFrameTime = null;
  };

  const showToast = (message) => {
    window.clearTimeout(toastTimer);
    toast.textContent = message;
    toast.classList.add("is-visible");
    toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 1900);
  };

  const setInsightOpen = (isOpen) => {
    app.classList.toggle("is-insight-open", isOpen);
    insightToggle.setAttribute("aria-expanded", String(isOpen));
  };

  const animationFrame = (timestamp) => {
    if (isPlaying) {
      if (lastFrameTime !== null) {
        const delta = (timestamp - lastFrameTime) / 1000;
        updateTimeline(currentSeconds + delta);
        if (currentSeconds >= DURATION_SECONDS) {
          setPlaying(false);
        }
      }
      lastFrameTime = timestamp;
    }
    window.requestAnimationFrame(animationFrame);
  };

  playButton.addEventListener("click", () => {
    if (!isPlaying && currentSeconds >= DURATION_SECONDS) {
      updateTimeline(0);
    }
    setPlaying(!isPlaying);
  });

  scrubber.addEventListener("input", (event) => {
    updateTimeline(event.target.value);
  });

  replayButton.addEventListener("click", () => {
    updateTimeline(currentSeconds - 5);
    showToast("已回看 5 秒");
  });

  bookmarkButton.addEventListener("click", () => {
    const isBookmarked = bookmarkButton.getAttribute("aria-pressed") === "true";
    bookmarkButton.setAttribute("aria-pressed", String(!isBookmarked));
    bookmarkButton.querySelector("span").textContent = isBookmarked ? "标记片段" : "已标记";
    showToast(isBookmarked ? "已取消片段标记" : `已标记 ${formatTime(currentSeconds)} 附近片段`);
  });

  stopButton.addEventListener("click", () => {
    const isPaused = app.classList.toggle("is-paused");
    stopButton.querySelector(".stop-label").textContent = isPaused ? "继续推理" : "停止推理";
    stopButton.setAttribute("aria-pressed", String(isPaused));
    document.querySelector(".connection-status dt").lastChild.textContent = isPaused ? "摄像头保持连接" : "摄像头已连接";
    document.querySelector(".inference-rate dd").textContent = isPaused ? "暂停" : "32 FPS";
    showToast(isPaused ? "实时推理已暂停" : "实时推理已恢复");
  });

  recordButton.addEventListener("click", () => {
    const isRecording = recordButton.getAttribute("aria-pressed") === "true";
    recordButton.setAttribute("aria-pressed", String(!isRecording));
    recordButton.setAttribute("aria-label", isRecording ? "开始录制" : "停止录制");
    showToast(isRecording ? "录制已停止" : "录制已开始");
  });

  menuButton.addEventListener("click", () => {
    const isCollapsed = app.classList.toggle("is-toolrail-collapsed");
    menuButton.setAttribute("aria-pressed", String(isCollapsed));
    menuButton.setAttribute("aria-label", isCollapsed ? "展开工具栏" : "收起工具栏");
  });

  insightToggle.addEventListener("click", () => {
    setInsightOpen(!app.classList.contains("is-insight-open"));
  });

  panelClose.addEventListener("click", () => setInsightOpen(false));

  toolButtons.forEach((button) => {
    button.addEventListener("click", () => {
      toolButtons.forEach((item) => {
        const isCurrent = item === button;
        item.classList.toggle("is-active", isCurrent);
        item.setAttribute("aria-pressed", String(isCurrent));
      });
      showToast(`已切换至${button.dataset.tool}`);
    });
  });

  seekButtons.forEach((button) => {
    button.addEventListener("click", () => updateTimeline(button.dataset.seek, true));
  });

  fullscreenButton.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        showToast("已退出全屏");
      } else {
        await document.documentElement.requestFullscreen();
        showToast("已进入全屏");
      }
    } catch {
      showToast("当前浏览器未允许全屏");
    }
  });

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLButtonElement) {
      return;
    }

    if (event.code === "Space") {
      event.preventDefault();
      playButton.click();
    } else if (event.code === "ArrowLeft") {
      updateTimeline(currentSeconds - 1);
    } else if (event.code === "ArrowRight") {
      updateTimeline(currentSeconds + 1);
    } else if (event.key.toLowerCase() === "r") {
      replayButton.click();
    } else if (event.key.toLowerCase() === "b") {
      bookmarkButton.click();
    }
  });

  const desktopQuery = window.matchMedia("(min-width: 1081px)");
  desktopQuery.addEventListener("change", (event) => {
    if (event.matches) {
      setInsightOpen(false);
    }
  });

  updateTimeline(currentSeconds);
  window.requestAnimationFrame(animationFrame);
})();
