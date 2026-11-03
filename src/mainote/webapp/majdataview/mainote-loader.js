/*
 * mainote <-> MajdataView WebGL bridge.
 *
 * The player is the unmodified MajdataView WebGL build (GPL-3.0).  This file
 * only feeds maidata.txt / track.mp3 / background into the upload input that
 * MajdataView itself creates, so the React UI can open any local chart without
 * the three manual file pickers.
 *
 * Query parameters:
 *   maidata=<url>   compiled maidata.txt (required)
 *   audio=<url>      mp3 track, or a server-generated silent track (required)
 *   bg=<url>         optional background image
 *   autoload=0       wait for window.__mainoteStart() instead of starting
 *
 * The chart is compiled into the first difficulty slot (EASY) by the backend,
 * because this build's level selector defaults to index 0.
 */
(function () {
  "use strict";

  var params = new URLSearchParams(window.location.search);
  var wanted = {
    chart: params.get("maidata"),
    audio: params.get("audio"),
    bg: params.get("bg")
  };
  var autoload = params.get("autoload") !== "0";
  var title = params.get("title") || "MajdataView";

  function finiteNumber(value, fallback) {
    var number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function parseCharts(value) {
    try {
      var items = JSON.parse(value || "[]");
      if (!Array.isArray(items)) return [];
      return items.filter(function (item) {
        return item && Number.isFinite(Number(item.index));
      });
    } catch (error) {
      return [];
    }
  }

  var currentIndex = Math.trunc(finiteNumber(params.get("index"), 1));
  var timeline = Math.max(1, finiteNumber(params.get("duration"), 60));
  var initialSeek = Math.min(
    Math.max(0, finiteNumber(params.get("seek"), 0)),
    Math.max(0, timeline - 0.1)
  );
  var levelOptions = parseCharts(params.get("charts"));
  var playback = {
    base: initialSeek,
    playing: false,
    startedAt: 0,
    duration: timeline,
    ready: false,
    canStart: false,
    scrubbing: false
  };

  var state = {
    files: { chart: null, audio: null, bg: null },
    fed: { chart: 0, audio: 0, bg: 0 },
    requested: { chart: 0, audio: 0, bg: 0 },
    pending: { chart: 0, audio: 0, bg: 0 },
    inputs: { chart: null, audio: null, bg: null },
    loading: false,
    failed: false,
    unityReady: false,
    started: false,
    autoAdvance: false
  };

  function statusNode() {
    return document.getElementById("mv-status");
  }

  function setStatus(message, sticky) {
    var node = statusNode();
    if (!node) return;
    node.textContent = message || "";
    node.style.display = message ? "block" : "none";
    if (sticky) node.dataset.sticky = "1";
    else delete node.dataset.sticky;
  }

  function notifyParent(type, message, extra) {
    try {
      if (window.parent && window.parent !== window) {
        var payload = { type: type, message: message || "" };
        if (extra) {
          Object.keys(extra).forEach(function (key) {
            payload[key] = extra[key];
          });
        }
        window.parent.postMessage(payload, "*");
      }
    } catch (error) {
      /* cross-origin parents cannot be notified; ignore */
    }
  }

  function embedded() {
    return window.parent && window.parent !== window;
  }

  function fail(message) {
    state.failed = true;
    state.loading = false;
    setStatus("MajdataView 加载失败：" + message, true);
    notifyParent("mainote:majdataview-error", message);
  }

  /* ------------------------------------------------------------------ */
  /* file input interception                                             */
  /* ------------------------------------------------------------------ */

  function stageFromAccept(accept) {
    var value = (accept || "").toLowerCase();
    if (value.indexOf("txt") >= 0) return "chart";
    if (value.indexOf("mp3") >= 0 || value.indexOf("mpeg") >= 0 || value.indexOf("audio") >= 0) {
      return "audio";
    }
    if (
      value.indexOf("jpg") >= 0 ||
      value.indexOf("jpeg") >= 0 ||
      value.indexOf("png") >= 0 ||
      value.indexOf("image") >= 0
    ) {
      return "bg";
    }
    return null;
  }

  function nextStageByOrder() {
    var total = state.requested.chart + state.requested.audio + state.requested.bg;
    return ["chart", "audio", "bg"][Math.min(total, 2)];
  }

  function feed(stage, input) {
    var file = state.files[stage];
    input = input || state.inputs[stage];
    if (!file || !input) return false;
    try {
      var transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
    } catch (error) {
      fail("无法把文件注入播放器: " + error);
      return false;
    }
    state.fed[stage] += 1;
    state.pending[stage] = 0;
    var target = input;
    window.setTimeout(function () {
      target.dispatchEvent(new Event("change", { bubbles: true }));
    }, 0);
    updateStatus();
    if (stage === "bg") {
      state.loading = false;
      window.setTimeout(function () {
        if (!statusNode() || !statusNode().dataset.sticky) setStatus("");
      }, 800);
      notifyParent("mainote:majdataview-ready");
      window.__mainoteReady = true;
      playback.ready = true;
      refreshUI();
      if (initialSeek > 0) {
        setStatus(
          "已定位到 " + formatTime(initialSeek) + "，点击播放继续",
          true
        );
      }
      // The game switches to "ready" mode after decoding the background; give
      // it a moment before forwarding TogglePlay from the control bar.
      window.setTimeout(function () {
        playback.canStart = true;
        refreshUI();
      }, 1400);
    }
    return true;
  }

  function flushPending() {
    ["chart", "audio", "bg"].forEach(function (stage) {
      if (state.pending[stage] > 0 && state.files[stage]) {
        feed(stage);
      }
    });
  }

  /*
   * The build only opens its uploaded-file pickers from inside the game
   * ("Upload" button -> GameMain.ChickLoad).  Calling ChickLoad ourselves
   * advances the same three stages; each stage opens the hidden input which
   * the click interceptor above fills with a prefetched file.  One call
   * advances one stage, so keep nudging until the background was accepted.
   */
  function sendChickLoad() {
    try {
      if (window.__majdataUnity && window.__majdataUnity.SendMessage) {
        window.__majdataUnity.SendMessage("GameMain", "ChickLoad");
        return true;
      }
    } catch (error) {
      /* Unity may still be booting; retry on the next tick */
    }
    return false;
  }

  function autoAdvance() {
    if (state.autoAdvance || !autoload) return;
    state.autoAdvance = true;
    var attempts = 0;
    var lastDone = -1;
    function tick() {
      if (state.failed || state.fed.bg > 0) {
        state.autoAdvance = false;
        return;
      }
      var done = state.fed.chart + state.fed.audio + state.fed.bg;
      if (done !== lastDone) {
        lastDone = done;
        attempts = Math.max(0, attempts - 2);
      }
      if (attempts >= 20) {
        state.autoAdvance = false;
        fail("播放器没有接受完整文件(谱面/音频/背景)");
        return;
      }
      sendChickLoad();
      attempts += 1;
      window.setTimeout(tick, 2200);
    }
    tick();
  }

  var originalClick = HTMLInputElement.prototype.click;
  HTMLInputElement.prototype.click = function () {
    if (this.id !== "fileuploader") {
      return originalClick.apply(this, arguments);
    }
    var stage = stageFromAccept(this.getAttribute("accept")) || nextStageByOrder();
    state.requested[stage] += 1;
    state.inputs[stage] = this;

    if (state.files[stage]) {
      feed(stage, this);
    } else {
      state.pending[stage] += 1;
      state.loading = true;
      setStatus("正在准备" + stageLabel(stage) + "…");
      updateStatus();
    }
    // Never open the OS file picker: the bridge supplies the file instead.
    return undefined;
  };

  function stageLabel(stage) {
    if (stage === "chart") return "谱面";
    if (stage === "audio") return "音频";
    return "背景";
  }

  function updateStatus() {
    if (state.failed) return;
    var stages = ["chart", "audio", "bg"];
    var done = stages.filter(function (stage) {
      return state.fed[stage] > 0;
    }).length;
    if (done === 0) {
      setStatus(state.loading ? "正在准备谱面与音频…" : "");
      return;
    }
    if (done < 3) {
      setStatus("正在载入 MajdataView（" + done + "/3）…");
    }
  }

  /* ------------------------------------------------------------------ */
  /* playback controls                                                   */
  /* ------------------------------------------------------------------ */

  function formatTime(seconds) {
    var total = Math.max(0, Math.floor(seconds || 0));
    var hours = Math.floor(total / 3600);
    var minutes = Math.floor((total % 3600) / 60);
    var rest = total % 60;
    var body =
      String(minutes).padStart(2, "0") + ":" + String(rest).padStart(2, "0");
    return hours > 0 ? String(hours) + ":" + body : body;
  }

  function currentTime() {
    if (!playback.playing) return playback.base;
    var elapsed = (performance.now() - playback.startedAt) / 1000;
    return Math.min(playback.duration, playback.base + elapsed);
  }

  function sendGameCommand(method) {
    try {
      if (window.__majdataUnity && window.__majdataUnity.SendMessage) {
        window.__majdataUnity.SendMessage("GameMain", method);
        return true;
      }
    } catch (error) {
      /* the build may still be booting */
    }
    return false;
  }

  function refreshUI() {
    var button = document.getElementById("mv-play");
    var time = document.getElementById("mv-time");
    var slider = document.getElementById("mv-seek");
    if (!button || !time || !slider) return;
    var now = currentTime();
    var shown = playback.scrubbing ? Number(slider.value) || 0 : now;
    button.disabled = !(playback.ready && playback.canStart);
    button.textContent = playback.playing ? "❚❚" : "▶";
    button.setAttribute("aria-label", playback.playing ? "暂停" : "播放");
    time.textContent =
      formatTime(shown) + " / " + formatTime(playback.duration);
    slider.max = String(playback.duration);
    if (!playback.scrubbing) slider.value = String(now);
  }

  function tickUI() {
    if (playback.playing) {
      var now = currentTime();
      if (now >= playback.duration) {
        playback.playing = false;
        playback.base = playback.duration;
      }
    }
    refreshUI();
  }

  function togglePlayback() {
    if (!playback.ready || !playback.canStart) return;
    var now = currentTime();
    if (playback.playing) {
      if (!sendGameCommand("TogglePlay")) return;
      playback.base = now;
      playback.playing = false;
    } else {
      if (now >= playback.duration - 0.15) {
        requestSeek(0);
        return;
      }
      if (!sendGameCommand("TogglePlay")) {
        setStatus("播放器尚未就绪，请稍后重试", true);
        return;
      }
      playback.base = now;
      playback.startedAt = performance.now();
      playback.playing = true;
      setStatus("");
    }
    refreshUI();
  }

  function requestSeek(seconds) {
    var target = Math.min(
      Math.max(0, Number(seconds) || 0),
      playback.duration
    );
    playback.playing = false;
    playback.base = target;
    refreshUI();
    if (embedded()) {
      notifyParent("mainote:majdataview-seek", "", { seconds: target });
    } else {
      reloadAtSeek(target);
    }
  }

  function chooseDifficulty(index) {
    var value = Number(index);
    if (!Number.isFinite(value) || value === currentIndex) return;
    playback.playing = false;
    refreshUI();
    if (embedded()) {
      notifyParent("mainote:majdataview-difficulty", "", { index: value });
    } else {
      reloadAtDifficulty(value);
    }
  }

  function rewriteUrl(raw, update) {
    if (!raw) return raw;
    var url = new URL(raw, window.location.href);
    update(url.searchParams);
    return url.pathname + "?" + url.searchParams.toString();
  }

  function replaceSearch(update) {
    var next = new URLSearchParams(window.location.search);
    update(next);
    window.location.replace(
      window.location.pathname + "?" + next.toString()
    );
  }

  function reloadAtDifficulty(index) {
    replaceSearch(function (query) {
      query.set("index", String(index));
      query.set("seek", "0");
      var chart = query.get("maidata");
      if (chart) {
        query.set(
          "maidata",
          rewriteUrl(chart, function (chartQuery) {
            chartQuery.set("index", String(index));
          })
        );
      }
    });
  }

  function reloadAtSeek(seconds) {
    replaceSearch(function (query) {
      query.set("seek", String(seconds));
      var chart = query.get("maidata");
      if (chart) {
        query.set(
          "maidata",
          rewriteUrl(chart, function (chartQuery) {
            if (seconds > 0) chartQuery.set("seek", String(seconds));
            else chartQuery.delete("seek");
          })
        );
      }
      var audio = query.get("audio");
      if (audio) {
        query.set(
          "audio",
          rewriteUrl(audio, function (audioQuery) {
            var remaining = Math.max(
              1,
              Math.ceil(playback.duration - seconds)
            );
            audioQuery.set("duration", String(remaining));
            if (audioQuery.has("root") || audioQuery.has("path")) {
              audioQuery.set("start", String(seconds));
            } else {
              audioQuery.delete("start");
            }
          })
        );
      }
    });
  }

  function renderLevels() {
    var host = document.getElementById("mv-levels");
    if (!host) return;
    host.textContent = "";
    if (!levelOptions.length) {
      host.hidden = true;
      return;
    }
    host.hidden = false;
    levelOptions.forEach(function (item) {
      var index = Number(item.index);
      var button = document.createElement("button");
      button.type = "button";
      button.className =
        "mv-level" + (index === currentIndex ? " active" : "");
      button.dataset.index = String(index);
      var label = document.createElement("span");
      label.textContent =
        item.label ||
        String(item.difficulty || "?").slice(0, 3).toUpperCase();
      var level = document.createElement("b");
      level.textContent = String(item.level || "");
      button.appendChild(label);
      if (level.textContent) button.appendChild(level);
      if (index === currentIndex) {
        button.style.background = item.bg || "rgba(255,255,255,0.08)";
        button.style.color = item.fg || "#e8e8f2";
        button.style.borderColor = "transparent";
        button.setAttribute("aria-current", "true");
      } else if (item.bg) {
        button.style.borderColor = item.bg;
      }
      button.addEventListener("click", function () {
        chooseDifficulty(index);
      });
      host.appendChild(button);
    });
  }

  function initControls() {
    var button = document.getElementById("mv-play");
    var slider = document.getElementById("mv-seek");
    if (!button || !slider) return;
    button.addEventListener("click", togglePlayback);
    slider.addEventListener("pointerdown", function () {
      playback.scrubbing = true;
    });
    slider.addEventListener("input", function () {
      playback.scrubbing = true;
      refreshUI();
    });
    slider.addEventListener("change", function () {
      playback.scrubbing = false;
      var target = Number(slider.value);
      if (Number.isFinite(target)) requestSeek(target);
    });
    slider.max = String(playback.duration);
    slider.value = String(initialSeek);
    renderLevels();
    refreshUI();
    window.setInterval(tickUI, 200);
  }

  /* ------------------------------------------------------------------ */
  /* assets                                                              */
  /* ------------------------------------------------------------------ */

  async function fetchFile(url, name, type) {
    var response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(name + " HTTP " + response.status);
    }
    return new File([await response.blob()], name, { type: type });
  }

  function makeBackground() {
    var canvas = document.createElement("canvas");
    canvas.width = 1920;
    canvas.height = 1080;
    var context = canvas.getContext("2d");
    var gradient = context.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, "#161426");
    gradient.addColorStop(0.55, "#0d0c17");
    gradient.addColorStop(1, "#05050a");
    context.fillStyle = gradient;
    context.fillRect(0, 0, canvas.width, canvas.height);

    var halo = context.createRadialGradient(960, 480, 40, 960, 480, 620);
    halo.addColorStop(0, "rgba(176, 156, 255, 0.20)");
    halo.addColorStop(1, "rgba(176, 156, 255, 0)");
    context.fillStyle = halo;
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.fillStyle = "rgba(238, 241, 248, 0.92)";
    context.font = "600 46px system-ui, -apple-system, 'Segoe UI', sans-serif";
    context.textAlign = "center";
    context.fillText(title.slice(0, 48), 960, 522);
    context.fillStyle = "rgba(238, 241, 248, 0.38)";
    context.font = "20px system-ui, -apple-system, 'Segoe UI', sans-serif";
    context.fillText("MajdataView", 960, 566);

    return new Promise(function (resolve) {
      canvas.toBlob(function (blob) {
        resolve(new File([blob], "bg.png", { type: "image/png" }));
      }, "image/png");
    });
  }

  async function loadAssets() {
    if (!wanted.chart) {
      fail("缺少谱面地址");
      return;
    }
    if (!wanted.audio) {
      fail("缺少音频地址");
      return;
    }
    try {
      state.loading = true;
      updateStatus();
      var results = await Promise.all([
        fetchFile(wanted.chart, "maidata.txt", "text/plain;charset=utf-8"),
        fetchFile(wanted.audio, "track.mp3", "audio/mpeg")
      ]);
      state.files.chart = results[0];
      state.files.audio = results[1];
      if (wanted.bg) {
        try {
          state.files.bg = await fetchFile(wanted.bg, "bg.png", "image/png");
        } catch (error) {
          state.files.bg = await makeBackground();
        }
      } else {
        state.files.bg = await makeBackground();
      }
      updateStatus();
      flushPending();
      maybeAutoAdvance();
    } catch (error) {
      fail(error && error.message ? error.message : String(error));
    }
  }

  function maybeAutoAdvance() {
    if (state.unityReady && state.files.chart && state.files.audio && state.files.bg) {
      autoAdvance();
    }
  }

  /* ------------------------------------------------------------------ */
  /* lifecycle                                                           */
  /* ------------------------------------------------------------------ */

  window.__mainoteStart = function () {
    if (state.started) return;
    state.started = true;
    loadAssets();
  };

  window.__mainoteUnityFailed = function (message) {
    fail(message || "Unity 实例创建失败");
  };

  window.addEventListener("mainote:unity-ready", function () {
    state.unityReady = true;
    if (autoload) {
      window.__mainoteStart();
      maybeAutoAdvance();
    }
  });

  if (autoload) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () {
        window.__mainoteStart();
      });
    } else {
      window.__mainoteStart();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initControls);
  } else {
    initControls();
  }

  // Debug hooks used by the webapp test suite.
  window.__mainoteState = state;
  window.__mainotePlayback = playback;
})();
