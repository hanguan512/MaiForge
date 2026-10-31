(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    options: null,
    optionsLoaded: false,
    job: null,
    resultLoadedFor: null,
    resultFailedFor: null,
    selectedRun: null,
    runDetail: null,
    pollBusy: false,
    pollErrorShown: false,
    logAuto: true,
    lastJobStatus: null,
  };

  const STATUS_TEXT = {
    idle: "IDLE",
    running: "RUNNING",
    stopping: "STOPPING",
    completed: "DONE",
    failed: "FAILED",
    stopped: "STOPPED",
  };

  const STATUS_CLASS = {
    idle: "idle",
    running: "running",
    stopping: "stopping",
    completed: "completed",
    failed: "failed",
    stopped: "stopped",
  };

  const TYPE_LABELS = {
    tap: "tap",
    hold: "hold",
    slide: "slide",
    touch: "touch",
    touch_hold: "touch hold",
  };

  const FILE_LABELS = {
    "maidata.txt": "maidata.txt",
    "p3_ir.json": "IR JSON",
    "p3_summary.json": "summary",
    "timing.json": "timing",
    "p3_validation.json": "validation",
    "metrics.jsonl": "metrics.jsonl",
    "config.json": "config.json",
  };

  // ---------------------------------------------------------------- helpers
  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]));
  }

  function setText(id, value) {
    const node = $(id);
    if (node) node.textContent = value == null || value === "" ? "—" : String(value);
  }

  function setHtml(id, value) {
    const node = $(id);
    if (node) node.innerHTML = value;
  }

  function formatNumber(value, digits = 3) {
    if (value == null || value === "" || Number.isNaN(Number(value))) return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (number !== 0 && Math.abs(number) < 0.001) return number.toExponential(2);
    return Number(number.toFixed(digits)).toString();
  }

  function formatPct(value) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function formatTime(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) return "—";
    const total = Math.max(0, Math.round(Number(seconds)));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return hours > 0
      ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
      : `${minutes}:${String(secs).padStart(2, "0")}`;
  }

  function basename(path) {
    const text = String(path || "").replace(/\\/g, "/");
    return text.slice(text.lastIndexOf("/") + 1) || text;
  }

  function dirname(path) {
    const text = String(path || "").replace(/\\/g, "/");
    const index = text.lastIndexOf("/");
    return index > 0 ? text.slice(0, index) : "";
  }

  function fileUrl(path) {
    return `/api/file?path=${encodeURIComponent(path)}`;
  }

  function downloadUrl(path) {
    return `${fileUrl(path)}&download=1`;
  }

  function audioUrl(path) {
    return `/api/audio?path=${encodeURIComponent(path)}`;
  }

  function toast(message, type = "info", timeout = 4200) {
    const host = $("toasts");
    const node = document.createElement("div");
    node.className = `toast ${type}`;
    node.textContent = message;
    node.addEventListener("click", () => node.remove());
    host.appendChild(node);
    window.setTimeout(() => {
      node.style.opacity = "0";
      node.style.transform = "translateY(6px)";
      node.style.transition = "0.2s ease";
      window.setTimeout(() => node.remove(), 220);
    }, timeout);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      cache: "no-store",
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    const text = await response.text();
    let data;
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { error: text.slice(0, 400) || response.statusText };
    }
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  function post(path, payload) {
    return api(path, { method: "POST", body: JSON.stringify(payload || {}) });
  }

  function setTab(name) {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === name);
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `tab-${name}`);
    });
    if (name === "runs") loadRuns().catch((error) => toast(error.message, "error"));
    if (name === "train" || name === "infer") {
      window.setTimeout(() => drawLossChart($("lossCanvas"), state.job?.metrics || [], false), 30);
    }
  }

  // ---------------------------------------------------------------- options
  function fillSelect(select, items, preferred = null) {
    if (!select) return;
    const current = select.value;
    select.innerHTML = "";
    for (const item of items) {
      const option = document.createElement("option");
      option.value = item.value;
      option.textContent = item.label;
      select.appendChild(option);
    }
    const values = new Set(items.map((item) => item.value));
    if (preferred != null && values.has(preferred)) {
      select.value = preferred;
    } else if (current && values.has(current)) {
      select.value = current;
    } else if (items.length) {
      select.value = items[0].value;
    } else {
      select.value = "";
    }
  }

  function renderGpu(gpu) {
    const pill = $("gpuPill");
    if (!pill) return;
    if (!gpu) {
      pill.textContent = "GPU 不可用";
      pill.className = "pill muted";
      pill.title = "nvidia-smi 不可用";
      return;
    }
    const used = gpu.used_mb / 1024;
    const total = gpu.total_mb / 1024;
    const ratio = total > 0 ? used / total : 0;
    pill.textContent = `${gpu.name.replace(/^NVIDIA GeForce\s+/i, "")} · ${used.toFixed(1)}/${total.toFixed(1)} GB · ${gpu.util}% · ${gpu.temp}°C`;
    pill.className = `pill ${ratio > 0.9 ? "failed" : ratio > 0.72 ? "stopping" : "running"}`;
    pill.title = `${gpu.name} | ${gpu.used_mb} / ${gpu.total_mb} MiB | ${gpu.util}% | ${gpu.temp}°C`;
  }

  function renderOverview(overview) {
    const data = overview || {};
    const split = data.split_song_counts || {};
    const values = $("datasetOverview")?.querySelectorAll(".stat strong") || [];
    if (values[0]) values[0].textContent = data.songs ?? "—";
    if (values[1]) values[1].textContent = data.charts ?? "—";
    if (values[2]) values[2].textContent = data.features ?? "—";
    if (values[3]) {
      values[3].textContent = [split.train, split.val, split.test]
        .map((item) => (item == null ? "—" : item))
        .join(" / ");
    }
    const pill = $("overviewPill");
    if (pill) {
      pill.textContent = `${data.songs ?? "—"} 首 · ${data.charts ?? "—"} 谱面 · ${data.features ?? "—"} 特征`;
    }
  }

  function renderAudioDatalist(items) {
    const datalist = $("audioOptions");
    if (!datalist) return;
    datalist.innerHTML = "";
    const groups = { desktop: "桌面", raw: "data/raw", dataset: "dataset" };
    for (const group of ["desktop", "raw", "dataset"]) {
      const rows = (items || []).filter((item) => item.group === group);
      for (const item of rows) {
        const option = document.createElement("option");
        option.value = item.value;
        option.label = `${groups[group]} · ${item.label}`;
        datalist.appendChild(option);
      }
    }
  }

  async function loadOptions() {
    const options = await api("/api/options");
    const first = !state.optionsLoaded;
    state.options = options;
    renderGpu(options.gpu);
    renderOverview(options.overview);

    const defaults = options.defaults || {};
    const configItems = options.configs || [];
    fillSelect($("trainConfig"), configItems, first ? defaults.config : null);
    fillSelect(
      $("trainResume"),
      [{ value: "", label: "不恢复（从新模型开始）" }, ...(options.checkpoints || [])],
      first ? "" : null
    );
    fillSelect(
      $("inferCheckpoint"),
      (options.checkpoints || []).length
        ? options.checkpoints
        : [{ value: "", label: "暂无 checkpoint" }],
      first ? defaults.checkpoint : null
    );
    fillSelect(
      $("inferTiming"),
      [{ value: "", label: "自动分析（不传 timing）" }, ...(options.timing || [])],
      first ? defaults.timing : null
    );
    fillSelect(
      $("inferFeatures"),
      [{ value: "", label: "自动计算（不传 features）" }, ...(options.features || [])],
      ""
    );
    renderAudioDatalist(options.audio || []);

    if (first) {
      if ($("trainDevice")) $("trainDevice").value = options.gpu ? "cuda" : "auto";
      if ($("inferDevice")) $("inferDevice").value = options.gpu ? "cuda" : "auto";
      if ($("trainOutput")) $("trainOutput").value = defaults.train_output || "runs/web_train";
      if ($("inferOutput")) $("inferOutput").value = defaults.infer_output || "out/web_infer";
      if ($("inferAudio")) $("inferAudio").value = defaults.audio || "";
    } else if ($("inferAudio") && !$("inferAudio").value) {
      $("inferAudio").value = defaults.audio || "";
    }

    renderRuns(options.runs || []);
    state.optionsLoaded = true;
  }

  // ---------------------------------------------------------------- monitor
  function renderJob(job) {
    const active = Boolean(job && ["running", "stopping"].includes(job.status));
    const jobPill = $("jobPill");
    if (jobPill) {
      jobPill.textContent = job ? STATUS_TEXT[job.status] || job.status.toUpperCase() : "IDLE";
      jobPill.className = `pill ${job ? STATUS_CLASS[job.status] || "muted" : "idle"}`;
    }

    const monitorStatus = $("monitorStatus");
    if (monitorStatus) {
      monitorStatus.textContent = job ? STATUS_TEXT[job.status] || job.status : "IDLE";
      monitorStatus.className = `pill ${job ? STATUS_CLASS[job.status] || "muted" : "idle"}`;
    }
    setText("monitorKind", job
      ? job.kind === "train"
        ? `训练 · ${job.extra?.config || "P3"}`
        : `推理 · ${job.extra?.difficulty || "master"} ${job.extra?.level ?? ""}`
      : "当前没有任务");

    setText("monitorElapsed", job ? formatTime(job.elapsed) : "—");
    setText("monitorOutput", job?.output_dir || "—");
    const outputNode = $("monitorOutput");
    if (outputNode) outputNode.title = job?.output_dir || "";
    setText("monitorCommand", job?.command?.join(" ") || "—");

    const metrics = job?.metrics || [];
    const last = metrics[metrics.length - 1] || {};
    const isTrain = job?.kind === "train";
    setText("mEpoch", isTrain && last.epoch != null ? Number(last.epoch) + 1 : "—");
    setText("mStep", isTrain ? formatNumber(last.step, 0) : "—");
    setText("mTrainLoss", isTrain ? formatNumber(last.train_loss) : "—");
    setText("mValLoss", isTrain ? formatNumber(last.val_loss) : "—");
    setText("mTrainAcc", isTrain ? formatPct(last.train_acc) : "—");
    setText("mValAcc", isTrain ? formatPct(last.val_acc) : "—");
    setText("mLr", isTrain ? formatNumber(last.lr, 8) : "—");
    const bestVal = metrics
      .map((row) => row.val_loss)
      .filter((value) => typeof value === "number" && Number.isFinite(value));
    setText("mBest", bestVal.length ? formatNumber(Math.min(...bestVal)) : "—");

    // progress
    let progress = 0;
    let progressText = job ? "任务运行中" : "等待任务";
    if (isTrain && job) {
      const epochs = Number(job.extra?.epochs || 0);
      const maxSteps = Number(job.extra?.max_steps || 0);
      const step = Number(last.step || 0);
      const epoch = Number(last.epoch);
      if (maxSteps && step) {
        progress = Math.min(100, (step / maxSteps) * 100);
        progressText = `step ${step}/${maxSteps} · epoch ${Number.isFinite(epoch) ? epoch + 1 : "—"}`;
      } else if (epochs && Number.isFinite(epoch)) {
        progress = Math.min(100, ((epoch + 1) / epochs) * 100);
        progressText = `epoch ${epoch + 1}/${epochs} · step ${step || "—"}`;
      } else {
        progressText = step ? `step ${step}` : "等待第一个 epoch";
      }
    } else if (job?.kind === "infer") {
      if (job.status === "completed") {
        progress = 100;
        progressText = "推理完成";
      } else if (job.status === "failed" || job.status === "stopped") {
        progressText = "推理未完成";
      } else {
        progress = 35;
        progressText = "逐窗口生成中…";
      }
    }
    const bar = $("monitorProgressBar");
    if (bar) bar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
    setText("monitorProgressPct", progress > 0 && progress < 100 ? `${Math.round(progress)}%` : progress === 100 ? "100%" : "");
    setText("monitorProgressText", progressText);

    // charts
    const canvas = $("lossCanvas");
    const empty = $("lossEmpty");
    if (isTrain && metrics.length) {
      if (empty) empty.classList.add("hidden");
      drawLossChart(canvas, metrics);
    } else {
      if (empty) {
        empty.classList.remove("hidden");
        empty.textContent = isTrain ? "等待 metrics…" : job ? "推理任务没有 loss 曲线" : "metrics 出现后显示 loss 曲线";
      }
      drawLossChart(canvas, [], false);
    }

    // log
    const logPre = $("logPre");
    if (logPre) {
      const lines = job?.log || [];
      logPre.textContent = lines.length ? lines.join("\n") : job ? "等待日志…" : "等待任务…";
      logPre.classList.toggle(
        "log-error",
        lines.some((line) => /Traceback|error:|Error:|failed/i.test(line))
      );
      if (state.logAuto) logPre.scrollTop = logPre.scrollHeight;
    }

    // buttons
    if ($("trainStart")) $("trainStart").disabled = active;
    if ($("inferStart")) $("inferStart").disabled = active;
    if ($("trainStop")) $("trainStop").disabled = !(active && job?.kind === "train");
    if ($("inferStop")) $("inferStop").disabled = !(active && job?.kind === "infer");
  }

  async function poll() {
    if (state.pollBusy || document.hidden) return;
    state.pollBusy = true;
    try {
      const snapshot = await api("/api/job");
      const previous = state.job;
      const job = snapshot.job
        ? { ...snapshot.job, metrics: snapshot.metrics || [], log: snapshot.log || [] }
        : null;
      state.job = job;
      renderJob(job);
      if (
        job &&
        job.kind === "infer" &&
        job.status === "completed" &&
        state.resultLoadedFor !== job.id &&
        state.resultFailedFor !== job.id
      ) {
        loadResult(job).catch((error) => {
          state.resultFailedFor = job.id;
          toast(error.message, "error");
        });
      }
      if (
        previous &&
        job &&
        previous.id === job.id &&
        ["running", "stopping"].includes(previous.status) &&
        ["completed", "failed", "stopped"].includes(job.status)
      ) {
        await onJobFinished(job);
      }
      state.lastJobStatus = job?.status || null;
      state.pollErrorShown = false;
    } catch (error) {
      if (!state.pollErrorShown) {
        toast(`无法连接控制服务: ${error.message}`, "error");
        state.pollErrorShown = true;
      }
    } finally {
      state.pollBusy = false;
    }
  }

  async function onJobFinished(job) {
    const label = job.kind === "train" ? "训练" : "推理";
    if (job.status === "completed") toast(`${label}完成`, "success");
    else if (job.status === "failed") toast(`${label}失败，查看实时日志`, "error", 7000);
    else toast(`${label}已停止`, "warn");
    try {
      await loadOptions();
    } catch {
      /* status refresh is best-effort */
    }
    if (job.kind === "infer" && job.status === "completed") {
      try {
        await loadResult(job);
      } catch (error) {
        toast(error.message, "error");
      }
    }
    if (job.kind === "train") {
      loadRuns().catch(() => {});
    }
  }

  // ---------------------------------------------------------------- start / stop
  async function startTrain() {
    if (state.job && ["running", "stopping"].includes(state.job.status)) {
      toast("已有任务在运行", "warn");
      return;
    }
    const button = $("trainStart");
    button.disabled = true;
    try {
      const result = await post("/api/train/start", {
        config: $("trainConfig").value,
        device: $("trainDevice").value,
        output_dir: $("trainOutput").value.trim(),
        resume: $("trainResume").value || null,
        max_steps: $("trainMaxSteps").value || null,
        num_workers: $("trainWorkers").value || null,
        dry_run: $("trainDry").checked,
      });
      state.job = result.job ? { ...result.job, metrics: [], log: [] } : null;
      renderJob(state.job);
      toast(`训练已启动：${result.job?.output_dir || ""}`, "success");
    } catch (error) {
      toast(error.message, "error", 6000);
      button.disabled = false;
    }
  }

  async function startInfer() {
    if (state.job && ["running", "stopping"].includes(state.job.status)) {
      toast("已有任务在运行", "warn");
      return;
    }
    const audio = $("inferAudio").value.trim();
    const checkpoint = $("inferCheckpoint").value;
    if (!audio) return toast("请选择音频文件", "error");
    if (!checkpoint) return toast("请选择 checkpoint", "error");
    const button = $("inferStart");
    button.disabled = true;
    try {
      const result = await post("/api/infer/start", {
        audio,
        checkpoint,
        timing: $("inferTiming").value || null,
        features: $("inferFeatures").value || null,
        difficulty: $("inferDifficulty").value,
        level: $("inferLevel").value || null,
        temperature: $("inferTemperature").value || null,
        top_k: $("inferTopK").value || null,
        max_new_tokens: $("inferMaxTokens").value || null,
        device: $("inferDevice").value,
        output_dir: $("inferOutput").value.trim(),
      });
      state.job = result.job ? { ...result.job, metrics: [], log: [] } : null;
      state.resultLoadedFor = null;
      state.resultFailedFor = null;
      $("inferResult").classList.add("hidden");
      renderJob(state.job);
      toast(`推理已启动：${result.job?.output_dir || ""}`, "success");
    } catch (error) {
      toast(error.message, "error", 6000);
      button.disabled = false;
    }
  }

  async function stopJob() {
    if (!state.job || !["running", "stopping"].includes(state.job.status)) return;
    if (!window.confirm("停止当前任务？训练会丢失未保存的当前 epoch。")) return;
    try {
      await post("/api/job/stop", {});
      toast("正在停止…", "warn");
      await poll();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  // ---------------------------------------------------------------- result
  async function loadResult(job) {
    const path = job?.extra?.result_maidata || job?.result_maidata || `${job?.output_dir || ""}/maidata.txt`;
    if (!path) throw new Error("没有找到生成结果路径");
    const chart = await api(`/api/chart?path=${encodeURIComponent(path)}`);
    state.resultLoadedFor = job.id;
    renderResult(chart);
  }

  function renderResult(chart) {
    const result = $("inferResult");
    if (!result) return;
    result.classList.remove("hidden");
    const stats = chart.stats || {};
    const validation = chart.validation || {};
    const errors = validation.errors || [];
    const warnings = validation.warnings || [];

    setText("resultTitle", chart.title || "生成结果");
    const meta = [
      `${chart.difficulty || "master"} ${chart.level || ""}`.trim(),
      chart.bpm ? `${chart.bpm} BPM` : null,
      stats.duration_seconds ? `${formatNumber(stats.duration_seconds, 1)}s` : null,
      chart.artist || null,
    ].filter(Boolean);
    setText("resultMeta", meta.join(" · "));

    const badge = $("resultValidation");
    if (badge) {
      if (errors.length) {
        badge.textContent = `${errors.length} 个校验错误`;
        badge.className = "badge bad";
        badge.title = errors.join("\n");
      } else {
        badge.textContent = warnings.length ? `校验通过 · ${warnings.length} warning` : "校验通过";
        badge.className = "badge good";
        badge.title = warnings.join("\n");
      }
    }

    setText("resultNotes", stats.notes_total ?? "—");
    setText("resultNps", formatNumber(stats.notes_per_second));
    setText("resultMaxSim", stats.max_simultaneous ?? "—");
    setText("resultDiv", stats.finest_division ?? "—");

    const counts = stats.counts || {};
    const total = Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0) || 1;
    const bars = Object.keys(TYPE_LABELS).map((key) => {
      const value = Number(counts[key] || 0);
      const pct = (value / total) * 100;
      return `
        <div class="type-row">
          <span>${TYPE_LABELS[key]}</span>
          <div class="type-track"><div class="type-fill" style="width:${pct.toFixed(1)}%"></div></div>
          <em>${value}</em>
        </div>`;
    }).join("");
    setHtml("resultTypeBars", bars);

    const audio = $("resultAudio");
    if (audio) {
      if (chart.audio) {
        audio.src = audioUrl(chart.audio);
        audio.classList.remove("hidden");
      } else {
        audio.removeAttribute("src");
        audio.classList.add("hidden");
      }
    }

    const links = (chart.files || []).map((file) => {
      const label = FILE_LABELS[file.name] || file.name;
      return `<a href="${downloadUrl(file.path)}" title="下载 ${escapeHtml(file.name)}">${escapeHtml(label)}</a>`;
    }).join("");
    setHtml("resultLinks", links);

    const preview = $("resultPreview");
    if (preview) {
      const lines = String(chart.text || "").split(/\r?\n/).slice(0, 80);
      preview.textContent = lines.join("\n");
    }
  }

  // ---------------------------------------------------------------- runs
  async function loadRuns() {
    const data = await api("/api/runs");
    renderRuns(data.runs || []);
  }

  function renderRuns(runs) {
    const host = $("runsList");
    if (!host) return;
    if (!runs.length) {
      host.innerHTML = `<p class="muted small">runs/ 下还没有训练记录。</p>`;
      return;
    }
    host.innerHTML = runs.map((run) => {
      const best = run.best_val_loss == null ? "—" : formatNumber(run.best_val_loss);
      const epoch = run.last_epoch == null ? "—" : Number(run.last_epoch) + 1;
      const checkpoints = run.checkpoints || [];
      const bestMark = checkpoints.includes("best.pt")
        ? '<span class="badge good">best.pt</span>'
        : "";
      return `
        <div class="run-card">
          <div>
            <h3>${escapeHtml(run.name)} ${bestMark}</h3>
            <p>${escapeHtml(run.path)}</p>
            <div class="run-metrics">
              <span>best val <strong>${best}</strong></span>
              <span>epoch <strong>${epoch}</strong></span>
              <span>step <strong>${run.last_step ?? "—"}</strong></span>
              <span>checkpoints <strong>${checkpoints.length}</strong></span>
            </div>
          </div>
          <button class="btn small ghost" type="button" data-run="${escapeHtml(run.name)}">查看</button>
        </div>`;
    }).join("");
    host.querySelectorAll("[data-run]").forEach((button) => {
      button.addEventListener("click", () => {
        loadRun(button.dataset.run).catch((error) => toast(error.message, "error"));
      });
    });
  }

  async function loadRun(name) {
    const data = await api(`/api/run/${encodeURIComponent(name)}`);
    state.selectedRun = name;
    state.runDetail = data;
    const panel = $("runDetail");
    panel.classList.remove("hidden");
    setText("runDetailTitle", data.name || name);
    const best = data.best_val_loss == null ? "—" : formatNumber(data.best_val_loss);
    const epoch = data.last_epoch == null ? "—" : Number(data.last_epoch) + 1;
    setText(
      "runDetailMeta",
      `${data.path || ""} · best val ${best} · epoch ${epoch} · step ${data.last_step ?? "—"}`
    );
    requestAnimationFrame(() => drawLossChart($("runCanvas"), data.metrics || []));

    const files = data.files || [];
    const checkpoints = files.filter((file) => file.name.endsWith(".pt"));
    const others = files.filter((file) => !file.name.endsWith(".pt"));
    $("runCheckpoints").innerHTML = checkpoints.length
      ? checkpoints.map((file) => `
          <div class="file-item">
            <code title="${escapeHtml(file.path)}">${escapeHtml(file.name)}</code>
            <button class="btn tiny ghost" type="button" data-ckpt="${escapeHtml(file.path)}">用于推理</button>
          </div>`).join("")
      : `<p class="muted small">没有 checkpoint</p>`;
    $("runFiles").innerHTML = others.length
      ? others.map((file) => `
          <div class="file-item">
            <code title="${escapeHtml(file.path)}">${escapeHtml(file.name)}</code>
            <span>
              <button class="btn tiny ghost" type="button" data-open="${escapeHtml(file.path)}">打开</button>
              <a class="btn tiny ghost" href="${downloadUrl(file.path)}">下载</a>
            </span>
          </div>`).join("")
      : `<p class="muted small">没有产物</p>`;

    $("runCheckpoints").querySelectorAll("[data-ckpt]").forEach((button) => {
      button.addEventListener("click", () => selectCheckpoint(button.dataset.ckpt));
    });
    $("runFiles").querySelectorAll("[data-open]").forEach((button) => {
      button.addEventListener("click", () => window.open(fileUrl(button.dataset.open), "_blank"));
    });
  }

  function selectCheckpoint(path) {
    const select = $("inferCheckpoint");
    if (!select) return;
    let found = Array.from(select.options).some((option) => option.value === path);
    if (!found) {
      const option = document.createElement("option");
      option.value = path;
      option.textContent = path;
      select.appendChild(option);
    }
    select.value = path;
    setTab("infer");
    toast(`已选择 checkpoint：${basename(path)}`, "success");
  }

  // ---------------------------------------------------------------- charts
  function drawLossChart(canvas, metrics, showEmpty = true) {
    if (!canvas) return;
    const width = canvas.clientWidth || canvas.parentElement?.clientWidth || 300;
    const height = 220;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.floor(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const rows = (metrics || []).filter((row) =>
      (typeof row.train_loss === "number" && Number.isFinite(row.train_loss)) ||
      (typeof row.val_loss === "number" && Number.isFinite(row.val_loss))
    );
    if (!rows.length) {
      if (showEmpty) {
        ctx.fillStyle = "rgba(139,155,178,0.6)";
        ctx.font = "11px ui-sans-serif, system-ui";
        ctx.textAlign = "center";
        ctx.fillText("暂无 loss 数据", width / 2, height / 2);
      }
      return;
    }

    const pad = { left: 40, right: 12, top: 18, bottom: 24 };
    const innerW = Math.max(1, width - pad.left - pad.right);
    const innerH = Math.max(1, height - pad.top - pad.bottom);
    const values = [];
    for (const row of rows) {
      if (typeof row.train_loss === "number") values.push(row.train_loss);
      if (typeof row.val_loss === "number") values.push(row.val_loss);
    }
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      min -= 0.05;
      max += 0.05;
    }
    const range = max - min;
    min -= range * 0.08;
    max += range * 0.08;

    const xAt = (index) => pad.left + (rows.length <= 1 ? innerW / 2 : (index / (rows.length - 1)) * innerW);
    const yAt = (value) => pad.top + (1 - (value - min) / (max - min)) * innerH;

    ctx.font = "10px ui-sans-serif, system-ui";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.strokeStyle = "rgba(126,157,196,0.13)";
    ctx.fillStyle = "rgba(139,155,178,0.75)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i += 1) {
      const value = min + ((max - min) * i) / 4;
      const y = yAt(value);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      ctx.fillText(value.toFixed(2), pad.left - 6, y);
    }

    const drawSeries = (key, color, dashed = false) => {
      const points = rows
        .map((row, index) => ({ index, value: row[key] }))
        .filter((point) => typeof point.value === "number" && Number.isFinite(point.value));
      if (!points.length) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.setLineDash(dashed ? [4, 4] : []);
      points.forEach((point, i) => {
        const x = xAt(point.index);
        const y = yAt(point.value);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      for (const point of points) {
        ctx.beginPath();
        ctx.arc(xAt(point.index), yAt(point.value), 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    drawSeries("train_loss", "#ff8a4c");
    drawSeries("val_loss", "#42d4ff", true);

    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#ff8a4c";
    ctx.fillRect(pad.left, 6, 14, 3);
    ctx.fillStyle = "rgba(232,238,248,0.82)";
    ctx.fillText("train", pad.left + 19, 8);
    ctx.fillStyle = "#42d4ff";
    ctx.fillRect(pad.left + 64, 6, 14, 3);
    ctx.fillStyle = "rgba(232,238,248,0.82)";
    ctx.fillText("val", pad.left + 83, 8);

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "rgba(139,155,178,0.7)";
    if (rows.length) {
      const firstEpoch = Number(rows[0].epoch);
      const lastEpoch = Number(rows[rows.length - 1].epoch);
      if (Number.isFinite(firstEpoch) && Number.isFinite(lastEpoch)) {
        ctx.fillText(`epoch ${firstEpoch + 1}`, pad.left + 14, height - pad.bottom + 6);
        ctx.fillText(`epoch ${lastEpoch + 1}`, width - pad.right - 20, height - pad.bottom + 6);
      }
    }
  }

  // ---------------------------------------------------------------- bind
  function bind() {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => setTab(tab.dataset.tab));
    });
    $("trainStart")?.addEventListener("click", startTrain);
    $("inferStart")?.addEventListener("click", startInfer);
    $("trainStop")?.addEventListener("click", stopJob);
    $("inferStop")?.addEventListener("click", stopJob);
    $("refreshBtn")?.addEventListener("click", async () => {
      try {
        await loadOptions();
        await loadRuns();
        await poll();
        toast("已刷新", "success");
      } catch (error) {
        toast(error.message, "error");
      }
    });
    $("runsRefresh")?.addEventListener("click", () => {
      loadRuns().catch((error) => toast(error.message, "error"));
    });
    $("runUseBest")?.addEventListener("click", () => {
      const files = state.runDetail?.files || [];
      const best = files.find((file) => file.name === "best.pt") || files.find((file) => file.name.endsWith(".pt"));
      if (best) selectCheckpoint(best.path);
      else toast("这个 run 里没有 checkpoint", "warn");
    });
    $("trainResume")?.addEventListener("change", (event) => {
      const value = event.target.value;
      if (value) {
        const folder = dirname(value);
        if (folder) $("trainOutput").value = folder;
      }
    });
    $("inferDifficulty")?.addEventListener("change", (event) => {
      const defaults = { basic: 7, advanced: 10, expert: 12.5, master: 14, remaster: 14.5 };
      const level = $("inferLevel");
      if (level) level.value = defaults[event.target.value] ?? level.value;
    });
    $("logAuto")?.addEventListener("change", (event) => {
      state.logAuto = event.target.checked;
    });
    $("logDownload")?.addEventListener("click", () => {
      const path = state.job?.log_path;
      if (!path) return toast("当前没有日志", "warn");
      window.open(downloadUrl(path), "_blank");
    });
    window.addEventListener("resize", () => {
      drawLossChart($("lossCanvas"), state.job?.metrics || [], false);
      if (state.runDetail) drawLossChart($("runCanvas"), state.runDetail.metrics || []);
    });
  }

  async function init() {
    bind();
    try {
      await loadOptions();
    } catch (error) {
      toast(`加载选项失败: ${error.message}`, "error", 7000);
    }
    await poll();
    window.setInterval(poll, 1000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
