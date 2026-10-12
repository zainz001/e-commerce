import { Body3DMeshTryOn } from "./body3d_mesh.js";

const camera = document.querySelector("#camera");
const render = document.querySelector("#render");
const canvas3d = document.querySelector("#canvas3d");
const context = render.getContext("2d");
const placeholder = document.querySelector("#placeholder");
const status = document.querySelector("#status");
const toggleButton = document.querySelector("#toggle-camera");
const tryOnButton = document.querySelector("#try-on");
const recordButton = document.querySelector("#record-calibration");
const calibrationStage = document.querySelector("#calibration-stage");
const recordingStatus = document.querySelector("#recording-status");
const garment = new Image();
garment.src = "../data/derived/product_cutouts/elaf_premium_10_front_cutout.png";

const TASKS_MODULE_URL = "./vendor/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";
const WASM_URL = "./vendor/node_modules/@mediapipe/tasks-vision/wasm";
const POSE_MODEL_URL = "./vendor/models/pose_landmarker_lite.task";
let stream = null;
let animationFrame = null;
let fittedStill = null;
let poseLandmarker = null;
let lastVideoTime = -1;
let fallbackFit = null;
let generating = false;
let mediaRecorder = null;
let recordingChunks = [];
let frameLog = [];
let frameLogTimer = null;
let frameCallbackId = null;
let recordingStartedAt = null;
let recordingShouldSave = false;

// Initialize 3D Body Mesh Try-On Engine
let body3d = null;
try {
  if (canvas3d) {
    body3d = new Body3DMeshTryOn(canvas3d, camera);
    body3d.loadGarment(garment.src);
  }
} catch (err) {
  console.warn("3D WebGL engine initialization:", err);
}

function setStatus(message, kind = "idle") {
  status.textContent = message;
  status.dataset.kind = kind;
}

const suitSelect = document.querySelector("#suit-select");
const suitUpload = document.querySelector("#suit-upload");
const garmentTitle = document.querySelector("#garment-title");
const suits = new Map([["builtin", { url: garment.src, label: "Elaf Premium 10" }]]);

function selectSuit(id) {
  const suit = suits.get(id);
  if (!suit) return;
  garment.src = suit.url;
  garmentTitle.textContent = suit.label;
  fittedStill = null;
  fallbackFit = null;
  neuralFrame = null;
  garmentDataUrl = null;
  if (body3d) body3d.loadGarment(suit.url);
  tryOnButton.textContent = "Generate fitted still (slow, optional)";
}

  suitSelect.addEventListener("change", () => selectSuit(suitSelect.value));
  suitUpload.addEventListener("change", () => {
    const file = suitUpload.files?.[0];
    if (!file) return;
    const id = `upload_${suits.size}`;
    const label = file.name.replace(/\.[^.]+$/, "").slice(0, 40) || `Suit ${suits.size}`;
    suits.set(id, { url: URL.createObjectURL(file), label });
    suitSelect.add(new Option(`${label} (added)`, id));
    suitSelect.value = id;
    suitUpload.value = "";
    selectSuit(id);
  });

  const stage = document.querySelector("#stage");
  if (stage) {
    stage.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      stage.style.outline = "2px dashed #00e5ff";
    });
    stage.addEventListener("dragleave", () => {
      stage.style.outline = "";
    });
    stage.addEventListener("drop", (e) => {
      e.preventDefault();
      stage.style.outline = "";
      const file = e.dataTransfer.files?.[0];
      if (file && file.type.startsWith("image/")) {
        const id = `upload_${suits.size}`;
        const label = file.name.replace(/\.[^.]+$/, "").slice(0, 40) || `Suit ${suits.size}`;
        suits.set(id, { url: URL.createObjectURL(file), label });
        suitSelect.add(new Option(`${label} (dropped)`, id));
        suitSelect.value = id;
        selectSuit(id);
      }
    });
  }

  function hexDigest(buffer) {
    return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function startFrameLogging() {
    frameLog = [];
    const recordFrame = (now, metadata = {}) => {
      if (!mediaRecorder || mediaRecorder.state !== "recording") return;
      frameLog.push({
        index: frameLog.length,
        media_time_seconds: metadata.mediaTime ?? camera.currentTime ?? null,
        expected_display_time_ms: metadata.expectedDisplayTime ?? now,
        presented_frames: metadata.presentedFrames ?? null,
      });
      if (typeof camera.requestVideoFrameCallback === "function") {
        frameCallbackId = camera.requestVideoFrameCallback(recordFrame);
      }
    };
    if (typeof camera.requestVideoFrameCallback === "function") {
      frameCallbackId = camera.requestVideoFrameCallback(recordFrame);
    } else {
      frameLogTimer = window.setInterval(() => recordFrame(performance.now()), 33);
    }
  }

  function stopFrameLogging() {
    if (frameCallbackId !== null && typeof camera.cancelVideoFrameCallback === "function") {
      camera.cancelVideoFrameCallback(frameCallbackId);
    }
    if (frameLogTimer !== null) window.clearInterval(frameLogTimer);
    frameCallbackId = null;
    frameLogTimer = null;
  }

  function recordingMimeType() {
    return ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]
      .find((type) => MediaRecorder.isTypeSupported(type)) || "";
  }

  function startCalibrationRecording() {
    if (!stream || mediaRecorder) return;
    if (!window.MediaRecorder) {
      recordingStatus.textContent = "This browser does not support local video recording.";
      return;
    }
    const mimeType = recordingMimeType();
    try {
      mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    } catch (error) {
      recordingStatus.textContent = `Could not start recording: ${error.message}`;
      return;
    }
    recordingChunks = [];
    recordingStartedAt = new Date();
    recordingShouldSave = true;
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) recordingChunks.push(event.data);
    };
    mediaRecorder.onstop = finishCalibrationRecording;
    mediaRecorder.start(1000);
    startFrameLogging();
    recordButton.textContent = "Stop & save full recording";
    recordButton.classList.add("recording");
    calibrationStage.disabled = true;
    recordingStatus.textContent = "Recording every available video frame locally…";
  }

  async function finishCalibrationRecording() {
    stopFrameLogging();
    const shouldSave = recordingShouldSave;
    const startedAt = recordingStartedAt;
    const chunks = recordingChunks;
    const type = mediaRecorder?.mimeType || "video/webm";
    mediaRecorder = null;
    recordingChunks = [];
    recordingShouldSave = false;
    recordButton.classList.remove("recording");
    recordButton.textContent = "Start full-video recording";
    recordButton.disabled = !stream;
    calibrationStage.disabled = false;

    if (!shouldSave || !chunks.length) {
      recordingStatus.textContent = "Recording discarded locally.";
      return;
    }

    const videoBlob = new Blob(chunks, { type });
    const finishedAt = new Date();
    const captureId = `calibration_${calibrationStage.value}_${finishedAt.toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}`;
    const metadata = {
      schema_version: "0.1.0",
      capture_id: captureId,
      capture_stage: calibrationStage.value,
      source_type: "browser_camera_local_capture",
      local_only: true,
      started_at: startedAt?.toISOString() ?? null,
      finished_at: finishedAt.toISOString(),
      duration_seconds: startedAt ? (finishedAt - startedAt) / 1000 : null,
      mime_type: type,
      width: camera.videoWidth || null,
      height: camera.videoHeight || null,
      frame_timing_count: frameLog.length,
      frame_timing: frameLog,
      note: "Full local browser recording; frame_timing preserves observed presentation timestamps. This is not a physical-fit or measurement record.",
    };
    try {
      metadata.original_sha256 = hexDigest(await crypto.subtle.digest("SHA-256", await videoBlob.arrayBuffer()));
      downloadBlob(videoBlob, `${captureId}.webm`);
      downloadBlob(new Blob([JSON.stringify(metadata, null, 2)], { type: "application/json" }), `${captureId}.json`);
      recordingStatus.textContent = `Saved full video and timing metadata (${frameLog.length} observed frames).`;
    } catch (error) {
      recordingStatus.textContent = `Recording ready but could not save metadata: ${error.message}`;
    }
  }

  function stopCalibrationRecording(save) {
    if (!mediaRecorder || mediaRecorder.state !== "recording") return;
    recordingShouldSave = save;
    mediaRecorder.stop();
    recordButton.disabled = true;
    recordingStatus.textContent = save ? "Finishing and hashing the full recording…" : "Stopping without saving…";
  }

  function drawMirroredVideo() {
    context.save();
    context.translate(render.width, 0);
    context.scale(-1, 1);
    context.drawImage(camera, 0, 0, render.width, render.height);
    context.restore();
  }

  function point(landmark) {
    return { x: (1 - landmark.x) * render.width, y: landmark.y * render.height };
  }

  function midpoint(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  // ---- Live garment warp (per video frame, runs fully in the browser) ----
  // The garment image is split into triangles pinned to tracked body points
  // (shoulders, hips, wrists). 2D visual warp only: NOT cloth simulation, NOT size/fit.
  let garmentBox = null;
  let smoothedBody = null;
  let missedFrames = 0;

  const ANCHORS = {
    neck: [0.5, 0.0], shL: [0.27, 0.07], shR: [0.73, 0.07],
    cuffLo: [0.01, 0.58], cuffLi: [0.17, 0.66], cuffRo: [0.99, 0.58], cuffRi: [0.83, 0.66],
    pitL: [0.29, 0.32], pitR: [0.71, 0.32], hemL: [0.2, 1.0], hemR: [0.8, 1.0],
  };
  const TRIANGLES = [
    ["neck", "shL", "shR"], ["shL", "shR", "pitR"], ["shL", "pitR", "pitL"],
    ["pitL", "pitR", "hemR"], ["pitL", "hemR", "hemL"],
    ["shL", "cuffLo", "cuffLi"], ["shL", "cuffLi", "pitL"],
    ["shR", "cuffRo", "cuffRi"], ["shR", "cuffRi", "pitR"],
  ];

  function measureGarment() {
    garmentBox = null;
    if (!garment.naturalWidth) return;
    const w = 256;
    const h = Math.max(1, Math.round((w * garment.naturalHeight) / garment.naturalWidth));
    const scratch = document.createElement("canvas");
    scratch.width = w;
    scratch.height = h;
    const g = scratch.getContext("2d", { willReadFrequently: true });
    g.drawImage(garment, 0, 0, w, h);
    const data = g.getImageData(0, 0, w, h).data;
    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        if (data[(y * w + x) * 4 + 3] > 24) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    const sx = garment.naturalWidth / w;
    const sy = garment.naturalHeight / h;
    garmentBox = maxX < 0
      ? { x: 0, y: 0, w: garment.naturalWidth, h: garment.naturalHeight }
      : { x: minX * sx, y: minY * sy, w: (maxX - minX + 1) * sx, h: (maxY - minY + 1) * sy };
  }
  garment.addEventListener("load", measureGarment);
  if (garment.complete) measureGarment();

  function smoothLandmarks(landmarks) {
    const next = landmarks.map((item) => ({ ...point(item), visibility: item.visibility ?? 1 }));
    if (!smoothedBody) {
      smoothedBody = next;
    } else {
      const k = 0.55;
      smoothedBody = next.map((item, i) => ({
        x: smoothedBody[i].x + (item.x - smoothedBody[i].x) * k,
        y: smoothedBody[i].y + (item.y - smoothedBody[i].y) * k,
        visibility: item.visibility,
      }));
    }
    return smoothedBody;
  }

  function add(a, b, s = 1) { return { x: a.x + b.x * s, y: a.y + b.y * s }; }
  function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
  function unit(v) { const n = Math.hypot(v.x, v.y) || 1; return { x: v.x / n, y: v.y / n }; }

  function buildLiveMesh(landmarks) {
    if (!landmarks || !garmentBox) return null;
    const body = smoothLandmarks(landmarks);
    if (![body[11], body[12]].every((p) => p.visibility >= 0.5)) return null;
    const sL = body[12];
    const sR = body[11];
    const sv = sub(sR, sL);
    const sw = Math.hypot(sv.x, sv.y);
    if (sw < 35) return null;
    const e = unit(sv);
    const mid = { x: (sL.x + sR.x) / 2, y: (sL.y + sR.y) / 2 };

    // Hips: tracked when visible, else estimated straight down from the shoulders.
    let down = unit({ x: -sv.y, y: sv.x });
    if (down.y < 0) down = { x: -down.x, y: -down.y };
    const hipsVisible = body[23].visibility >= 0.5 && body[24].visibility >= 0.5;
    const hL = hipsVisible ? body[24] : add(add(mid, down, sw * 1.35), e, -sw * 0.38);
    const hR = hipsVisible ? body[23] : add(add(mid, down, sw * 1.35), e, sw * 0.38);
    const hipMid = { x: (hL.x + hR.x) / 2, y: (hL.y + hR.y) / 2 };
    const torso = Math.max(Math.hypot(hipMid.x - mid.x, hipMid.y - mid.y), sw * 0.8);
    const dir = unit(sub(hipMid, mid));

    const cuffTarget = (shoulder, wristLm, side) => {
      const rest = add(add(shoulder, dir, torso * 0.95), e, side * sw * 0.3);
      const wrist = wristLm.visibility >= 0.5 ? wristLm : rest;
      const end = add(shoulder, sub(wrist, shoulder), 0.88);
      const arm = unit(sub(wrist, shoulder));
      let perp = { x: -arm.y, y: arm.x };
      // Orient perp away from the body on this side (left -> -e, right -> +e).
      if ((perp.x * e.x + perp.y * e.y) * side < 0) perp = { x: -perp.x, y: -perp.y };
      const cw = sw * 0.3;
      return { outer: add(end, perp, cw), inner: add(end, perp, -cw * 0.35) };
    };
    const left = cuffTarget(sL, body[16], -1);
    const right = cuffTarget(sR, body[15], 1);

    const dst = {
      neck: add(mid, dir, -torso * 0.06),
      shL: add(sL, e, -sw * 0.04), shR: add(sR, e, sw * 0.04),
      pitL: add(add(sL, sub(hL, sL), 0.35), e, -sw * 0.02),
      pitR: add(add(sR, sub(hR, sR), 0.35), e, sw * 0.02),
      hemL: add(add(hL, dir, torso * 0.8), e, -sw * 0.1),
      hemR: add(add(hR, dir, torso * 0.8), e, sw * 0.1),
      cuffLo: left.outer, cuffLi: left.inner, cuffRo: right.outer, cuffRi: right.inner,
    };
    const src = Object.fromEntries(Object.entries(ANCHORS).map(([key, [u, v]]) => [
      key, { x: garmentBox.x + u * garmentBox.w, y: garmentBox.y + v * garmentBox.h },
    ]));
    return { src, dst };
  }

  function drawTriangle(s, d, ctx = context) {
    const det = (s[1].x - s[0].x) * (s[2].y - s[0].y) - (s[2].x - s[0].x) * (s[1].y - s[0].y);
    if (Math.abs(det) < 1e-6) return;
    const a = ((d[1].x - d[0].x) * (s[2].y - s[0].y) - (d[2].x - d[0].x) * (s[1].y - s[0].y)) / det;
    const c = ((s[1].x - s[0].x) * (d[2].x - d[0].x) - (s[2].x - s[0].x) * (d[1].x - d[0].x)) / det;
    const b = ((d[1].y - d[0].y) * (s[2].y - s[0].y) - (d[2].y - d[0].y) * (s[1].y - s[0].y)) / det;
    const dd = ((s[1].x - s[0].x) * (d[2].y - d[0].y) - (s[2].x - s[0].x) * (d[1].y - d[0].y)) / det;
    const e = d[0].x - a * s[0].x - c * s[0].y;
    const f = d[0].y - b * s[0].x - dd * s[0].y;
    const cx = (d[0].x + d[1].x + d[2].x) / 3;
    const cy = (d[0].y + d[1].y + d[2].y) / 3;
    ctx.save();
    ctx.beginPath();
    d.forEach((p, i) => {
      const grow = unit({ x: p.x - cx, y: p.y - cy });
      const x = p.x + grow.x * 0.8;
      const y = p.y + grow.y * 0.8;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.clip();
    ctx.transform(a, b, c, dd, e, f);
    ctx.drawImage(garment, 0, 0);
    ctx.restore();
  }

  // Person silhouette from the pose model, mirrored to match the displayed video.
  const bodyMask = document.createElement("canvas");
  const bodyMaskContext = bodyMask.getContext("2d");
  let bodyMaskReady = false;
  const garmentLayer = document.createElement("canvas");
  const layerContext = garmentLayer.getContext("2d");

  function updateBodyMask(mask) {
    if (!mask) { bodyMaskReady = false; return; }
    try {
      const w = mask.width;
      const h = mask.height;
      const values = mask.getAsFloat32Array();
      if (bodyMask.width !== w || bodyMask.height !== h) {
        bodyMask.width = w;
        bodyMask.height = h;
      }
      const image = bodyMaskContext.createImageData(w, h);
      for (let i = 0; i < values.length; i += 1) {
        const o = i * 4;
        image.data[o] = image.data[o + 1] = image.data[o + 2] = 255;
        image.data[o + 3] = values[i] > 0.45 ? 255 : 0;
      }
      bodyMaskContext.putImageData(image, 0, 0);
      bodyMaskReady = true;
    } catch (error) {
      bodyMaskReady = false;
    } finally {
      mask.close?.();
    }
  }

  function drawLiveGarment(mesh) {
    if (!mesh || !garment.complete || !garment.naturalWidth) return;
    if (garmentLayer.width !== render.width || garmentLayer.height !== render.height) {
      garmentLayer.width = render.width;
      garmentLayer.height = render.height;
    }
    const lctx = layerContext;
    lctx.clearRect(0, 0, garmentLayer.width, garmentLayer.height);
    TRIANGLES.forEach((names) => drawTriangle(names.map((n) => mesh.src[n]), names.map((n) => mesh.dst[n]), lctx));

    // Soft shading so the print is not uniformly bright like a flat cutout:
    // darker toward the body sides, slightly dimmed to sit in the room's light.
    const left = Math.min(mesh.dst.shL.x, mesh.dst.hemL.x);
    const right = Math.max(mesh.dst.shR.x, mesh.dst.hemR.x);
    const shade = lctx.createLinearGradient(left, 0, right, 0);
    shade.addColorStop(0, "rgba(0,0,0,0.30)");
    shade.addColorStop(0.28, "rgba(0,0,0,0.04)");
    shade.addColorStop(0.72, "rgba(0,0,0,0.04)");
    shade.addColorStop(1, "rgba(0,0,0,0.30)");
    lctx.globalCompositeOperation = "source-atop";
    lctx.fillStyle = shade;
    lctx.fillRect(0, 0, garmentLayer.width, garmentLayer.height);
    lctx.fillStyle = "rgba(0,0,0,0.10)";
    lctx.fillRect(0, 0, garmentLayer.width, garmentLayer.height);

    // Keep the suit inside the person's outline (slightly dilated) so it hugs the
    // body instead of floating over the background.
    if (bodyMaskReady) {
      lctx.globalCompositeOperation = "destination-in";
      lctx.save();
      lctx.translate(garmentLayer.width, 0);
      lctx.scale(-1, 1);
      lctx.filter = "blur(2px)";
      const grow = Math.max(3, garmentLayer.width * 0.012);
      [[0, 0], [grow, 0], [-grow, 0], [0, grow], [0, -grow]].forEach(([dx, dy]) => {
        lctx.drawImage(bodyMask, dx, dy, garmentLayer.width, garmentLayer.height);
      });
      lctx.restore();
    }
    lctx.globalCompositeOperation = "source-over";
    context.save();
    context.globalAlpha = 0.97;
    context.drawImage(garmentLayer, 0, 0);
    context.restore();
  }

  async function loadPoseLandmarker() {
    if (poseLandmarker) return;
    try {
      const { FilesetResolver, PoseLandmarker } = await import(TASKS_MODULE_URL);
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      try {
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1,
          outputSegmentationMasks: true,
          minPoseDetectionConfidence: 0.55,
          minPosePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
      } catch {
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "CPU" },
          runningMode: "VIDEO",
          numPoses: 1,
          outputSegmentationMasks: true,
          minPoseDetectionConfidence: 0.55,
          minPosePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55,
        });
      }
    } catch (error) {
      console.error("Fallback pose guide unavailable", error);
    }
  }

  let neuralFrame = null;
  let neuralStreaming = false;
  let garmentDataUrl = null;
  let lastLandmarks = null;

  async function streamNeuralTryOn() {
    if (!stream || neuralStreaming || !camera.videoWidth) return;
    neuralStreaming = true;

    try {
      const snap = document.createElement("canvas");
      snap.width = camera.videoWidth;
      snap.height = camera.videoHeight;
      snap.getContext("2d").drawImage(camera, 0, 0, snap.width, snap.height);
      const personImage = snap.toDataURL("image/jpeg", 0.82);

      if (!garmentDataUrl || garmentDataUrl.src !== garment.src) {
        const resp = await fetch(garment.src);
        if (resp.ok) {
          garmentDataUrl = {
            src: garment.src,
            data: await blobToDataUrl(await resp.blob()),
          };
        }
      }

      let maskImage = null;
      if (bodyMaskReady && bodyMask.width > 0) {
        maskImage = bodyMask.toDataURL("image/png");
      }

      const res = await fetch("/api/live_neural_tryon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          person_image: personImage,
          garment_image: garmentDataUrl ? garmentDataUrl.data : null,
          body_mask: maskImage,
          landmarks: lastLandmarks,
        }),
      });

      if (res.ok) {
        const payload = await res.json();
        if (payload.image) {
          const img = new Image();
          await new Promise((resolve) => {
            img.onload = resolve;
            img.onerror = resolve;
            img.src = payload.image;
          });
          neuralFrame = img;
        }
      }
    } catch {
      // Keep feed smooth if a single frame drops
    } finally {
      neuralStreaming = false;
      if (stream) {
        requestAnimationFrame(streamNeuralTryOn);
      }
    }
  }

  function renderFrame() {
    if (!stream || camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    if (render.width !== camera.videoWidth || render.height !== camera.videoHeight) {
      render.width = camera.videoWidth;
      render.height = camera.videoHeight;
      if (body3d) body3d.resize(camera.videoWidth, camera.videoHeight);
    }

    let detectedLms = null;
    if (poseLandmarker && camera.currentTime !== lastVideoTime) {
      lastVideoTime = camera.currentTime;
      const result = poseLandmarker.detectForVideo(camera, performance.now());
      detectedLms = result.landmarks?.[0] || null;
      lastLandmarks = detectedLms;
      updateBodyMask(result.segmentationMasks?.[0]);
    }

    if (fittedStill) {
      context.drawImage(fittedStill, 0, 0, render.width, render.height);
      if (body3d) body3d.group.visible = false;
    } else {
      drawMirroredVideo();

      if (detectedLms && body3d) {
        // Mirror landmarks horizontally for mirror-view camera display
        const mirrored = detectedLms.map((p) => ({
          x: 1.0 - p.x,
          y: p.y,
          z: p.z || 0,
        }));
        body3d.updatePose(mirrored);
        body3d.render();
        setStatus("3D Mesh Fitting (Real-Time 60 FPS)", "live");
      } else if (body3d) {
        body3d.render();
      }
    }
    animationFrame = requestAnimationFrame(renderFrame);
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async function captureFittedStill() {
    if (fittedStill) {
      fittedStill = null;
      tryOnButton.textContent = "Generate fitted still (slow, optional)";
      return;
    }
    if (!stream || camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      setStatus("Turn on the camera first", "error");
      return;
    }
    if (generating) return;
    try {
      tryOnButton.disabled = true;
      generating = true;
      setStatus("Generating a fitted still locally…", "loading");
      const snapshot = document.createElement("canvas");
      snapshot.width = camera.videoWidth;
      snapshot.height = camera.videoHeight;
      // Keep the model input in the camera's natural orientation. Mirroring is
      // only a display choice and must not change garment/body orientation.
      snapshot.getContext("2d").drawImage(camera, 0, 0, snapshot.width, snapshot.height);
      const personImage = snapshot.toDataURL("image/jpeg", 0.94);
      const garmentResponse = await fetch(garment.src);
      if (!garmentResponse.ok) throw new Error("garment image could not be loaded");
      const garmentImage = await blobToDataUrl(await garmentResponse.blob());
      const response = await fetch("/api/tryon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ person_image: personImage, garment_image: garmentImage }),
      });
      const responseText = await response.text();
      let payload;
      try {
        payload = JSON.parse(responseText);
      } catch {
        throw new Error("Start prototype/local_server.py; the static server has no try-on endpoint");
      }
      if (!response.ok || !payload.image) throw new Error(payload.error || "local try-on failed");
      const image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
        image.src = payload.image;
      });
      fittedStill = image;
      tryOnButton.textContent = "Back to live view";
      setStatus("Fitted still ready (not live)", "live");
    } catch (error) {
      console.error("Local fitted still failed", error);
      setStatus(`Fitted still unavailable (live view continues): ${error.message || "start local_server.py"}`, "error");
    } finally {
      generating = false;
      tryOnButton.disabled = !stream;
    }
  }

  async function toggleCamera() {
    if (stream) {
      cancelAnimationFrame(animationFrame);
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
      camera.srcObject = null;
      context.clearRect(0, 0, render.width, render.height);
      placeholder.classList.remove("hidden");
      toggleButton.textContent = "Turn on camera";
      setStatus("Camera is off");
      fittedStill = null;
      fallbackFit = null;
      liveMesh = null;
      smoothedBody = null;
      neuralFrame = null;
      neuralStreaming = false;
      missedFrames = 0;
      lastVideoTime = -1;
      tryOnButton.disabled = true;
      if (mediaRecorder) stopCalibrationRecording(false);
      recordButton.disabled = true;
      poseLandmarker?.close();
      poseLandmarker = null;
      if (body3d) body3d.group.visible = false;
      return;
    }

    try {
      if (!window.isSecureContext) throw new Error("insecure_context");
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("camera_unavailable");
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      camera.srcObject = stream;
      await camera.play();
      placeholder.classList.add("hidden");
      toggleButton.textContent = "Turn off camera";
      tryOnButton.disabled = false;
      recordButton.disabled = false;
      setStatus("Starting Live 3D Mesh Try-On…", "loading");
      animationFrame = requestAnimationFrame(renderFrame);
      loadPoseLandmarker();
      streamNeuralTryOn();
    } catch (error) {
      console.error("Private try-on startup failed", error);
      const detail = error?.message ? `: ${error.message.slice(0, 120)}` : "";
      const message = error.name === "NotAllowedError"
        ? "Camera permission is blocked — allow Camera for localhost, then try again"
        : error.name === "NotFoundError"
          ? "No camera was found"
          : error.name === "NotReadableError"
            ? "The camera is busy in another app"
            : error.name === "SecurityError"
              ? "The browser blocked camera access for this page"
              : error.message === "insecure_context"
                ? "Open this page at http://localhost:8000/prototype/"
          : error.message === "camera_unavailable"
              ? "This browser does not provide camera access"
              : `Could not start the camera${detail}`;
      setStatus(message, "error");
    }
  }

  tryOnButton.addEventListener("click", captureFittedStill);
  recordButton.addEventListener("click", () => {
    if (mediaRecorder) stopCalibrationRecording(true);
    else startCalibrationRecording();
  });
  toggleButton.addEventListener("click", toggleCamera);
  window.addEventListener("beforeunload", () => {
    cancelAnimationFrame(animationFrame);
    stream?.getTracks().forEach((track) => track.stop());
    poseLandmarker?.close();
  });
