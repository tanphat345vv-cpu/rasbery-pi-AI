/**
 * Driver Drowsiness Detection - Mobile Edge AI Engine
 * Student: Bồ Tấn Phát - MSSV 123000144 - Đại học Lạc Hồng
 * Uses: Google MediaPipe Tasks Vision (FaceLandmarker) on WebAssembly + WebGL
 */

import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// DOM Elements
const video = document.getElementById("webcam");
const canvas = document.getElementById("output-canvas");
const ctx = canvas.getContext("2d");

const fpsVal = document.getElementById("fps-val");
const earVal = document.getElementById("ear-val");
const earFill = document.getElementById("ear-fill");
const frameVal = document.getElementById("frame-val");
const statusTag = document.getElementById("status-tag");

const alertBanner = document.getElementById("alert-banner");
const alertIcon = document.getElementById("alert-icon");
const alertTitle = document.getElementById("alert-title");
const alertSubtitle = document.getElementById("alert-subtitle");

const btnToggleCam = document.getElementById("btn-toggle-cam");
const btnCamText = document.getElementById("btn-cam-text");
const btnSwitchCam = document.getElementById("btn-switch-cam");
const btnSoundTest = document.getElementById("btn-sound-test");

const audioAlarm = document.getElementById("audio-alarm");

// Settings Modal Elements
const btnSettings = document.getElementById("btn-settings");
const btnCloseModal = document.getElementById("btn-close-modal");
const settingsModal = document.getElementById("settings-modal");
const threshEarSlider = document.getElementById("thresh-ear");
const threshEarVal = document.getElementById("thresh-ear-val");
const threshDrowsySlider = document.getElementById("thresh-drowsy");
const threshDrowsyVal = document.getElementById("thresh-drowsy-val");
const threshSleepSlider = document.getElementById("thresh-sleep");
const threshSleepVal = document.getElementById("thresh-sleep-val");
const toggleVibrate = document.getElementById("toggle-vibrate");

// Algorithm Configuration
let THRESHOLD_EAR = 0.22;
let FRAMES_DROWSY = 20;
let FRAMES_SLEEP = 40;
let ALLOW_VIBRATE = true;

// State Variables
let faceLandmarker = null;
let runningMode = "VIDEO";
let isCameraActive = false;
let currentFacingMode = "user"; // "user" (front) or "environment" (back)
let lastVideoTime = -1;

let consecutiveClosedFrames = 0;
let isSleeping = false;
let isAlarmPlaying = false;

// FPS tracking
let frameTimes = [];
let lastFpsUpdate = 0;

// Standard MediaPipe Face Mesh Landmark Indices for Eyes
// Soukupová and Čech (2016) 6-point formulation
const LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144];

// Initialize MediaPipe FaceLandmarker
async function initMediaPipe() {
  btnCamText.innerText = "ĐANG TẢI AI...";
  btnToggleCam.disabled = true;

  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    // Try local asset first, fallback to CDN if not hosted locally
    let modelPath = "./assets/face_landmarker.task";
    try {
      const testReq = await fetch(modelPath, { method: "HEAD" });
      if (!testReq.ok) throw new Error("Local model not reachable");
    } catch {
      modelPath = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
    }

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath,
        delegate: "GPU"
      },
      outputFaceBlendshapes: false,
      runningMode: runningMode,
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    console.log("[EdgeAI] MediaPipe FaceLandmarker ready!");
    btnCamText.innerText = "BẬT CAMERA";
    btnToggleCam.disabled = false;
  } catch (err) {
    console.error("[EdgeAI] Model loading error:", err);
    alert("Lỗi tải mô hình AI: " + err.message + "\nVui lòng tải lại trang.");
    btnCamText.innerText = "LỖI TẢI AI";
  }
}

// Distance between 2 points in 2D space
function dist2D(p1, p2, width, height) {
  const dx = (p1.x - p2.x) * width;
  const dy = (p1.y - p2.y) * height;
  return Math.sqrt(dx * dx + dy * dy);
}

// Calculate Eye Aspect Ratio (EAR)
function calculateEAR(landmarks, indices, width, height) {
  const p0 = landmarks[indices[0]]; // Outer corner
  const p1 = landmarks[indices[1]]; // Upper outer
  const p2 = landmarks[indices[2]]; // Upper inner
  const p3 = landmarks[indices[3]]; // Inner corner
  const p4 = landmarks[indices[4]]; // Lower inner
  const p5 = landmarks[indices[5]]; // Lower outer

  const v1 = dist2D(p1, p5, width, height);
  const v2 = dist2D(p2, p4, width, height);
  const h = dist2D(p0, p3, width, height);

  if (h === 0) return 0.0;
  return (v1 + v2) / (2.0 * h);
}

// Draw eye polygon contour
function drawEyeContour(ctx, landmarks, indices, width, height, color = "#10b981") {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.fillStyle = color + "33"; // transparent fill
  ctx.beginPath();

  for (let i = 0; i < indices.length; i++) {
    const pt = landmarks[indices[i]];
    const px = pt.x * width;
    const py = pt.y * height;
    if (i === 0) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  }

  ctx.closePath();
  ctx.stroke();
  ctx.fill();
  ctx.restore();
}

// Play Sound Alarm safely
function triggerAlarm(shouldPlay) {
  if (shouldPlay) {
    if (!isAlarmPlaying) {
      audioAlarm.currentTime = 0;
      audioAlarm.play().catch(e => console.log("Audio play blocked by browser:", e));
      isAlarmPlaying = true;
    }
    if (ALLOW_VIBRATE && navigator.vibrate) {
      navigator.vibrate([400, 200, 400]);
    }
  } else {
    if (isAlarmPlaying) {
      audioAlarm.pause();
      audioAlarm.currentTime = 0;
      isAlarmPlaying = false;
    }
  }
}

// Main Frame Processing Loop
async function renderLoop() {
  if (!isCameraActive) return;

  const now = performance.now();
  frameTimes.push(now);
  while (frameTimes.length > 0 && frameTimes[0] <= now - 1000) {
    frameTimes.shift();
  }
  if (now - lastFpsUpdate > 300) {
    fpsVal.innerText = frameTimes.length;
    lastFpsUpdate = now;
  }

  if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;

    const canvasWidth = video.videoWidth;
    const canvasHeight = video.videoHeight;

    if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
    }

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    const startTimeMs = performance.now();
    const results = faceLandmarker.detectForVideo(video, startTimeMs);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];

      // Calculate EAR for both eyes
      const leftEAR = calculateEAR(landmarks, LEFT_EYE_INDICES, canvasWidth, canvasHeight);
      const rightEAR = calculateEAR(landmarks, RIGHT_EYE_INDICES, canvasWidth, canvasHeight);
      const avgEAR = (leftEAR + rightEAR) / 2.0;

      // Update HUD
      earVal.innerText = avgEAR.toFixed(2);
      const pct = Math.min(Math.max((avgEAR / 0.40) * 100, 0), 100);
      earFill.style.width = pct + "%";

      const EAR_CLOSE = 0.20;
      const EAR_OPEN = 0.25;

      let eyeColor = "#10b981";

      // Case 1: Eyes Closed
      if (leftEAR < EAR_CLOSE && rightEAR < EAR_CLOSE) {
        consecutiveClosedFrames++;
        eyeColor = "#ef4444";
        earFill.style.backgroundColor = "var(--accent-red)";

        if (consecutiveClosedFrames >= FRAMES_SLEEP) {
          // Level 3: Deep Sleep Alarm
          triggerAlarm(true);
          showAlert("sleep", "🚨 CẢNH BÁO: NGỦ GẬT SÂU!", `Tài xế nhắm mắt liên tục ${consecutiveClosedFrames} frames!`);
          statusTag.innerText = "NGỦ GẬT SÂU";
          statusTag.style.color = "var(--accent-red)";
          statusTag.style.background = "rgba(239, 68, 68, 0.2)";
        } else if (consecutiveClosedFrames >= 15) {
          // Level 2: Microsleep
          showAlert("drowsy", "⚠️ NGUY HIỂM: GIẤC NGỦ TRẮNG!", `Mắt đã nhắm ${consecutiveClosedFrames} frames.`);
          statusTag.innerText = "GIẤC NGỦ TRẮNG";
          statusTag.style.color = "var(--accent-orange)";
          statusTag.style.background = "rgba(245, 158, 11, 0.2)";
        }
      }
      // Case 2: Drowsy / Squinting
      else if (EAR_CLOSE <= avgEAR && avgEAR <= EAR_OPEN) {
        consecutiveClosedFrames++;
        eyeColor = "#f59e0b";
        earFill.style.backgroundColor = "var(--accent-orange)";

        if (consecutiveClosedFrames >= FRAMES_DROWSY) {
          showAlert("drowsy", "⚡ NHẮC NHỞ: MẮT LỜ ĐỜ!", "Tài xế có dấu hiệu mệt mỏi, mất tập trung.");
          statusTag.innerText = "MẮT LỜ ĐỜ";
          statusTag.style.color = "var(--accent-orange)";
          statusTag.style.background = "rgba(245, 158, 11, 0.2)";
        }
      }
      // Case 3: Awake / Open
      else {
        consecutiveClosedFrames = 0;
        triggerAlarm(false);
        hideAlert();
        earFill.style.backgroundColor = "var(--accent-green)";
        statusTag.innerText = "TỈNH TÁO";
        statusTag.style.color = "var(--accent-green)";
        statusTag.style.background = "rgba(16, 185, 129, 0.2)";
      }

      frameVal.innerText = consecutiveClosedFrames;

      // Draw Eye Contours
      drawEyeContour(ctx, landmarks, LEFT_EYE_INDICES, canvasWidth, canvasHeight, eyeColor);
      drawEyeContour(ctx, landmarks, RIGHT_EYE_INDICES, canvasWidth, canvasHeight, eyeColor);

    } else {
      // No face detected
      consecutiveClosedFrames = 0;
      triggerAlarm(false);
      hideAlert();
      earVal.innerText = "--";
      earFill.style.width = "0%";
      statusTag.innerText = "KHÔNG THẤY MẶT";
      statusTag.style.color = "var(--text-muted)";
      statusTag.style.background = "rgba(255, 255, 255, 0.05)";
    }
  }

  requestAnimationFrame(renderLoop);
}

function showAlert(type, title, subtitle) {
  alertBanner.className = `alert-banner ${type}`;
  alertIcon.innerText = type === "sleep" ? "🚨" : "⚠️";
  alertTitle.innerText = title;
  alertSubtitle.innerText = subtitle;
}

function hideAlert() {
  alertBanner.className = "alert-banner hidden";
}

// Start / Stop Camera Stream
async function toggleCamera() {
  if (isCameraActive) {
    // Stop Camera
    const stream = video.srcObject;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    isCameraActive = false;
    btnCamText.innerText = "BẬT CAMERA";
    btnToggleCam.className = "btn btn-primary";
    triggerAlarm(false);
    hideAlert();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  } else {
    // Start Camera
    try {
      const constraints = {
        video: {
          facingMode: currentFacingMode,
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      video.addEventListener("loadeddata", () => {
        isCameraActive = true;
        btnCamText.innerText = "TẮT CAMERA";
        btnToggleCam.className = "btn btn-secondary";
        renderLoop();
      }, { once: true });

    } catch (err) {
      console.error("Camera access error:", err);
      alert("Không thể mở Camera: " + err.message + "\nVui lòng cấp quyền truy cập Camera trên trình duyệt.");
    }
  }
}

// Switch Camera front/back
async function switchCamera() {
  currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
  video.style.transform = currentFacingMode === "user" ? "scaleX(-1)" : "scaleX(1)";
  canvas.style.transform = currentFacingMode === "user" ? "scaleX(-1)" : "scaleX(1)";

  if (isCameraActive) {
    await toggleCamera();
    await toggleCamera();
  }
}

// Event Listeners
btnToggleCam.addEventListener("click", toggleCamera);
btnSwitchCam.addEventListener("click", switchCamera);

btnSoundTest.addEventListener("click", () => {
  audioAlarm.currentTime = 0;
  audioAlarm.play().catch(e => alert("Âm thanh bị chặn: " + e.message));
  if (navigator.vibrate) navigator.vibrate([200, 100, 200]);
});

// Settings Modal Listeners
btnSettings.addEventListener("click", () => settingsModal.classList.remove("hidden"));
btnCloseModal.addEventListener("click", () => settingsModal.classList.add("hidden"));
settingsModal.addEventListener("click", (e) => {
  if (e.target === settingsModal) settingsModal.classList.add("hidden");
});

threshEarSlider.addEventListener("input", (e) => {
  THRESHOLD_EAR = parseFloat(e.target.value);
  threshEarVal.innerText = THRESHOLD_EAR.toFixed(2);
});

threshDrowsySlider.addEventListener("input", (e) => {
  FRAMES_DROWSY = parseInt(e.target.value);
  threshDrowsyVal.innerText = FRAMES_DROWSY;
});

threshSleepSlider.addEventListener("input", (e) => {
  FRAMES_SLEEP = parseInt(e.target.value);
  threshSleepVal.innerText = FRAMES_SLEEP;
});

toggleVibrate.addEventListener("change", (e) => {
  ALLOW_VIBRATE = e.target.checked;
});

// Start initialization
initMediaPipe();
