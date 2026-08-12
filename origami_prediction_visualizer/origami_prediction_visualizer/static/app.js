import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import URDFLoader from './vendor/URDFLoader.js';

const $ = id => document.getElementById(id);
const ui = {
  predictionBadge: $('predictionBadge'), gtBadge: $('gtBadge'), urdfBadge: $('urdfBadge'), meshBadge: $('meshBadge'),
  metrics: $('metrics'), warnings: $('warnings'), stateRows: $('stateRows'), robotView: $('robotView'),
  timeline: $('timeline'), stepText: $('stepText'), stageText: $('stageText'), horizonText: $('horizonText'), timeText: $('timeText'),
  playPause: $('playPause'), stepBack: $('stepBack'), stepForward: $('stepForward'),
  showPrediction: $('showPrediction'), showGT: $('showGT'),
  headLeftVideo: $('headLeftVideo'), wristLeftVideo: $('wristLeftVideo'),
  headLeftStatus: $('headLeftStatus'), wristLeftStatus: $('wristLeftStatus'),
  recordViews: $('recordViews'), recordButton: $('recordButton'), recordStatus: $('recordStatus'),
};
const model = {
  config: null, trajectory: null, predictionRobot: null, gtRobot: null,
  step: 0, playing: false, raf: null, lastUiUpdate: 0, lastStepRender: -1,
  recording: false, recorder: null, recordCanvas: null, recordCtx: null, recordStream: null,
};
let scene, camera, renderer, controls;

const COLORS = { prediction: 0x79b7d8, gt: 0x82c99a };
// Camera coordinates are WORLD coordinates in the Three.js scene.  The
// built-ins are intentionally explicit so they are also a usable starter
// config for --camera-config.
const BUILTIN_CAMERA_VIEWS = {
  front: {position: [0.0, 0.85, 2.9], target: [0.0, 1.30, 0.0], fov: 42, label: 'Front'},
  top: {position: [0.0, 3.60, 0.05], target: [0.0, 0.0, 0.0], fov: 42, label: 'Top'},
  hands: {position: [1.25, 1.15, 1.25], target: [0.0, 1.60, 0.0], fov: 38, label: 'Hands close-up'},
};
let VIEW_PRESETS = BUILTIN_CAMERA_VIEWS;

async function api(path) {
  const response = await fetch(path, {cache: 'no-store'});
  const value = await response.json();
  if (!response.ok || value?.ok === false) throw new Error(String(value?.error || `HTTP ${response.status}`));
  return value;
}
function badge(node, text, tone = '') { node.textContent = String(text); node.className = `badge${tone ? ` ${tone}` : ''}`; }
function metric(label, value) {
  const row = document.createElement('div'); row.className = 'metric';
  const a = document.createElement('span'); const b = document.createElement('span');
  a.textContent = label; b.textContent = value == null ? '—' : String(value); row.append(a, b); return row;
}
function totalSteps() {
  const p = model.trajectory?.prediction?.length || 0, g = model.trajectory?.ground_truth_action?.length || 0;
  return Math.max(p, g);
}
function currentPrediction() { return model.trajectory?.prediction?.[model.step] || null; }
function currentGT() { return model.trajectory?.ground_truth_action?.[model.step] || null; }
function currentTime() {
  const ts = model.trajectory?.timestamps;
  if (ts?.length) {
    if (model.step < ts.length) return Number(ts[model.step]);
    const fps = Number(model.trajectory.dataset_fps || model.trajectory.control_hz || 30);
    return Number(ts[ts.length - 1]) + (model.step - ts.length + 1) / fps;
  }
  return model.step / Number(model.trajectory?.control_hz || 30);
}
function episodeRelativeTime(absTime = currentTime()) {
  return Math.max(0, Number(absTime) - Number(model.trajectory?.episode_time_origin || 0));
}

function renderMetrics() {
  const t = model.trajectory, gt = t.ground_truth_action, pLen = t.prediction.length, gLen = gt?.length || 0;
  const overlap = Math.min(pLen, gLen), errs = [];
  for (let i = 0; i < overlap; i++) {
    let sum = 0, max = 0;
    for (let j = 0; j < 65; j++) { const e = Math.abs(Number(t.prediction[i][j]) - Number(gt[i][j])); sum += e; if (e > max) max = e; }
    errs.push({mean: sum / 65, max});
  }
  const meanError = errs.length ? errs.reduce((a,b) => a + b.mean, 0) / errs.length : null;
  const maxError = errs.length ? Math.max(...errs.map(x => x.max)) : null;
  const rows = [
    ['Prediction', `${pLen} steps`], ['GT action', gt ? `${gLen} steps` : 'not loaded'], ['Overlap', overlap ? `${overlap} steps` : '—'],
    ['Control Hz', t.control_hz], ['Pred stages', t.stage_count], ['Action horizon', t.action_horizon],
    ['Mean |pred-GT|', meanError == null ? '—' : `${meanError.toFixed(5)} rad`],
    ['Max |pred-GT|', maxError == null ? '—' : `${maxError.toFixed(5)} rad`], ['Prediction limits', t.compatible ? 'OK' : 'violations'],
  ];
  ui.metrics.replaceChildren(...rows.map(([a,b]) => metric(a,b)));
  badge(ui.predictionBadge, `${pLen} predicted steps · ${t.stage_count} stages`, 'ok');
  badge(ui.gtBadge, gt ? `${gLen} GT action steps` : 'GT not loaded', gt ? 'ok' : 'warn');
}
function renderState() {
  const pred = currentPrediction(), gt = currentGT(); if (!pred && !gt) return;
  const limits = model.config?.limits || {}, names = model.trajectory.metadata.joint_names, fragment = document.createDocumentFragment();
  names.forEach((name, i) => {
    const tr = document.createElement('tr'), limit = limits[name];
    const range = limit ? `[${limit.lower.toFixed(3)}, ${limit.upper.toFixed(3)}]` : '—';
    const vel = limit ? `${limit.velocity.toFixed(3)} rad/s` : '—';
    const pv = pred ? Number(pred[i]).toFixed(6) : '—', gv = gt ? Number(gt[i]).toFixed(6) : '—';
    const ev = pred && gt ? Math.abs(Number(pred[i]) - Number(gt[i])).toFixed(6) : '—';
    for (const text of [i, name, pv, gv, ev, range, vel]) { const td = document.createElement('td'); td.textContent = String(text); tr.append(td); }
    fragment.append(tr);
  });
  ui.stateRows.replaceChildren(fragment);
}
function renderWarnings() {
  const report = model.trajectory.validation.step_reports?.[model.step], issues = report?.violations || [];
  if (!issues.length) { const li = document.createElement('li'); li.textContent = model.step < model.trajectory.prediction.length ? 'None' : 'No prediction at this step'; ui.warnings.replaceChildren(li); return; }
  const fragment = document.createDocumentFragment();
  issues.slice(0, 100).forEach(issue => { const li = document.createElement('li'); li.textContent = `${issue.joint_name}: ${issue.type}`; fragment.append(li); });
  ui.warnings.replaceChildren(fragment);
}

function segmentVideoTime(video, absTime, segment) {
  const episodeTime = episodeRelativeTime(absTime), start = Number(segment.from_timestamp || 0);
  return Math.max(0, start + episodeTime);
}
function seekVideos(absTime = currentTime()) {
  const segments = model.trajectory.video_segments || {};
  for (const [key, video, status] of [['head_left', ui.headLeftVideo, ui.headLeftStatus], ['wrist_left', ui.wristLeftVideo, ui.wristLeftStatus]]) {
    const segment = segments[key]; if (!video.src || !segment) continue;
    const episodeTime = episodeRelativeTime(absTime), duration = Number(segment.episode_duration ?? Infinity);
    if (episodeTime > duration + 0.05) { status.textContent = `Episode video ends at ${duration.toFixed(3)} s`; continue; }
    const target = segmentVideoTime(video, absTime, segment);
    if (!Number.isFinite(video.duration) || Math.abs(video.currentTime - target) > 0.04) {
      try { video.currentTime = Math.max(0, Math.min(target, Number.isFinite(video.duration) ? video.duration : target)); } catch {}
    }
    status.textContent = `Synced · episode ${episodeTime.toFixed(3)} s · file ${target.toFixed(3)} s`;
  }
}
function configureVideos(t) {
  const vids = [['head_left', ui.headLeftVideo, ui.headLeftStatus], ['wrist_left', ui.wristLeftVideo, ui.wristLeftStatus]];
  for (const [key, video, status] of vids) {
    const url = t.video_urls?.[key], segment = t.video_segments?.[key];
    if (!url || !segment) { status.textContent = 'Not configured'; continue; }
    video.src = url; video.load();
    video.addEventListener('loadedmetadata', () => { status.textContent = `Ready · ${video.duration.toFixed(3)} s`; }, {once:true});
  }
}
function updatePlayer({seekVideo = false, fullUI = true} = {}) {
  const total = totalSteps(), step = model.step;
  if (!total) return;
  const stage = Math.floor(step / model.trajectory.action_horizon), horizon = step % model.trajectory.action_horizon;
  ui.timeline.max = String(Math.max(0, total - 1)); ui.timeline.value = String(step);
  ui.stepText.textContent = `Step ${step + 1} / ${total}`;
  ui.stageText.textContent = `Stage ${stage + 1} / ${model.trajectory.stage_count}`;
  ui.horizonText.textContent = `Horizon ${horizon + 1} / ${model.trajectory.action_horizon}`;
  ui.timeText.textContent = `Time ${currentTime().toFixed(3)} s`;
  ui.stepBack.disabled = step <= 0; ui.stepForward.disabled = step >= total - 1; ui.playPause.disabled = total <= 1;
  renderRobots();
  if (fullUI) { renderState(); renderWarnings(); }
  if (seekVideo) seekVideos();
}
function setStep(step, {seekVideo = true} = {}) {
  const total = totalSteps(); model.step = Math.max(0, Math.min(Number(step) || 0, Math.max(0, total - 1))); updatePlayer({seekVideo, fullUI: true});
}
function setVideosPlaying(playing) {
  const vids = [ui.headLeftVideo, ui.wristLeftVideo].filter(v => v.src);
  for (const v of vids) {
    if (playing) v.play().catch(() => {}); else v.pause();
  }
}
function stopPlaying() {
  model.playing = false; ui.playPause.textContent = '▶';
  if (model.raf !== null) cancelAnimationFrame(model.raf); model.raf = null; setVideosPlaying(false);
}
function nearestStepForEpisodeTime(t) {
  const ts = model.trajectory?.timestamps || [], origin = Number(model.trajectory?.episode_time_origin || 0), target = t + origin;
  if (!ts.length) return Math.round(t * Number(model.trajectory?.control_hz || 30));
  let lo = 0, hi = Math.min(ts.length, totalSteps()) - 1;
  while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if (Number(ts[mid]) < target) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(Number(ts[lo - 1]) - target) < Math.abs(Number(ts[lo]) - target)) return lo - 1;
  return lo;
}
function playbackLoop(now) {
  if (!model.playing) return;
  const head = ui.headLeftVideo.src ? ui.headLeftVideo : (ui.wristLeftVideo.src ? ui.wristLeftVideo : null);
  let nextStep;
  if (head && !head.paused && Number.isFinite(head.currentTime)) {
    const segment = model.trajectory.video_segments?.head_left || model.trajectory.video_segments?.wrist_left;
    const start = Number(segment?.from_timestamp || 0);
    const episodeTime = Math.max(0, head.currentTime - start);
    nextStep = nearestStepForEpisodeTime(episodeTime);
    // Keep the second camera close without continuously seeking it.
    if (ui.wristLeftVideo.src && Number.isFinite(ui.wristLeftVideo.currentTime) && (now % 1000) < 25) {
      const wristSeg = model.trajectory.video_segments?.wrist_left;
      const desired = segmentVideoTime(ui.wristLeftVideo, currentTime(), wristSeg);
      if (Math.abs(ui.wristLeftVideo.currentTime - desired) > 0.10) { try { ui.wristLeftVideo.currentTime = desired; } catch {} }
    }
  } else {
    const elapsed = (now - (model.playStartPerf || now)) / 1000;
    nextStep = nearestStepForEpisodeTime(elapsed);
  }
  nextStep = Math.min(nextStep, totalSteps() - 1);
  if (nextStep !== model.step) {
    model.step = nextStep;
    renderRobots();
    if (now - model.lastUiUpdate > 80 || nextStep === totalSteps() - 1) { model.lastUiUpdate = now; updatePlayer({seekVideo:false, fullUI:true}); }
  }
  if (model.step >= totalSteps() - 1) { stopPlaying(); return; }
  model.raf = requestAnimationFrame(playbackLoop);
}
function startPlaying() {
  stopPlaying(); if (totalSteps() <= 1) return;
  seekVideos(); setVideosPlaying(true); model.playing = true; model.playStartPerf = performance.now() - episodeRelativeTime() * 1000; ui.playPause.textContent = '❚❚';
  model.raf = requestAnimationFrame(playbackLoop);
}
ui.playPause.addEventListener('click', () => model.playing ? stopPlaying() : startPlaying());
ui.stepBack.addEventListener('click', () => { stopPlaying(); setStep(model.step - 1); });
ui.stepForward.addEventListener('click', () => { stopPlaying(); setStep(model.step + 1); });
ui.timeline.addEventListener('input', e => { stopPlaying(); setStep(e.target.value, {seekVideo:true}); });
ui.showPrediction.addEventListener('change', () => { if (model.predictionRobot) model.predictionRobot.visible = ui.showPrediction.checked; });
ui.showGT.addEventListener('change', () => { if (model.gtRobot) model.gtRobot.visible = ui.showGT.checked; });

function initScene() {
  scene = new THREE.Scene(); scene.background = new THREE.Color(0x07101b);
  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000); camera.position.set(2.6, 1.8, 3.2);
  renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:false}); renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.outputColorSpace = THREE.SRGBColorSpace;
  ui.robotView.append(renderer.domElement); controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.target.set(0, 0.8, 0);
  scene.add(new THREE.HemisphereLight(0xd7e8ff, 0x26313d, 2.4)); const key = new THREE.DirectionalLight(0xffffff, 3); key.position.set(4, 7, 5); scene.add(key);
  scene.add(new THREE.GridHelper(12, 24, 0x36506f, 0x1a2a3d));
  const resize = () => { const width = Math.max(1, ui.robotView.clientWidth), height = Math.max(1, ui.robotView.clientHeight); renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix(); };
  new ResizeObserver(resize).observe(ui.robotView); resize(); requestAnimationFrame(renderFrame);
}
function renderFrame() { controls.update(); renderer.render(scene, camera); if (model.recording) drawRecordingFrame(); requestAnimationFrame(renderFrame); }
function fitRobots() {
  const objects = [model.predictionRobot, model.gtRobot].filter(Boolean); if (!objects.length) return;
  const box = new THREE.Box3(); objects.forEach(o => box.expandByObject(o)); if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3()), radius = Math.max(size.x, size.y, size.z, 0.5);
  controls.target.copy(center); camera.position.copy(center).add(new THREE.Vector3(radius * 1.8, radius * 1.15, radius * 2.1)); camera.near = Math.max(radius / 1000, 0.005); camera.far = Math.max(radius * 100, 100); camera.updateProjectionMatrix(); model.viewScale = {center, radius};
}
function styleRobot(robot, opacity, color) {
  robot.traverse(object => {
    if (!object.isMesh) return;
    object.castShadow = true; object.receiveShadow = true;
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach(mat => {
      if (!mat) return;
      mat.color = new THREE.Color(color); mat.transparent = true; mat.opacity = opacity; mat.depthWrite = opacity >= 0.9;
      mat.roughness = Math.max(Number(mat.roughness ?? 0.8), 0.65);
    });
  });
}
function renderRobot(robot, state) {
  if (!robot || !state) return;
  for (const entry of model.config?.joint_map || []) {
    const value = Number(state[entry.index]), joint = robot.joints?.[entry.name];
    if (!joint || !Number.isFinite(value) || typeof joint.setJointValue !== 'function') continue;
    try { joint.setJointValue(value); } catch {}
  }
}
function renderRobots() {
  renderRobot(model.predictionRobot, currentPrediction()); renderRobot(model.gtRobot, currentGT());
  if (model.predictionRobot) model.predictionRobot.visible = ui.showPrediction.checked && !!currentPrediction();
  if (model.gtRobot) model.gtRobot.visible = ui.showGT.checked && !!currentGT();
}
function loadUrdf(config, opacity, label, color) {
  return new Promise((resolve, reject) => {
    const url = new URL(config.urdf_url, window.location.href).href, manager = new THREE.LoadingManager();
    manager.onStart = () => badge(ui.meshBadge, `Loading ${label} mesh…`, 'warn'); manager.onLoad = () => badge(ui.meshBadge, 'Meshes ready', 'ok'); manager.onError = () => badge(ui.meshBadge, 'Some meshes failed', 'bad');
    const loader = new URDFLoader(manager), urdfBase = new URL('.', url); loader.packages = urdfBase.pathname.endsWith('/urdf/') ? new URL('../', urdfBase).href : urdfBase.href;
    loader.load(url, robot => { robot.rotation.x = -Math.PI / 2; styleRobot(robot, opacity, color); scene.add(robot); resolve(robot); }, undefined, error => reject(error instanceof Error ? error : new Error('URDF loading failed')));
  });
}
function configureCameraViews(t) {
  const configured = t.camera_views || {};
  VIEW_PRESETS = {...BUILTIN_CAMERA_VIEWS};
  for (const [name, raw] of Object.entries(configured)) {
    if (!raw || !Array.isArray(raw.position) || raw.position.length !== 3 ||
        !Array.isArray(raw.target) || raw.target.length !== 3) continue;
    VIEW_PRESETS[name] = {
      position: raw.position.map(Number),
      target: raw.target.map(Number),
      fov: Number(raw.fov ?? 42),
      near: Number(raw.near ?? 0.01),
      far: Number(raw.far ?? 1000),
      label: String(raw.label ?? name),
    };
  }
}
function setCameraPreset(name) {
  const v = VIEW_PRESETS[name];
  if (!v) throw new Error(`Unknown camera view '${name}'. Define it in --camera-config.`);
  const pos = new THREE.Vector3(...v.position.map(Number));
  const target = new THREE.Vector3(...v.target.map(Number));
  camera.position.copy(pos);
  controls.target.copy(target);
  camera.fov = Number(v.fov ?? 42);
  camera.near = Number(v.near ?? 0.01);
  camera.far = Number(v.far ?? 1000);
  camera.updateProjectionMatrix();
  camera.lookAt(target);
  controls.update();
}
function availableRecordViews() { return model.trajectory?.record_views || []; }
function setupRecordingUI() {
  const views = availableRecordViews(); ui.recordViews.replaceChildren();
  if (!views.length) { ui.recordButton.disabled = true; ui.recordStatus.textContent = 'No recording views enabled. Use --record-views front top hands or names from --camera-config'; return; }
  for (const name of views) {
    const label = document.createElement('label'); const cb = document.createElement('input'); cb.type = 'checkbox'; cb.value = name; cb.checked = true;
    const view = VIEW_PRESETS[name];
    label.append(cb, document.createTextNode(` ${view?.label || name}`)); ui.recordViews.append(label);
  }
  ui.recordButton.disabled = false; ui.recordStatus.textContent = `${views.length} recording view(s) enabled`;
}
function selectedRecordViews() { return [...ui.recordViews.querySelectorAll('input:checked')].map(x => x.value); }
function makeRecordCanvas() {
  const w = 1280, h = 720; const c = document.createElement('canvas'); c.width = w; c.height = h; model.recordCanvas = c; model.recordCtx = c.getContext('2d'); return c;
}
function drawRecordingFrame() {
  if (!model.recording || !model.recordCtx) return;
  const ctx = model.recordCtx, c = model.recordCanvas; ctx.fillStyle = '#07101b'; ctx.fillRect(0,0,c.width,c.height);
  const src = renderer.domElement, scale = Math.min((c.width-40)/src.width, (c.height-90)/src.height); const dw = src.width*scale, dh=src.height*scale;
  ctx.drawImage(src, (c.width-dw)/2, 55+(c.height-90-dh)/2, dw, dh);
  ctx.fillStyle = '#edf5ff'; ctx.font = '600 22px system-ui'; ctx.fillText(`Origami inference · ${model.recordingViewLabel || ''}`, 24, 32);
  ctx.font = '14px system-ui'; ctx.fillStyle = '#a8bbd2'; ctx.fillText(`step ${model.step+1}/${totalSteps()} · ${episodeRelativeTime().toFixed(3)} s`, 24, c.height-20);
}
function startRecording(viewName) {
  if (!window.MediaRecorder || !renderer.domElement.captureStream) throw new Error('This browser does not support MediaRecorder canvas capture.');
  setCameraPreset(viewName); stopPlaying(); setStep(0, {seekVideo:true});
  makeRecordCanvas(); model.recording = true; model.recordingViewLabel = VIEW_PRESETS[viewName]?.label || viewName;
  const stream = model.recordCanvas.captureStream(Number(model.trajectory.control_hz || 30)); model.recordStream = stream;
  const mime = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(x => MediaRecorder.isTypeSupported(x)) || '';
  model.recorder = new MediaRecorder(stream, mime ? {mimeType:mime} : undefined); const chunks=[];
  model.recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  model.recorder.onstop = () => {
    model.recording = false; const blob = new Blob(chunks, {type:model.recorder.mimeType || 'video/webm'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=`origami_${viewName}.webm`; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  };
  model.recorder.start(250); ui.recordStatus.textContent = `Recording ${VIEW_PRESETS[viewName].label}…`; setTimeout(() => { if (model.recorder?.state === 'recording') model.recorder.stop(); }, Math.ceil((totalSteps()/Number(model.trajectory.control_hz||30))*1000)+500);
  startPlaying();
}
ui.recordButton.addEventListener('click', async () => {
  if (model.recording) return;
  const views = selectedRecordViews(); if (!views.length) { ui.recordStatus.textContent = 'Select at least one view.'; return; }
  ui.recordButton.disabled = true;
  try {
    for (const view of views) { await new Promise((resolve,reject)=>{ model._recordResolve=resolve; model._recordReject=reject; try { startRecording(view); } catch(e){reject(e);} }); }
  } catch (e) { ui.recordStatus.textContent = `Recording failed: ${e.message}`; }
  ui.recordButton.disabled = false;
});
// Resolve each recording sequentially when its recorder finishes.
const _origStart = startRecording;
// Patch the recorder stop callback by wrapping via a lightweight completion hook.
const recordButtonHandler = ui.recordButton;
// The sequential loop above waits on this event.
setInterval(() => { if (!model.recording && model._recordResolve) { const r=model._recordResolve; model._recordResolve=null; r(); } }, 100);

async function init() {
  initScene();
  try {
    model.config = await api('/api/robot/config'); if (!model.config.urdf_available) throw new Error(model.config.urdf_error || 'URDF unavailable');
    model.predictionRobot = await loadUrdf(model.config, 0.86, 'prediction', COLORS.prediction);
    model.gtRobot = await loadUrdf(model.config, 0.42, 'GT', COLORS.gt); model.gtRobot.visible = false;
    badge(ui.urdfBadge, model.config.urdf_limits_loaded ? 'URDF ready · dual robot' : 'URDF ready · limits unavailable', model.config.urdf_limits_loaded ? 'ok' : 'warn'); fitRobots();
  } catch (error) { badge(ui.urdfBadge, error.message, 'bad'); }
  try { model.trajectory = await api('/api/trajectory'); configureCameraViews(model.trajectory); renderMetrics(); configureVideos(model.trajectory); setupRecordingUI(); ui.timeline.disabled=false; setStep(0); }
  catch (error) { badge(ui.predictionBadge, error.message, 'bad'); }
}
init();
