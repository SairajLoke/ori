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
};
const model = {
  config: null, trajectory: null, predictionRobot: null, gtRobot: null,
  step: 0, playing: false, timer: null,
};
let scene, camera, renderer, controls;

async function api(path) {
  const response = await fetch(path, {cache: 'no-store'});
  const value = await response.json();
  if (!response.ok || value?.ok === false) throw new Error(String(value?.error || `HTTP ${response.status}`));
  return value;
}

function badge(node, text, tone = '') {
  node.textContent = String(text);
  node.className = `badge${tone ? ` ${tone}` : ''}`;
}

function metric(label, value) {
  const row = document.createElement('div'); row.className = 'metric';
  const a = document.createElement('span'); const b = document.createElement('span');
  a.textContent = label; b.textContent = value == null ? '—' : String(value); row.append(a, b); return row;
}

function totalSteps() {
  const p = model.trajectory?.prediction?.length || 0;
  const g = model.trajectory?.ground_truth_action?.length || 0;
  return Math.max(p, g);
}

function currentPrediction() { return model.trajectory?.prediction?.[model.step] || null; }
function currentGT() { return model.trajectory?.ground_truth_action?.[model.step] || null; }
function currentTime() {
  const ts = model.trajectory?.timestamps;
  if (ts && ts.length) {
    if (model.step < ts.length) return Number(ts[model.step]);
    const fps = Number(model.trajectory.dataset_fps || model.trajectory.control_hz || 30);
    return Number(ts[ts.length - 1]) + (model.step - ts.length + 1) / fps;
  }
  return model.step / Number(model.trajectory?.control_hz || 30);
}

function renderMetrics() {
  const t = model.trajectory, v = t.validation;
  const gt = t.ground_truth_action;
  const pLen = t.prediction.length, gLen = gt?.length || 0;
  const overlap = Math.min(pLen, gLen);
  const errs = [];
  for (let i = 0; i < overlap; i++) {
    let sum = 0, max = 0;
    for (let j = 0; j < 65; j++) {
      const e = Math.abs(Number(t.prediction[i][j]) - Number(gt[i][j]));
      sum += e; if (e > max) max = e;
    }
    errs.push({mean: sum / 65, max});
  }
  const meanError = errs.length ? errs.reduce((a,b) => a + b.mean, 0) / errs.length : null;
  const maxError = errs.length ? Math.max(...errs.map(x => x.max)) : null;
  const rows = [
    ['Prediction', `${pLen} steps`], ['GT action', gt ? `${gLen} steps` : 'not loaded'],
    ['Overlap', overlap ? `${overlap} steps` : '—'], ['Control Hz', t.control_hz],
    ['Pred stages', t.stage_count], ['Action horizon', t.action_horizon],
    ['Mean |pred-GT|', meanError == null ? '—' : `${meanError.toFixed(5)} rad`],
    ['Max |pred-GT|', maxError == null ? '—' : `${maxError.toFixed(5)} rad`],
    ['Prediction limits', t.compatible ? 'OK' : 'violations'],
  ];
  ui.metrics.replaceChildren(...rows.map(([a,b]) => metric(a,b)));
  badge(ui.predictionBadge, `${pLen} predicted steps · ${t.stage_count} stages`, 'ok');
  badge(ui.gtBadge, gt ? `${gLen} GT action steps` : 'GT not loaded', gt ? 'ok' : 'warn');
}

function renderState() {
  const pred = currentPrediction(), gt = currentGT();
  if (!pred && !gt) return;
  const limits = model.config?.limits || {}, names = model.trajectory.metadata.joint_names;
  const fragment = document.createDocumentFragment();
  names.forEach((name, i) => {
    const tr = document.createElement('tr'), limit = limits[name];
    const range = limit ? `[${limit.lower.toFixed(3)}, ${limit.upper.toFixed(3)}]` : '—';
    const vel = limit ? `${limit.velocity.toFixed(3)} rad/s` : '—';
    const pv = pred ? Number(pred[i]).toFixed(6) : '—';
    const gv = gt ? Number(gt[i]).toFixed(6) : '—';
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

function syncVideo(video, time, segment, statusNode) {
  if (!video.src || !segment) return;

  // LeRobot v3.0 stores multiple episodes back-to-back in one MP4.
  // `from_timestamp` is the episode's start inside that shared MP4.
  const episodeStart = Number(segment.from_timestamp || 0);
  const episodeEnd = segment.to_timestamp == null ? Infinity : Number(segment.to_timestamp);
  const datasetEpisodeTime = Math.max(0, Number(time) - Number(model.trajectory?.episode_time_origin || 0));
  let videoTime = episodeStart + datasetEpisodeTime;

  if (datasetEpisodeTime > (episodeEnd - episodeStart) + 0.05) {
    statusNode.textContent = `Episode video ends at ${(episodeEnd - episodeStart).toFixed(3)} s`;
    return;
  }

  if (Number.isFinite(video.duration)) videoTime = Math.min(videoTime, video.duration);
  try {
    video.currentTime = Math.max(0, videoTime);
    statusNode.textContent = `Synced · episode ${datasetEpisodeTime.toFixed(3)} s · file ${videoTime.toFixed(3)} s`;
  } catch {}
}

function updatePlayer() {
  const total = totalSteps(), step = model.step;
  const stage = Math.floor(step / model.trajectory.action_horizon), horizon = step % model.trajectory.action_horizon;
  ui.timeline.max = String(Math.max(0, total - 1)); ui.timeline.value = String(step);
  ui.stepText.textContent = `Step ${step + 1} / ${total}`;
  ui.stageText.textContent = `Stage ${stage + 1} / ${model.trajectory.stage_count}`;
  ui.horizonText.textContent = `Horizon ${horizon + 1} / ${model.trajectory.action_horizon}`;
  ui.timeText.textContent = `Time ${currentTime().toFixed(3)} s`;
  ui.stepBack.disabled = step <= 0; ui.stepForward.disabled = step >= total - 1; ui.playPause.disabled = total <= 1;
  renderRobots(); renderState(); renderWarnings();
  const segments = model.trajectory.video_segments || {};
  syncVideo(ui.headLeftVideo, currentTime(), segments.head_left, ui.headLeftStatus);
  syncVideo(ui.wristLeftVideo, currentTime(), segments.wrist_left, ui.wristLeftStatus);
}

function setStep(step) {
  const total = totalSteps(); model.step = Math.max(0, Math.min(Number(step) || 0, total - 1)); updatePlayer();
}
function stopPlaying() { model.playing = false; ui.playPause.textContent = '▶'; if (model.timer !== null) clearInterval(model.timer); model.timer = null; }
function startPlaying() {
  stopPlaying(); const total = totalSteps(); if (total <= 1) return;
  model.playing = true; ui.playPause.textContent = '❚❚';
  const delay = Math.max(20, 1000 / model.trajectory.control_hz);
  model.timer = setInterval(() => { if (model.step >= total - 1) { stopPlaying(); return; } setStep(model.step + 1); }, delay);
}
ui.playPause.addEventListener('click', () => model.playing ? stopPlaying() : startPlaying());
ui.stepBack.addEventListener('click', () => { stopPlaying(); setStep(model.step - 1); });
ui.stepForward.addEventListener('click', () => { stopPlaying(); setStep(model.step + 1); });
ui.timeline.addEventListener('input', e => { stopPlaying(); setStep(e.target.value); });
ui.showPrediction.addEventListener('change', () => { if (model.predictionRobot) model.predictionRobot.visible = ui.showPrediction.checked; });
ui.showGT.addEventListener('change', () => { if (model.gtRobot) model.gtRobot.visible = ui.showGT.checked; });

function initScene() {
  scene = new THREE.Scene(); scene.background = new THREE.Color(0x07101b);
  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000); camera.position.set(2.6, 1.8, 3.2);
  renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.shadowMap.enabled = true;
  ui.robotView.append(renderer.domElement); controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.target.set(0, 0.8, 0);
  scene.add(new THREE.HemisphereLight(0xd7e8ff, 0x26313d, 2.4));
  const key = new THREE.DirectionalLight(0xffffff, 3); key.position.set(4, 7, 5); scene.add(key);
  scene.add(new THREE.GridHelper(12, 24, 0x36506f, 0x1a2a3d));
  const resize = () => { const width = Math.max(1, ui.robotView.clientWidth), height = Math.max(1, ui.robotView.clientHeight); renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix(); };
  new ResizeObserver(resize).observe(ui.robotView); resize(); requestAnimationFrame(renderFrame);
}
function renderFrame() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(renderFrame); }

function fitRobots() {
  const objects = [model.predictionRobot, model.gtRobot].filter(Boolean);
  if (!objects.length) return;
  const box = new THREE.Box3(); objects.forEach(o => box.expandByObject(o)); if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3()), radius = Math.max(size.x, size.y, size.z, 0.5);
  controls.target.copy(center); camera.position.copy(center).add(new THREE.Vector3(radius * 1.8, radius * 1.15, radius * 2.1)); camera.near = Math.max(radius / 1000, 0.005); camera.far = Math.max(radius * 100, 100); camera.updateProjectionMatrix();
}
function styleRobot(robot, opacity) {
  robot.traverse(object => {
    if (!object.isMesh) return;
    object.castShadow = true; object.receiveShadow = true;
    if (object.material) {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach(mat => { mat.transparent = opacity < 1; mat.opacity = opacity; if (opacity < 1) mat.depthWrite = false; });
    }
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
  renderRobot(model.predictionRobot, currentPrediction());
  renderRobot(model.gtRobot, currentGT());
  if (model.predictionRobot) model.predictionRobot.visible = ui.showPrediction.checked && !!currentPrediction();
  if (model.gtRobot) model.gtRobot.visible = ui.showGT.checked && !!currentGT();
}

function loadUrdf(config, opacity, label) {
  return new Promise((resolve, reject) => {
    const url = new URL(config.urdf_url, window.location.href).href, manager = new THREE.LoadingManager();
    manager.onStart = () => badge(ui.meshBadge, `Loading ${label} mesh…`, 'warn');
    manager.onLoad = () => badge(ui.meshBadge, 'Meshes ready', 'ok');
    manager.onError = () => badge(ui.meshBadge, 'Some meshes failed', 'bad');
    const loader = new URDFLoader(manager), urdfBase = new URL('.', url);
    loader.packages = urdfBase.pathname.endsWith('/urdf/') ? new URL('../', urdfBase).href : urdfBase.href;
    loader.load(url, robot => { robot.rotation.x = -Math.PI / 2; styleRobot(robot, opacity); scene.add(robot); resolve(robot); }, undefined, error => reject(error instanceof Error ? error : new Error('URDF loading failed')));
  });
}

function configureVideos(t) {
  const vids = [['head_left', ui.headLeftVideo, ui.headLeftStatus], ['wrist_left', ui.wristLeftVideo, ui.wristLeftStatus]];
  for (const [key, video, status] of vids) {
    const url = t.video_urls?.[key];
    const segment = t.video_segments?.[key];
    if (!url || !segment) { status.textContent = 'Not configured'; continue; }
    video.src = url; video.load();
    const start = Number(segment.from_timestamp || 0);
    const end = segment.to_timestamp == null ? null : Number(segment.to_timestamp);
    status.textContent = end == null
      ? 'Loading…'
      : `Loading · episode ${Math.max(0, end - start).toFixed(3)} s`;
    video.addEventListener('loadedmetadata', () => {
      status.textContent = end == null
        ? `Ready · ${video.duration.toFixed(3)} s`
        : `Ready · shared MP4 ${start.toFixed(3)}–${end.toFixed(3)} s`;
    }, {once:true});
  }
}

async function init() {
  initScene();
  try {
    model.config = await api('/api/robot/config');
    if (!model.config.urdf_available) throw new Error(model.config.urdf_error || 'URDF unavailable');
    model.predictionRobot = await loadUrdf(model.config, 1.0, 'prediction');
    model.gtRobot = await loadUrdf(model.config, 0.28, 'GT');
    model.gtRobot.visible = false;
    badge(ui.urdfBadge, model.config.urdf_limits_loaded ? 'URDF ready · dual robot' : 'URDF ready · limits unavailable', model.config.urdf_limits_loaded ? 'ok' : 'warn');
    fitRobots();
  } catch (error) { badge(ui.urdfBadge, error.message, 'bad'); }

  try {
    model.trajectory = await api('/api/trajectory');
    renderMetrics(); configureVideos(model.trajectory); ui.timeline.disabled = false; setStep(0);
  } catch (error) { badge(ui.predictionBadge, error.message, 'bad'); }
}
init();
