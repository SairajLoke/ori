import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import URDFLoader from './vendor/URDFLoader.js';

const $ = id => document.getElementById(id);
const ui = {
  predictionBadge: $('predictionBadge'), urdfBadge: $('urdfBadge'),
  meshBadge: $('meshBadge'), compatBadge: $('compatBadge'),
  metrics: $('metrics'), warnings: $('warnings'), stateRows: $('stateRows'),
  robotView: $('robotView'), timeline: $('timeline'),
  stepText: $('stepText'), stageText: $('stageText'), horizonText: $('horizonText'),
  playPause: $('playPause'), stepBack: $('stepBack'), stepForward: $('stepForward'),
  statusText: $('statusText'),
};
const model = {
  config: null, trajectory: null, robot: null, step: 0, playing: false, timer: null,
};
let scene, camera, renderer, controls;

async function api(path) {
  const response = await fetch(path, {cache: 'no-store'});
  const value = await response.json();
  if (!response.ok || value?.ok === false) {
    throw new Error(String(value?.error || `HTTP ${response.status}`));
  }
  return value;
}

function badge(node, text, tone = '') {
  node.textContent = String(text);
  node.className = `badge${tone ? ` ${tone}` : ''}`;
}

function metric(label, value) {
  const row = document.createElement('div');
  row.className = 'metric';
  const a = document.createElement('span');
  const b = document.createElement('span');
  a.textContent = label; b.textContent = value == null ? '—' : String(value);
  row.append(a, b);
  return row;
}

function renderMetrics() {
  const t = model.trajectory;
  const v = t.validation;
  const rows = [
    ['Shape', JSON.stringify(t.action_shape)],
    ['Stages', t.stage_count],
    ['Action horizon', t.action_horizon],
    ['Control Hz', t.control_hz],
    ['Range', `${t.action_min.toFixed(4)} … ${t.action_max.toFixed(4)}`],
    ['Compatible', t.compatible ? 'YES' : 'NO'],
    ['Validation', v.validation_level],
    ['Steps with warnings', v.steps_with_violations ?? '—'],
    ['Initial state', v.initial_state_provided ? 'supplied' : 'not supplied'],
  ];
  ui.metrics.replaceChildren(...rows.map(([a,b]) => metric(a,b)));
  badge(ui.compatBadge, t.compatible ? 'All limits OK' : 'Limit/velocity violations', t.compatible ? 'ok' : 'bad');
}

function selectedState() {
  return model.trajectory?.prediction?.[model.step] || null;
}

function renderState() {
  const state = selectedState();
  if (!state) return;
  const limits = model.config?.limits || {};
  const names = model.trajectory.metadata.joint_names;
  const fragment = document.createDocumentFragment();

  names.forEach((name, i) => {
    const tr = document.createElement('tr');
    const limit = limits[name];
    const range = limit ? `[${limit.lower.toFixed(3)}, ${limit.upper.toFixed(3)}]` : '—';
    const vel = limit ? `${limit.velocity.toFixed(3)} rad/s` : '—';
    for (const text of [i, name, Number(state[i]).toFixed(6), range, vel]) {
      const td = document.createElement('td');
      td.textContent = String(text);
      tr.append(td);
    }
    fragment.append(tr);
  });
  ui.stateRows.replaceChildren(fragment);

  ui.statusText.textContent = JSON.stringify({
    step: model.step,
    stage: Math.floor(model.step / model.trajectory.action_horizon),
    horizon_step: model.step % model.trajectory.action_horizon,
    state: state,
  }, null, 2);
}

function renderWarnings() {
  const report = model.trajectory.validation.step_reports?.[model.step];
  const issues = report?.violations || [];
  if (!issues.length) {
    const li = document.createElement('li');
    li.textContent = 'None';
    ui.warnings.replaceChildren(li);
    return;
  }
  const fragment = document.createDocumentFragment();
  issues.slice(0, 100).forEach(issue => {
    const li = document.createElement('li');
    li.textContent = `${issue.joint_name}: ${issue.type}`;
    fragment.append(li);
  });
  ui.warnings.replaceChildren(fragment);
}

function updatePlayer() {
  const total = model.trajectory.prediction.length;
  const step = model.step;
  const stage = Math.floor(step / model.trajectory.action_horizon);
  const horizon = step % model.trajectory.action_horizon;

  ui.timeline.max = String(Math.max(0, total - 1));
  ui.timeline.value = String(step);
  ui.stepText.textContent = `Step ${step + 1} / ${total}`;
  ui.stageText.textContent = `Stage ${stage + 1} / ${model.trajectory.stage_count}`;
  ui.horizonText.textContent = `Horizon ${horizon + 1} / ${model.trajectory.action_horizon}`;

  ui.stepBack.disabled = step <= 0;
  ui.stepForward.disabled = step >= total - 1;
  ui.playPause.disabled = total <= 1;

  renderRobot(selectedState());
  renderState();
  renderWarnings();
}

function setStep(step) {
  const total = model.trajectory.prediction.length;
  model.step = Math.max(0, Math.min(Number(step) || 0, total - 1));
  updatePlayer();
}

function stopPlaying() {
  model.playing = false;
  ui.playPause.textContent = '▶';
  if (model.timer !== null) clearInterval(model.timer);
  model.timer = null;
}

function startPlaying() {
  stopPlaying();
  const total = model.trajectory.prediction.length;
  if (total <= 1) return;
  model.playing = true;
  ui.playPause.textContent = '❚❚';
  const delay = Math.max(20, 1000 / model.trajectory.control_hz);
  model.timer = setInterval(() => {
    if (model.step >= total - 1) { stopPlaying(); return; }
    setStep(model.step + 1);
  }, delay);
}

ui.playPause.addEventListener('click', () => model.playing ? stopPlaying() : startPlaying());
ui.stepBack.addEventListener('click', () => { stopPlaying(); setStep(model.step - 1); });
ui.stepForward.addEventListener('click', () => { stopPlaying(); setStep(model.step + 1); });
ui.timeline.addEventListener('input', e => { stopPlaying(); setStep(e.target.value); });

function initScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x07101b);
  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);
  camera.position.set(2.6, 1.8, 3.2);
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  ui.robotView.append(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0.8, 0);
  scene.add(new THREE.HemisphereLight(0xd7e8ff, 0x26313d, 2.4));
  const key = new THREE.DirectionalLight(0xffffff, 3);
  key.position.set(4, 7, 5);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x72aaff, 1.4);
  rim.position.set(-5, 3, -4);
  scene.add(rim);
  scene.add(new THREE.GridHelper(12, 24, 0x36506f, 0x1a2a3d));
  const resize = () => {
    const width = Math.max(1, ui.robotView.clientWidth);
    const height = Math.max(1, ui.robotView.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(ui.robotView);
  resize();
  requestAnimationFrame(renderFrame);
}

function renderFrame() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(renderFrame);
}

function fitRobot(robot) {
  const box = new THREE.Box3().setFromObject(robot);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 0.5);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(radius * 1.8, radius * 1.15, radius * 2.1));
  camera.near = Math.max(radius / 1000, 0.005);
  camera.far = Math.max(radius * 100, 100);
  camera.updateProjectionMatrix();
}

function renderRobot(state) {
  if (!model.robot || !state) return;
  for (const entry of model.config?.joint_map || []) {
    const value = Number(state[entry.index]);
    const joint = model.robot.joints?.[entry.name];
    if (!joint || !Number.isFinite(value) || typeof joint.setJointValue !== 'function') continue;
    try { joint.setJointValue(value); } catch {}
  }
}

function loadUrdfModel(config) {
  return new Promise((resolve, reject) => {
    const url = new URL(config.urdf_url, window.location.href).href;
    const manager = new THREE.LoadingManager();
    manager.onStart = () => badge(ui.meshBadge, 'Loading mesh…', 'warn');
    manager.onProgress = (_url, loaded, total) => badge(ui.meshBadge, `Mesh ${loaded}/${total}`, 'warn');
    manager.onLoad = () => badge(ui.meshBadge, 'Mesh ready', 'ok');
    manager.onError = () => badge(ui.meshBadge, 'Some meshes failed', 'bad');

    const loader = new URDFLoader(manager);
    const urdfBase = new URL('.', url);
    loader.packages = urdfBase.pathname.endsWith('/urdf/')
      ? new URL('../', urdfBase).href
      : urdfBase.href;

    loader.load(url, robot => {
      model.robot = robot;
      robot.rotation.x = -Math.PI / 2;
      robot.traverse(object => {
        if (object.isMesh) { object.castShadow = true; object.receiveShadow = true; }
      });
      scene.add(robot);
      fitRobot(robot);
      badge(
        ui.urdfBadge,
        config.urdf_limits_loaded
          ? `URDF ready · ${Object.keys(robot.joints || {}).length} joints`
          : 'URDF ready · limits unavailable',
        config.urdf_limits_loaded ? 'ok' : 'warn',
      );
      renderRobot(selectedState());
      resolve();
    }, undefined, error => reject(error instanceof Error ? error : new Error('URDF loading failed')));
  });
}

async function init() {
  initScene();
  try {
    model.config = await api('/api/robot/config');
    if (!model.config.urdf_available) throw new Error(model.config.urdf_error || 'URDF unavailable');
    await loadUrdfModel(model.config);
  } catch (error) {
    badge(ui.urdfBadge, error.message, 'bad');
  }

  try {
    model.trajectory = await api('/api/trajectory');
    const t = model.trajectory;
    badge(ui.predictionBadge, `${t.action_shape[0]} steps · ${t.stage_count} stages`, 'ok');
    renderMetrics();
    ui.timeline.disabled = false;
    setStep(0);
  } catch (error) {
    badge(ui.predictionBadge, error.message, 'bad');
  }
}

init();
