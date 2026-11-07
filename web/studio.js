// The shared look of every 3D view: a clean robotics lab (sealed concrete floor, safety markings, no furniture),
// studio lighting, one black-and-white material scheme for every robot, and the post pipeline (MSAA, ambient
// occlusion, tone mapping, SMAA). Meshes and poses are not touched here: only materials and the set.
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { GTAOPass } from "three/addons/postprocessing/GTAOPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { SMAAPass } from "three/addons/postprocessing/SMAAPass.js";

export const HAZE = new THREE.Color(0xe7e8ea); // the page's card colour: the floor fades into it at the horizon

export function setupRenderer(renderer) {
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping; // keeps painted colours true, like a product render
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap; // PCF honours shadow.radius, so the key light casts a soft edge
}

export function makeEnvironment(renderer) {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const tex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  return tex;
}

// ---------------------------------------------------------------- post pipeline
// 4x MSAA scene target, ground-truth ambient occlusion (the soft darkening where parts meet the floor and each
// other), filmic-neutral tone mapping, then SMAA on the finished image for the shader aliasing MSAA cannot see.
export function makeStage(renderer, scene, camera) {
  const v = renderer.getSize(new THREE.Vector2());
  const w = Math.max(2, v.x), h = Math.max(2, v.y);
  const pr = renderer.getPixelRatio();
  const rt = new THREE.WebGLRenderTarget(w * pr, h * pr, { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, rt);
  composer.setPixelRatio(pr);
  composer.setSize(w, h);
  composer.addPass(new RenderPass(scene, camera));
  const ao = new GTAOPass(scene, camera, w * pr, h * pr);
  ao.updateGtaoMaterial({ radius: 0.07, distanceExponent: 1.6, thickness: 1.2, scale: 1.25, samples: 16, distanceFallOff: 1, screenSpaceRadius: false });
  ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 5, radiusExponent: 1, rings: 2, samples: 16 });
  ao.blendIntensity = 0.85;
  composer.addPass(ao);
  composer.addPass(new OutputPass());
  composer.addPass(new SMAAPass(w * pr, h * pr));
  return {
    composer,
    render: () => composer.render(),
    setSize(width, height) {
      composer.setPixelRatio(renderer.getPixelRatio());
      composer.setSize(width, height);
    },
    dispose() { composer.dispose(); },
  };
}

// ---------------------------------------------------------------- lighting
// Three-point studio rig: a warm soft key that casts the shadows, a cool sky fill and a rim from behind, on top of
// the image-based light of the environment map.
export function addStudioLights(scene, { target = [0.2, 0, 0], extent = 1.1, reach = 1 } = {}) {
  const t = new THREE.Vector3(...target);
  const key = new THREE.DirectionalLight(0xfff1e0, 2.9);
  key.position.set(t.x + 0.95 * reach, t.y - 0.8 * reach, 1.75 * reach);
  key.target.position.copy(t);
  key.castShadow = true;
  key.shadow.mapSize.set(4096, 4096);
  Object.assign(key.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near: 0.4, far: 6 });
  key.shadow.radius = 5;
  key.shadow.blurSamples = 20;
  key.shadow.bias = -0.0002;
  key.shadow.normalBias = 0.0012;
  scene.add(key, key.target);
  scene.add(new THREE.HemisphereLight(0xf2f5ff, 0xc9c8c4, 0.5));
  const rim = new THREE.DirectionalLight(0xdfe8ff, 1.0);
  rim.position.set(t.x - 1.1 * reach, t.y + 1.0 * reach, 1.1 * reach);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(0xfff8f0, 0.55);
  fill.position.set(t.x + 0.6 * reach, t.y + 1.2 * reach, 0.8 * reach);
  scene.add(fill);
  return { key, rim, fill };
}

// ---------------------------------------------------------------- the floor
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// sealed concrete: broad cloudy mottling, fine aggregate speckle, tileable (2 m per tile)
function concreteTexture(size = 2048) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d", { willReadFrequently: true });
  const r = rng(11);
  g.fillStyle = "#93969b";
  g.fillRect(0, 0, size, size);
  for (let i = 0; i < 420; i++) {
    const x = r() * size, y = r() * size, rad = 60 + r() * 340;
    const light = r() < 0.5;
    const a = 0.01 + r() * 0.025;
    for (const ox of [-size, 0, size]) for (const oy of [-size, 0, size]) {
      const cx = x + ox, cy = y + oy;
      if (cx + rad < 0 || cy + rad < 0 || cx - rad > size || cy - rad > size) continue;
      const gr = g.createRadialGradient(cx, cy, 0, cx, cy, rad);
      const col = light ? "255,255,255" : "50,54,60";
      gr.addColorStop(0, `rgba(${col},${a})`);
      gr.addColorStop(1, `rgba(${col},0)`);
      g.fillStyle = gr;
      g.fillRect(cx - rad, cy - rad, rad * 2, rad * 2);
    }
  }
  const d = g.getImageData(0, 0, size, size);
  const px = d.data;
  for (let i = 0; i < px.length; i += 4) {
    let n = (r() - 0.5) * 11;
    const q = r();
    if (q > 0.994) n += 38; else if (q < 0.006) n -= 34; // aggregate chips and pores
    px[i] += n; px[i + 1] += n; px[i + 2] += n * 1.03;
  }
  g.putImageData(d, 0, 0);
  return c;
}
let _floorTex = null;
function floorTexture(renderer) {
  if (_floorTex) return _floorTex;
  const t = new THREE.CanvasTexture(concreteTexture());
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.anisotropy = renderer?.capabilities.getMaxAnisotropy() ?? 8;
  t.repeat.set(20, 20);
  return (_floorTex = t);
}

const PAINT = (hex) => new THREE.MeshStandardMaterial({ color: hex, roughness: 0.55, metalness: 0, envMapIntensity: 0.5 });
const SAFETY = PAINT(0xdcb52b);
const JOINT = new THREE.MeshStandardMaterial({ color: 0x5d6065, roughness: 0.9, envMapIntensity: 0.2 });

function decal(w, h, mat, x, y, z = 0.0007, rot = 0) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
  m.position.set(x, y, z);
  m.rotation.z = rot;
  m.receiveShadow = true;
  return m;
}
export const tape = (g, x0, y0, x1, y1, w = 0.035, mat = SAFETY) => {
  const len = Math.hypot(x1 - x0, y1 - y0);
  g.add(decal(len, w, mat, (x0 + x1) / 2, (y0 + y1) / 2, 0.0007, Math.atan2(y1 - y0, x1 - x0)));
};
// L-shaped corner brackets of a taped work cell, so the cell reads without a hard fence
export function cellBrackets(g, [x0, y0, x1, y1], arm = 0.2, w = 0.035, mat = SAFETY) {
  for (const [cx, cy, sx, sy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x1, y1, -1, -1], [x0, y1, 1, -1]]) {
    tape(g, cx - sx * w / 2, cy, cx + sx * arm, cy, w, mat);
    tape(g, cx, cy - sy * w / 2, cx, cy + sy * arm, w, mat);
  }
}
function hazardTexture() {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 32;
  const g = c.getContext("2d");
  g.fillStyle = "#dcb52b"; g.fillRect(0, 0, 256, 32);
  g.fillStyle = "#1d1e21";
  for (let x = -32; x < 288; x += 32) { g.beginPath(); g.moveTo(x, 32); g.lineTo(x + 16, 32); g.lineTo(x + 32, 0); g.lineTo(x + 16, 0); g.fill(); }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  return t;
}
let _hazard = null;
export function hazardStrip(g, x0, y0, x1, y1, w = 0.04) {
  const len = Math.hypot(x1 - x0, y1 - y0);
  _hazard ||= new THREE.MeshStandardMaterial({ map: hazardTexture(), roughness: 0.6, envMapIntensity: 0.4 });
  const mat = _hazard.clone();
  mat.map = _hazard.map.clone();
  mat.map.needsUpdate = true;
  mat.map.repeat.set(len / (w * 8), 1);
  g.add(decal(len, w, mat, (x0 + x1) / 2, (y0 + y1) / 2, 0.0007, Math.atan2(y1 - y0, x1 - x0)));
}
// stencilled floor text (cell number and what stands in it)
export function floorLabel(g, text, x, y, rot = 0, h = 0.045, color = "rgba(246,247,248,0.92)") {
  const c = document.createElement("canvas");
  c.width = 1024; c.height = 128;
  const k = c.getContext("2d");
  k.font = "700 84px Figtree, system-ui, sans-serif";
  k.fillStyle = color;
  k.textBaseline = "middle";
  k.fillText(text, 4, 66);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  const m = new THREE.Mesh(
    new THREE.PlaneGeometry(h * 8, h),
    new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, toneMapped: false, opacity: 0.85 }),
  );
  m.geometry.translate(h * 4, 0, 0); // left-aligned at (x, y)
  m.position.set(x, y, 0.0009);
  m.rotation.z = rot;
  m.renderOrder = 2;
  g.add(m);
  return m;
}

// The lab: sealed concrete from here to the haze, expansion joints, and (optionally) taped cell markings.
export function buildLab(scene, renderer, { cell = [-0.3, -0.55, 0.7, 0.55], joints = true } = {}) {
  const g = new THREE.Group();
  const tex = floorTexture(renderer);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshPhysicalMaterial({
      map: tex, bumpMap: tex, bumpScale: 0.35, color: 0xffffff, roughness: 0.46, metalness: 0,
      clearcoat: 0.55, clearcoatRoughness: 0.32, envMapIntensity: 0.55,
      transparent: true, opacity: 0.93, // a faint mirror of the cell shows through the sealed surface
    }),
  );
  floor.receiveShadow = true;
  floor.renderOrder = 0;
  g.add(floor);
  if (joints) {
    for (const x of [-0.95, 1.45]) g.add(decal(0.006, 12, JOINT, x, 0, 0.0004));
    for (const y of [-1.25, 1.25]) g.add(decal(12, 0.006, JOINT, 0, y, 0.0004));
  }
  if (cell) cellBrackets(g, cell);
  scene.add(g);
  return g;
}

// Floor-mounted base plate under an arm (brushed aluminium, four hex bolts). Its top sits 6 mm above the floor and
// the arm's own base is a little inside it, so nothing floats and no table is needed.
const ALU = new THREE.MeshStandardMaterial({ color: 0xc9ccd1, roughness: 0.38, metalness: 0.95, envMapIntensity: 1.0 });
const BOLT = new THREE.MeshStandardMaterial({ color: 0x34363b, roughness: 0.4, metalness: 0.9, envMapIntensity: 0.9 });
export function basePlate(size = 0.23) {
  const g = new THREE.Group();
  const t = 0.012;
  const plate = new THREE.Mesh(new RoundedBoxGeometry(size, size, t, 4, 0.0035), ALU);
  plate.castShadow = plate.receiveShadow = true;
  g.add(plate); // spans z -6..+6 mm, bottom buried in the floor
  const inset = size / 2 - 0.021;
  for (const [x, y] of [[-inset, -inset], [inset, -inset], [inset, inset], [-inset, inset]]) {
    const b = new THREE.Mesh(new THREE.CylinderGeometry(0.0062, 0.0062, 0.004, 6), BOLT);
    b.rotation.x = Math.PI / 2;
    b.position.set(x, y, 0.008);
    b.castShadow = true;
    g.add(b);
  }
  return g;
}

// ---------------------------------------------------------------- the robots: black and white
// White printed shells, black servos and joints, a little graphite steel. One scheme for every arm and hand: the
// meshes are the real sourced ones, only the materials are ours.
export const MAT = {
  shell: new THREE.MeshPhysicalMaterial({ color: 0xf1f2f3, roughness: 0.46, metalness: 0, clearcoat: 0.28, clearcoatRoughness: 0.34, sheen: 0.2, sheenRoughness: 0.6, sheenColor: 0xffffff, envMapIntensity: 0.85 }),
  servo: new THREE.MeshPhysicalMaterial({ color: 0x141518, roughness: 0.36, metalness: 0.2, clearcoat: 0.5, clearcoatRoughness: 0.3, envMapIntensity: 1.0 }),
  steel: new THREE.MeshStandardMaterial({ color: 0x70747b, roughness: 0.34, metalness: 1.0, envMapIntensity: 1.0 }),
};
const SERVO_NAME = /motor|sts3215|horn|prop|gripper_bar|finger|_7_gripper|_6_wrist|_2_shoulder|hand_|robotiq.*(base|driver|coupler|follower|spring)/i;
// role of one source mesh: from its name where the source names it, else from the colour the source gave it
export function robotRole(rgba, name = "") {
  if (SERVO_NAME.test(name) && !/moving_jaw|Moving_Jaw|gripper_moving_finger|static_finger$/.test(name)) return "servo";
  const lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
  if (lum > 0.62) return "shell";
  if (lum < 0.3) return "servo";
  return "steel";
}
export const robotMaterial = (rgba, name = "") => MAT[robotRole(rgba, name)];

// ---------------------------------------------------------------- generic props (fixtures, trays, pallets, crates)
// sim/viz.py exports each fixture as its physics geoms: boxes and z-axis cylinders in the body frame.
const propMats = new Map();
function propMaterial(rgb, kind) {
  const k = `${rgb.join(",")}|${kind}`;
  if (!propMats.has(k)) {
    const c = new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
    const metal = /fixture|nest|pallet|socket|jig/.test(kind);
    propMats.set(k, metal
      ? new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.42, metalness: 0.55, clearcoat: 0.3, clearcoatRoughness: 0.4, envMapIntensity: 0.9 })
      : new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.5, clearcoat: 0.25, clearcoatRoughness: 0.4, envMapIntensity: 0.7 }));
  }
  return propMats.get(k);
}
export function partsMesh(parts, rgb, kind = "") {
  const g = new THREE.Group();
  for (const p of parts) {
    const mat = propMaterial(p.rgb || rgb, kind);
    let geo;
    if (p.type === "cylinder") {
      geo = new THREE.CylinderGeometry(p.size[0], p.size[0], p.size[1] * 2, 40);
      geo.rotateX(Math.PI / 2);
    } else {
      const [x, y, z] = p.size;
      geo = new RoundedBoxGeometry(x * 2, y * 2, z * 2, 3, Math.min(0.0015, x, y, z) * 0.9);
    }
    const m = new THREE.Mesh(geo, mat);
    m.position.fromArray(p.pos);
    if (p.quat) m.quaternion.set(p.quat[1], p.quat[2], p.quat[3], p.quat[0]);
    m.castShadow = m.receiveShadow = true;
    g.add(m);
  }
  return g;
}
