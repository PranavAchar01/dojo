// One stage for every picture Dojo makes: the Sim tab's lab (sealed concrete, haze, studio lights, ambient occlusion,
// neutral tone mapping) from studio.js, used here by the 3D model thumbnails and previews and by the factory view.
// Z is up, in metres, exactly as the engine exports its scenes.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { HAZE, setupRenderer, makeEnvironment, makeStage, addStudioLights, buildLab } from "./studio.js";

export const STAGE_TAG = "dojo-stage-v1"; // stamped into every thumbnail by scripts/stamp_thumbs.py

export function createStage(canvas, { width, height, target = [0.2, 0, 0], extent = 1.1, reach = 1, fov = 30, fog = [2.6, 7], preserve = false } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: preserve });
  renderer.setPixelRatio(width ? 1 : Math.min(devicePixelRatio, 2));
  if (width) renderer.setSize(width, height, false);
  setupRenderer(renderer);
  const scene = new THREE.Scene();
  scene.background = HAZE;
  if (fog) scene.fog = new THREE.Fog(HAZE, fog[0], fog[1]);
  scene.environment = makeEnvironment(renderer);
  scene.environmentIntensity = 0.55;
  const camera = new THREE.PerspectiveCamera(fov, (width || 1) / (height || 1), 0.01, 60);
  camera.up.set(0, 0, 1);
  const lights = addStudioLights(scene, { target, extent, reach });
  const lab = buildLab(scene, renderer, { cell: null, joints: false });
  const post = makeStage(renderer, scene, camera);
  const resize = (w, h) => {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    post.setSize(w, h);
  };
  return { THREE, renderer, scene, camera, lights, lab, post, resize, render: () => post.render() };
}

// A library model (a web GLB, y up) as a z-up group standing on the floor, scaled by `scale`.
export async function loadModel(url, scale = 1) {
  const g = (await new GLTFLoader().loadAsync(url)).scene;
  g.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = true; } });
  const inner = new THREE.Group();
  inner.rotation.x = Math.PI / 2;
  inner.scale.setScalar(scale);
  inner.add(g);
  return inner;
}

// Stand an object on the floor, centred, scaled so its longest side is `size` metres (a turntable convention: the
// stage's light and ambient occlusion are tuned for objects about the size of the arm's work area).
export function standOnFloor(obj, size = 0.42) {
  const holder = new THREE.Group();
  holder.add(obj);
  const box = new THREE.Box3().setFromObject(holder);
  const s = box.getSize(new THREE.Vector3());
  const k = size / Math.max(s.x, s.y, s.z, 1e-6);
  holder.scale.setScalar(k);
  const b2 = new THREE.Box3().setFromObject(holder);
  const c = b2.getCenter(new THREE.Vector3());
  holder.position.set(-c.x + 0.2, -c.y, -b2.min.z);
  holder.updateMatrixWorld(true);
  return { holder, size: s.toArray(), k };
}

// Three-quarter camera on a box.
export function frameBox(stage, box, dir = [0.62, -0.62, 0.5], fill = 1.0) {
  const c = box.getCenter(new THREE.Vector3());
  const rad = Math.max(0.05, box.getSize(new THREE.Vector3()).length() / 2);
  const d = (rad / Math.sin((stage.camera.fov * Math.PI) / 360)) * fill;
  const v = new THREE.Vector3(...dir).normalize().multiplyScalar(d);
  stage.camera.position.copy(c).add(v);
  stage.camera.lookAt(c);
  stage.camera.updateProjectionMatrix();
  return { center: c, distance: d };
}
