import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SplatSet } from "./splats";

const vertexShader = `#version 300 es
precision highp float;

layout(location=0) in vec2 aCorner;
layout(location=1) in vec3 iCenter;
layout(location=2) in float iRadius;
layout(location=3) in vec4 iColor;

uniform mat4 uViewProjection;
uniform vec2 uViewport;
uniform float uRadiusScale;

out vec2 vLocal;
out vec4 vColor;

void main() {
  vec4 clip = uViewProjection * vec4(iCenter, 1.0);
  vec2 pixel = vec2(iRadius * uRadiusScale * 2.0 / uViewport.x, iRadius * uRadiusScale * 2.0 / uViewport.y);
  clip.xy += aCorner * pixel * clip.w;
  gl_Position = clip;
  vLocal = aCorner;
  vColor = iColor;
}`;

const fragmentShader = `#version 300 es
precision highp float;

in vec2 vLocal;
in vec4 vColor;
out vec4 fragColor;

void main() {
  float r2 = dot(vLocal, vLocal);
  if (r2 > 1.0) discard;
  float a = exp(-r2 * 4.0) * vColor.a;
  if (a < 0.004) discard;
  fragColor = vec4(vColor.rgb, a);
}`;

function makeShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Could not allocate shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? "Shader compile failed");
  }
  return shader;
}

function makeProgram(gl: WebGL2RenderingContext) {
  const program = gl.createProgram();
  if (!program) throw new Error("Could not allocate program");
  gl.attachShader(program, makeShader(gl, gl.VERTEX_SHADER, vertexShader));
  gl.attachShader(program, makeShader(gl, gl.FRAGMENT_SHADER, fragmentShader));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) ?? "Program link failed");
  }
  return program;
}

function robustBounds(splats: SplatSet, low = 0.02, high = 0.98) {
  const sampleCount = Math.min(splats.count, 80000);
  const stride = Math.max(1, Math.floor(splats.count / sampleCount));
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  for (let i = 0; i < splats.count; i += stride) {
    const o = i * 3;
    xs.push(splats.centers[o]);
    ys.push(splats.centers[o + 1]);
    zs.push(splats.centers[o + 2]);
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const lo = Math.max(0, Math.min(xs.length - 1, Math.floor(xs.length * low)));
  const hi = Math.max(lo, Math.min(xs.length - 1, Math.floor(xs.length * high)));
  return {
    minX: xs[lo],
    maxX: xs[hi],
    minY: ys[lo],
    maxY: ys[hi],
    minZ: zs[lo],
    maxZ: zs[hi],
  };
}

export class GaussianRenderer {
  readonly camera = new THREE.PerspectiveCamera(58, 1, 0.02, 100);
  readonly controls: OrbitControls;
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly vao: WebGLVertexArrayObject;
  private readonly centerBuffer: WebGLBuffer;
  private readonly radiusBuffer: WebGLBuffer;
  private readonly colorBuffer: WebGLBuffer;
  private splatCount = 0;
  private readonly viewProjection = new THREE.Matrix4();
  private lastWidth = 1;
  private lastHeight = 1;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) throw new Error("WebGL2 is required for this viewer");
    this.gl = gl;
    this.program = makeProgram(gl);

    const vao = gl.createVertexArray();
    const centerBuffer = gl.createBuffer();
    const radiusBuffer = gl.createBuffer();
    const colorBuffer = gl.createBuffer();
    if (!vao || !centerBuffer || !radiusBuffer || !colorBuffer) throw new Error("Could not allocate GL buffers");
    this.vao = vao;
    this.centerBuffer = centerBuffer;
    this.radiusBuffer = radiusBuffer;
    this.colorBuffer = colorBuffer;

    const cornerBuffer = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, centerBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(1, 1);

    gl.bindBuffer(gl.ARRAY_BUFFER, radiusBuffer);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(2, 1);

    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(3, 1);
    gl.bindVertexArray(null);

    this.camera.position.set(0, 0.5, 5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (width === this.lastWidth && height === this.lastHeight) return;
    this.lastWidth = width;
    this.lastHeight = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.gl.viewport(0, 0, width, height);
  }

  upload(splats: SplatSet) {
    const gl = this.gl;
    this.splatCount = splats.count;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.centerBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, splats.centers, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.radiusBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, splats.radii, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, splats.colors, gl.DYNAMIC_DRAW);
  }

  fitToSplats(splats: SplatSet) {
    const { minX, maxX, minY, maxY, minZ, maxZ } = robustBounds(splats);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    this.resize();
    const width = Math.max(0.001, maxX - minX);
    const height = Math.max(0.001, maxY - minY);
    const depth = Math.max(0.001, maxZ - minZ);
    const fovY = THREE.MathUtils.degToRad(this.camera.fov);
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * this.camera.aspect);
    const fitY = height / (2 * Math.tan(fovY / 2));
    const fitX = width / (2 * Math.tan(fovX / 2));
    const distance = Math.max(2.2, Math.max(fitX, fitY) * 2.0 + depth);
    const radius = Math.max(1.5, Math.hypot(width, height, depth) * 0.5);
    this.controls.target.set(cx, cy, cz);
    this.camera.position.set(cx, cy, cz + distance);
    this.camera.near = Math.max(0.01, distance / 1000);
    this.camera.far = Math.max(distance * 4, radius * 20);
    this.camera.updateProjectionMatrix();
    this.controls.saveState();
    this.controls.update();
  }

  render(radiusScale: number) {
    this.resize();
    const gl = this.gl;
    gl.clearColor(0.01, 0.015, 0.03, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.program);

    this.viewProjection.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, "uViewProjection"), false, this.viewProjection.elements);
    gl.uniform2f(gl.getUniformLocation(this.program, "uViewport"), this.lastWidth, this.lastHeight);
    gl.uniform1f(gl.getUniformLocation(this.program, "uRadiusScale"), radiusScale);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.splatCount);
    gl.bindVertexArray(null);
  }
}
