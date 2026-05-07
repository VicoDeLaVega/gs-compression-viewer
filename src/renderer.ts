import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SplatSet } from "./splats";

// ---------------------------------------------------------------------------
// Vertex shader — matches Spark's splat rendering pipeline:
//   • ExtSplats mode uses opacity as-is, like Spark editor.
//   • PackedSplats mode applies byte/255 → ×2 → nonlinear boost if >1.
//   • All three render params (maxStdDev, blurAmount, falloff) are uniforms
//     so they can be tuned live from the UI to match Spark's debug sliders.
//   • SH degree-1: view-dependent colour correction via 9 SH coefficients.
// ---------------------------------------------------------------------------
const vertexShader = `#version 300 es
precision highp float;

layout(location=0) in vec2  aCorner;
layout(location=1) in vec3  iCenter;
layout(location=2) in vec3  iScale;
layout(location=3) in vec4  iQuat;
layout(location=4) in vec4  iColor;
layout(location=5) in vec3  iSh1r;
layout(location=6) in vec3  iSh1g;
layout(location=7) in vec3  iSh1b;

uniform mat4  uViewProjection;
uniform mat4  uViewMatrix;
uniform mat4  uProjectionMatrix;
uniform vec2  uViewport;
uniform float uRadiusScale;
uniform vec3  uCameraPos;
uniform bool  uHasSH1;
uniform float uSHScale;
uniform float uMaxStdDev;
uniform float uBlurAmount;
uniform float uFalloff;
uniform bool  uPackedAlphaBoost;

out vec2  vLocal;
out vec4  vColor;
flat out float vStdDev;

mat3 scaleQuaternionToMatrix(vec3 s, vec4 q) {
  return mat3(
    s.x*(1.0-2.0*(q.y*q.y+q.z*q.z)), s.x*(2.0*(q.x*q.y+q.w*q.z)), s.x*(2.0*(q.x*q.z-q.w*q.y)),
    s.y*(2.0*(q.x*q.y-q.w*q.z)), s.y*(1.0-2.0*(q.x*q.x+q.z*q.z)), s.y*(2.0*(q.y*q.z+q.w*q.x)),
    s.z*(2.0*(q.x*q.z+q.w*q.y)), s.z*(2.0*(q.y*q.z-q.w*q.x)), s.z*(1.0-2.0*(q.x*q.x+q.y*q.y))
  );
}

void main() {
  float maxStdDev      = uMaxStdDev;
  const float maxPixelRadius = 128.0;
  float blurAmount     = uBlurAmount;
  const float clipXY         = 1.4;
  const float SH_C1          = 0.4886025119029199;

  // Frustum culling
  vec4 clipCenter = uViewProjection * vec4(iCenter, 1.0);
  if (clipCenter.w <= 0.0 || abs(clipCenter.z) >= clipCenter.w) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    vLocal = vec2(99.0); vColor = vec4(0.0);
    return;
  }
  vec2 ndcXY = clipCenter.xy / clipCenter.w;
  if (abs(ndcXY.x) > clipXY || abs(ndcXY.y) > clipXY) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    vLocal = vec2(99.0); vColor = vec4(0.0);
    return;
  }

  // 2D covariance projection
  vec3 viewCenter = (uViewMatrix * vec4(iCenter, 1.0)).xyz;
  vec3 scale = max(iScale * uRadiusScale, vec3(0.00001));
  mat3 RS    = mat3(uViewMatrix) * scaleQuaternionToMatrix(scale, normalize(iQuat));
  mat3 cov3D = RS * transpose(RS);

  vec2  focal = 0.5 * uViewport * vec2(uProjectionMatrix[0][0], uProjectionMatrix[1][1]);
  float invZ  = 1.0 / max(0.0001, -viewCenter.z);
  vec2  J1    = focal * invZ;
  vec2  J2    = (J1 * viewCenter.xy) * invZ;
  mat3  J     = mat3(J1.x, 0.0, J2.x,  0.0, J1.y, J2.y,  0.0, 0.0, 0.0);

  mat3  cov2D = transpose(J) * cov3D * J;
  float a     = cov2D[0][0];
  float d     = cov2D[1][1];
  float b     = cov2D[0][1];

  // Energy conservation: blurAdjust scales alpha when blurring the Gaussian.
  float detOrig = max(0.0, a * d - b * b);
  float a_bl    = a + blurAmount;
  float d_bl    = d + blurAmount;
  float det     = max(0.000001, a_bl * d_bl - b * b);

  float eigenAvg   = 0.5 * (a_bl + d_bl);
  float eigenDelta = sqrt(max(0.0, eigenAvg * eigenAvg - det));
  float eigen1     = max(0.000001, eigenAvg + eigenDelta);
  float eigen2     = max(0.000001, eigenAvg - eigenDelta);

  vec2 ev1 = (abs(b) > 0.0001) ? normalize(vec2(b, eigen1 - a_bl)) : vec2(1.0, 0.0);
  vec2 ev2 = vec2(ev1.y, -ev1.x);

  // Spark ExtSplats keeps alpha as-is; PackedSplats doubles it before boost.
  float alpha = uPackedAlphaBoost ? iColor.a * 2.0 : iColor.a;
  float adjSD = maxStdDev;
  if (alpha > 1.0) {
    alpha = min(alpha * 4.0 - 3.0, 5.0);
    adjSD = maxStdDev + 0.7 * (alpha - 1.0);
  }
  float blurAdjust = sqrt(detOrig / det);
  alpha *= blurAdjust;

  float s1 = min(maxPixelRadius, adjSD * sqrt(eigen1));
  float s2 = min(maxPixelRadius, adjSD * sqrt(eigen2));

  vec2 pixelOff = aCorner.x * ev1 * s1 + aCorner.y * ev2 * s2;
  gl_Position   = vec4(clipCenter.xy + (2.0 / uViewport) * pixelOff * clipCenter.w, clipCenter.zw);
  vLocal        = aCorner * adjSD;
  vStdDev       = adjSD;

  // SH degree-1 view-dependent colour
  vec3 baseColor = iColor.rgb;
  if (uHasSH1) {
    vec3 dir = normalize(iCenter - uCameraPos);
    float dx =  dir.x;
    float dy = -dir.y;
    float dz = -dir.z;
    float shR = SH_C1 * (-dy * iSh1r.x + dz * iSh1r.y - dx * iSh1r.z);
    float shG = SH_C1 * (-dy * iSh1g.x + dz * iSh1g.y - dx * iSh1g.z);
    float shB = SH_C1 * (-dy * iSh1b.x + dz * iSh1b.y - dx * iSh1b.z);
    baseColor = clamp(baseColor + uSHScale * vec3(shR, shG, shB), 0.0, 1.0);
  }

  vColor = vec4(baseColor, alpha);
}`;

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------
const fragmentShader = `#version 300 es
precision highp float;

in  vec2  vLocal;
in  vec4  vColor;
flat in float vStdDev;
uniform float uFalloff;
out vec4  fragColor;

void main() {
  float r2 = dot(vLocal, vLocal);
  if (r2 > vStdDev * vStdDev) discard;

  // Matches Spark's splatFragment exactly:
  //   falloff=0 → flat disc,  falloff=1 → full Gaussian / hard-disc formula.
  float alpha = vColor.a;
  float a;
  if (alpha <= 1.0) {
    // Soft Gaussian: mix between flat (alpha) and Gaussian falloff.
    a = mix(alpha, alpha * exp(-0.5 * r2), uFalloff);
  } else {
    // Hard disc: mix between fully opaque (1.0) and the hard-disc formula.
    float hardA = exp((alpha * alpha - 1.0) / 2.718281828459045);
    float disc  = 1.0 - pow(1.0 - exp(-0.5 * r2), hardA);
    a = mix(1.0, disc, uFalloff);
  }

  if (a < 0.003) discard;
  fragColor = vec4(vColor.rgb, a);
}`;

const lineVertexShader = `#version 300 es
precision highp float;

layout(location=0) in vec3 aPosition;
uniform mat4 uViewProjection;

void main() {
  gl_Position = uViewProjection * vec4(aPosition, 1.0);
}`;

const lineFragmentShader = `#version 300 es
precision highp float;

uniform vec4 uColor;
out vec4 fragColor;

void main() {
  fragColor = uColor;
}`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
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

function makeLineProgram(gl: WebGL2RenderingContext) {
  const program = gl.createProgram();
  if (!program) throw new Error("Could not allocate line program");
  gl.attachShader(program, makeShader(gl, gl.VERTEX_SHADER, lineVertexShader));
  gl.attachShader(program, makeShader(gl, gl.FRAGMENT_SHADER, lineFragmentShader));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) ?? "Line program link failed");
  }
  return program;
}

function robustBounds(splats: SplatSet, low = 0.02, high = 0.98) {
  const sampleCount = Math.min(splats.count, 80000);
  const stride = Math.max(1, Math.floor(splats.count / sampleCount));
  const xs: number[] = [], ys: number[] = [], zs: number[] = [];
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
  return { minX: xs[lo], maxX: xs[hi], minY: ys[lo], maxY: ys[hi], minZ: zs[lo], maxZ: zs[hi] };
}

// ---------------------------------------------------------------------------
// Renderer class
// ---------------------------------------------------------------------------
export class GaussianRenderer {
  readonly camera   = new THREE.PerspectiveCamera(58, 1, 0.02, 100);
  readonly controls: OrbitControls;

  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly vao: WebGLVertexArrayObject;
  private readonly lineProgram: WebGLProgram;
  private readonly lineVao: WebGLVertexArrayObject;
  private readonly lineBuffer: WebGLBuffer;

  private readonly centerBuffer: WebGLBuffer;
  private readonly scaleBuffer:  WebGLBuffer;
  private readonly quatBuffer:   WebGLBuffer;
  private readonly colorBuffer:  WebGLBuffer;
  private readonly sh1rBuffer:   WebGLBuffer;
  private readonly sh1gBuffer:   WebGLBuffer;
  private readonly sh1bBuffer:   WebGLBuffer;

  // Cached uniform locations — resolved once at construction
  private readonly uViewProjection:   WebGLUniformLocation;
  private readonly uViewMatrix:       WebGLUniformLocation;
  private readonly uProjectionMatrix: WebGLUniformLocation;
  private readonly uViewport:         WebGLUniformLocation;
  private readonly uRadiusScale:      WebGLUniformLocation;
  private readonly uCameraPos:        WebGLUniformLocation;
  private readonly uHasSH1:           WebGLUniformLocation;
  private readonly uSHScale:          WebGLUniformLocation;
  private readonly uMaxStdDev:        WebGLUniformLocation;
  private readonly uBlurAmount:       WebGLUniformLocation;
  private readonly uFalloff:          WebGLUniformLocation;
  private readonly uPackedAlphaBoost: WebGLUniformLocation;
  private readonly uLineViewProjection: WebGLUniformLocation;
  private readonly uLineColor: WebGLUniformLocation;

  private splatCount = 0;
  private voxelLineVertexCount = 0;
  private hasSH1     = false;
  shScale            = 1.0;
  showVoxelGrid      = false;
  // Rendering parameters — exposed for UI control
  maxStdDev          = Math.sqrt(8); // Spark default √8 ≈ 2.83
  blurAmount         = 0.3;          // Spark default
  falloff            = 1.0;          // 0=flat disc, 1=full Gaussian (Spark default)
  packedAlphaBoost   = false;        // Spark editor uses ExtSplats by default
  private readonly viewProjection = new THREE.Matrix4();
  private lastWidth  = 1;
  private lastHeight = 1;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) throw new Error("WebGL2 is required for this viewer");
    this.gl      = gl;
    this.program = makeProgram(gl);
    this.lineProgram = makeLineProgram(gl);

    const loc = (name: string): WebGLUniformLocation => {
      const l = gl.getUniformLocation(this.program, name);
      if (l === null) throw new Error(`Uniform '${name}' not found in shader`);
      return l;
    };
    this.uViewProjection   = loc("uViewProjection");
    this.uViewMatrix       = loc("uViewMatrix");
    this.uProjectionMatrix = loc("uProjectionMatrix");
    this.uViewport         = loc("uViewport");
    this.uRadiusScale      = loc("uRadiusScale");
    this.uCameraPos        = loc("uCameraPos");
    this.uHasSH1           = loc("uHasSH1");
    this.uSHScale          = loc("uSHScale");
    this.uMaxStdDev        = loc("uMaxStdDev");
    this.uBlurAmount       = loc("uBlurAmount");
    this.uFalloff          = loc("uFalloff");
    this.uPackedAlphaBoost = loc("uPackedAlphaBoost");
    const lineLoc = (name: string): WebGLUniformLocation => {
      const l = gl.getUniformLocation(this.lineProgram, name);
      if (l === null) throw new Error(`Uniform '${name}' not found in line shader`);
      return l;
    };
    this.uLineViewProjection = lineLoc("uViewProjection");
    this.uLineColor          = lineLoc("uColor");

    const mkBuf = () => {
      const b = gl.createBuffer();
      if (!b) throw new Error("Could not allocate GL buffer");
      return b;
    };
    const vao = gl.createVertexArray();
    if (!vao) throw new Error("Could not allocate VAO");
    const lineVao = gl.createVertexArray();
    if (!lineVao) throw new Error("Could not allocate line VAO");
    this.vao          = vao;
    this.lineVao      = lineVao;
    this.centerBuffer = mkBuf();
    this.scaleBuffer  = mkBuf();
    this.quatBuffer   = mkBuf();
    this.colorBuffer  = mkBuf();
    this.sh1rBuffer   = mkBuf();
    this.sh1gBuffer   = mkBuf();
    this.sh1bBuffer   = mkBuf();
    this.lineBuffer   = mkBuf();

    gl.bindVertexArray(vao);

    // Corner quad
    const cornerBuf = mkBuf();
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const addInst = (buf: WebGLBuffer, attrib: number, size: number) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(attrib);
      gl.vertexAttribPointer(attrib, size, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(attrib, 1);
    };
    addInst(this.centerBuffer, 1, 3);
    addInst(this.scaleBuffer,  2, 3);
    addInst(this.quatBuffer,   3, 4);
    addInst(this.colorBuffer,  4, 4);
    addInst(this.sh1rBuffer,   5, 3);
    addInst(this.sh1gBuffer,   6, 3);
    addInst(this.sh1bBuffer,   7, 3);

    gl.bindVertexArray(null);

    gl.bindVertexArray(lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.camera.position.set(0, 0.5, 5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0);
  }

  resize() {
    const dpr    = Math.min(window.devicePixelRatio || 1, 2);
    const width  = Math.max(1, Math.floor(this.canvas.clientWidth  * dpr));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (width === this.lastWidth && height === this.lastHeight) return;
    this.lastWidth  = width;
    this.lastHeight = height;
    this.canvas.width  = width;
    this.canvas.height = height;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.gl.viewport(0, 0, width, height);
  }

  upload(splats: SplatSet) {
    const gl = this.gl;
    this.splatCount = splats.count;
    this.hasSH1     = !!splats.sh1;

    const up = (buf: WebGLBuffer, data: Float32Array) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    };

    up(this.centerBuffer, splats.centers);
    up(this.scaleBuffer,  splats.scales);
    up(this.quatBuffer,   splats.quats);
    up(this.colorBuffer,  splats.colors);

    // Unpack the interleaved sh1 [r0,r1,r2, g0,g1,g2, b0,b1,b2] × N
    // into three separate vec3 streams for the instanced attributes.
    if (splats.sh1) {
      const n    = splats.count;
      const sh1r = new Float32Array(n * 3);
      const sh1g = new Float32Array(n * 3);
      const sh1b = new Float32Array(n * 3);
      const src  = splats.sh1;
      // SPZ layout per splat: [c0r,c0g,c0b, c1r,c1g,c1b, c2r,c2g,c2b]
      // iSh1r needs [c0r, c1r, c2r], iSh1g needs [c0g, c1g, c2g], etc.
      for (let i = 0; i < n; i++) {
        const s = i * 9, r = i * 3;
        sh1r[r]=src[s];   sh1r[r+1]=src[s+3]; sh1r[r+2]=src[s+6];
        sh1g[r]=src[s+1]; sh1g[r+1]=src[s+4]; sh1g[r+2]=src[s+7];
        sh1b[r]=src[s+2]; sh1b[r+1]=src[s+5]; sh1b[r+2]=src[s+8];
      }
      up(this.sh1rBuffer, sh1r);
      up(this.sh1gBuffer, sh1g);
      up(this.sh1bBuffer, sh1b);
    } else {
      const zeros = new Float32Array(splats.count * 3);
      up(this.sh1rBuffer, zeros);
      up(this.sh1gBuffer, zeros);
      up(this.sh1bBuffer, zeros);
    }
  }

  uploadVoxelGrid(bounds: { min: [number, number, number]; max: [number, number, number] }, dims: [number, number, number]) {
    const [nx, ny, nz] = dims;
    const [minX, minY, minZ] = bounds.min;
    const [maxX, maxY, maxZ] = bounds.max;
    const vertices: number[] = [];
    const pushLine = (ax: number, ay: number, az: number, bx: number, by: number, bz: number) => {
      vertices.push(ax, ay, az, bx, by, bz);
    };
    const xAt = (i: number) => minX + (maxX - minX) * (i / nx);
    const yAt = (i: number) => minY + (maxY - minY) * (i / ny);
    const zAt = (i: number) => minZ + (maxZ - minZ) * (i / nz);

    for (let iy = 0; iy <= ny; iy++) {
      for (let iz = 0; iz <= nz; iz++) pushLine(minX, yAt(iy), zAt(iz), maxX, yAt(iy), zAt(iz));
    }
    for (let ix = 0; ix <= nx; ix++) {
      for (let iz = 0; iz <= nz; iz++) pushLine(xAt(ix), minY, zAt(iz), xAt(ix), maxY, zAt(iz));
    }
    for (let ix = 0; ix <= nx; ix++) {
      for (let iy = 0; iy <= ny; iy++) pushLine(xAt(ix), yAt(iy), minZ, xAt(ix), yAt(iy), maxZ);
    }

    const data = new Float32Array(vertices);
    this.voxelLineVertexCount = data.length / 3;
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.lineBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW);
  }

  fitToSplats(splats: SplatSet) {
    const { minX, maxX, minY, maxY, minZ, maxZ } = robustBounds(splats);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    this.resize();
    const width  = Math.max(0.001, maxX - minX);
    const height = Math.max(0.001, maxY - minY);
    const depth  = Math.max(0.001, maxZ - minZ);
    const fovY = THREE.MathUtils.degToRad(this.camera.fov);
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * this.camera.aspect);
    const fitY = height / (2 * Math.tan(fovY / 2));
    const fitX = width  / (2 * Math.tan(fovX / 2));
    const sceneSize = Math.hypot(width, height, depth);
    // Scale camera distance with actual scene size — no absolute minimums
    // so SPZ scenes with raw coordinates (any scale) get sensible framing.
    const distance = Math.max(fitX, fitY) * 2.0 + depth;
    const radius   = sceneSize * 0.5;
    this.controls.target.set(cx, cy, cz);
    this.camera.position.set(cx, cy, cz + Math.max(distance, sceneSize * 0.01));
    this.camera.near = Math.max(sceneSize * 0.0001, distance / 1000);
    this.camera.far  = Math.max(distance * 4, radius * 20);
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

    this.viewProjection.multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse,
    );
    gl.uniformMatrix4fv(this.uViewProjection,   false, this.viewProjection.elements);
    gl.uniformMatrix4fv(this.uViewMatrix,        false, this.camera.matrixWorldInverse.elements);
    gl.uniformMatrix4fv(this.uProjectionMatrix,  false, this.camera.projectionMatrix.elements);
    gl.uniform2f(this.uViewport,    this.lastWidth, this.lastHeight);
    gl.uniform1f(this.uRadiusScale, radiusScale);
    gl.uniform3f(
      this.uCameraPos,
      this.camera.position.x,
      this.camera.position.y,
      this.camera.position.z,
    );
    gl.uniform1i(this.uHasSH1, this.hasSH1 ? 1 : 0);
    gl.uniform1f(this.uSHScale, this.shScale);
    gl.uniform1f(this.uMaxStdDev, this.maxStdDev);
    gl.uniform1f(this.uBlurAmount, this.blurAmount);
    gl.uniform1f(this.uFalloff, this.falloff);
    gl.uniform1i(this.uPackedAlphaBoost, this.packedAlphaBoost ? 1 : 0);

    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.splatCount);
    gl.bindVertexArray(null);

    if (this.showVoxelGrid && this.voxelLineVertexCount > 0) {
      gl.useProgram(this.lineProgram);
      gl.uniformMatrix4fv(this.uLineViewProjection, false, this.viewProjection.elements);
      gl.uniform4f(this.uLineColor, 0.35, 0.66, 1.0, 0.22);
      gl.bindVertexArray(this.lineVao);
      gl.drawArrays(gl.LINES, 0, this.voxelLineVertexCount);
      gl.bindVertexArray(null);
    }
  }
}
