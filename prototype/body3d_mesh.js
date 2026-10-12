import * as THREE from "./vendor/three.module.js";

/**
 * High-performance 3D Body Mesh Try-On Engine using Three.js & MediaPipe.
 *
 * Constructs an articulated 3D mesh (Torso, Arms, Legs) that deforms
 * dynamically to 3D MediaPipe pose landmarks, wrapping the garment texture
 * around the body with true 3D curvature, depth, and lighting.
 */
export class Body3DMeshTryOn {
  constructor(canvas, videoElement) {
    this.canvas = canvas;
    this.video = videoElement;

    // 1. Three.js Scene & Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.scene = new THREE.Scene();

    // Camera: matching video pixel coordinates (0,0 top-left, w,h bottom-right)
    this.width = canvas.width || 640;
    this.height = canvas.height || 480;
    this.camera = new THREE.OrthographicCamera(
      0, this.width,
      0, this.height,
      -1000, 1000
    );
    this.camera.position.z = 500;

    // 2. Real-Time 3D Lighting (creates natural fabric highlights & depth)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xfff8ee, 0.7);
    dirLight.position.set(0, -200, 300);
    this.scene.add(dirLight);

    // 3. Garment Texture & Material
    this.textureLoader = new THREE.TextureLoader();
    this.garmentTexture = null;
    this.material = new THREE.MeshStandardMaterial({
      roughness: 0.8,
      metalness: 0.05,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: true,
    });

    // 4. Create 3D Meshes
    this.torsoMesh = this.createTorsoMesh();
    this.rArmMesh = this.createLimbMesh();
    this.lArmMesh = this.createLimbMesh();
    this.rLegMesh = this.createLimbMesh();
    this.lLegMesh = this.createLimbMesh();

    this.group = new THREE.Group();
    this.group.add(this.torsoMesh);
    this.group.add(this.rArmMesh);
    this.group.add(this.lArmMesh);
    this.group.add(this.rLegMesh);
    this.group.add(this.lLegMesh);
    this.scene.add(this.group);

    this.visible = false;
    this.group.visible = false;
  }

  loadGarment(url) {
    this.textureLoader.load(url, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.wrapS = THREE.ClampToEdgeWrapping;
      tex.wrapT = THREE.ClampToEdgeWrapping;
      this.garmentTexture = tex;
      this.material.map = tex;
      this.material.needsUpdate = true;
    });
  }

  resize(w, h) {
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.renderer.setSize(w, h, false);
    this.camera.right = w;
    this.camera.bottom = h;
    this.camera.updateProjectionMatrix();
  }

  createTorsoMesh() {
    // 3D Elliptical Cylinder: 16 radial segments, 8 vertical slices
    const radial = 16;
    const vertical = 8;
    const geom = new THREE.PlaneGeometry(1, 1, radial, vertical);
    const mesh = new THREE.Mesh(geom, this.material);
    mesh.frustumCulled = false;
    return mesh;
  }

  createLimbMesh() {
    // 3D Cylinder for limbs: 12 radial segments, 4 height slices
    const geom = new THREE.PlaneGeometry(1, 1, 12, 4);
    const mesh = new THREE.Mesh(geom, this.material);
    mesh.frustumCulled = false;
    return mesh;
  }

  /**
   * Update 3D mesh vertices from MediaPipe 33 landmarks.
   */
  updatePose(landmarks) {
    if (!landmarks || landmarks.length < 25) {
      this.group.visible = false;
      return;
    }

    const w = this.width;
    const h = this.height;

    // Convert normalized landmarks to screen pixel coordinates
    // MediaPipe: 11=LSh, 12=RSh, 13=LElb, 14=RElb, 15=LWri, 16=RWri, 23=LHip, 24=RHip, 25=LKnee, 26=RKnee, 27=LAnk, 28=RAnk
    const pt = (idx) => {
      const lm = landmarks[idx];
      return new THREE.Vector3(
        lm.x * w,
        lm.y * h,
        (lm.z || 0) * -w * 0.8 // scale z to pixel depth
      );
    };

    const rSh = pt(12);
    const lSh = pt(11);
    const rElb = pt(14);
    const lElb = pt(13);
    const rWri = pt(16);
    const lWri = pt(15);
    const rHip = pt(24);
    const lHip = pt(23);
    const rKnee = pt(26);
    const lKnee = pt(25);
    const rAnk = pt(28);
    const lAnk = pt(27);

    // Shoulder width and orientation
    const shVec = new THREE.Vector3().subVectors(lSh, rSh);
    const shWidth = shVec.length();
    if (shWidth < 30) {
      this.group.visible = false;
      return;
    }

    this.group.visible = true;

    // 1. UPDATE TORSO 3D MESH (Neck/Shoulders -> Hips -> Knees)
    this.deformTorso(rSh, lSh, rHip, lHip, rKnee, lKnee, shWidth);

    // 2. UPDATE ARMS 3D MESH (Shoulder -> Elbow -> Wrist)
    this.deformLimb(this.rArmMesh, rSh, rElb, rWri, shWidth * 0.35, { uMin: 0.0, uMax: 0.35, vMin: 0.05, vMax: 0.70 });
    this.deformLimb(this.lArmMesh, lSh, lElb, lWri, shWidth * 0.35, { uMin: 0.65, uMax: 1.0, vMin: 0.05, vMax: 0.70 });

    // 3. UPDATE LEGS 3D MESH (Hip -> Knee -> Ankle)
    this.deformLimb(this.rLegMesh, rHip, rKnee, rAnk, shWidth * 0.30, { uMin: 0.28, uMax: 0.50, vMin: 0.75, vMax: 1.0 });
    this.deformLimb(this.lLegMesh, lHip, lKnee, lAnk, shWidth * 0.30, { uMin: 0.50, uMax: 0.72, vMin: 0.75, vMax: 1.0 });
  }

  deformTorso(rSh, lSh, rHip, lHip, rKnee, lKnee, shWidth) {
    const geom = this.torsoMesh.geometry;
    const pos = geom.attributes.position;
    const uv = geom.attributes.uv;

    const radial = 16;
    const vertical = 8;

    const midSh = new THREE.Vector3().addVectors(rSh, lSh).multiplyScalar(0.5);
    const midHip = new THREE.Vector3().addVectors(rHip, lHip).multiplyScalar(0.5);
    const midKnee = new THREE.Vector3().addVectors(rKnee, lKnee).multiplyScalar(0.5);

    // Estimate 3D normal / facing direction
    const rightDir = new THREE.Vector3().subVectors(lSh, rSh).normalize();
    const downDir = new THREE.Vector3().subVectors(midHip, midSh).normalize();
    const frontDir = new THREE.Vector3().crossVectors(rightDir, downDir).normalize();

    let idx = 0;
    for (let r = 0; r <= vertical; r++) {
      const vT = r / vertical; // 0 (top/neck) to 1 (hem/knees)
      let center, radX;

      if (vT < 0.5) {
        // Upper torso (shoulders to hips)
        const t = vT / 0.5;
        center = new THREE.Vector3().lerpVectors(midSh, midHip, t);
        center.y -= shWidth * 0.12 * (1 - t); // raise collar slightly above shoulders
        radX = (shWidth * 0.52) * (1 - t * 0.15);
      } else {
        // Lower torso / Kameez drape (hips to knees)
        const t = (vT - 0.5) / 0.5;
        center = new THREE.Vector3().lerpVectors(midHip, midKnee, t);
        radX = (shWidth * 0.44) * (1 + t * 0.18); // slight flare at kameez hem
      }

      // Torso 3D thickness (front to back)
      const radZ = radX * 0.55;

      for (let c = 0; c <= radial; c++) {
        // Front arc from -PI/2 to +PI/2
        const theta = -Math.PI * 0.5 + (c / radial) * Math.PI;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        // 3D position = center + X*rightDir + Z*frontDir
        const px = center.x + (cosT * radX * rightDir.x) + (sinT * radZ * frontDir.x);
        const py = center.y + (cosT * radX * rightDir.y) + (sinT * radZ * frontDir.y);
        const pz = center.z + (cosT * radX * rightDir.z) + (sinT * radZ * frontDir.z);

        pos.setXYZ(idx, px, py, pz);

        // Torso front UV: U from 0.28 to 0.72, V from 0.05 to 0.78
        const u = 0.28 + (c / radial) * (0.72 - 0.28);
        const v = 0.95 - (vT * 0.72); // Invert V for Three.js
        uv.setXY(idx, u, v);

        idx++;
      }
    }

    pos.needsUpdate = true;
    uv.needsUpdate = true;
    geom.computeVertexNormals();
  }

  deformLimb(mesh, start, mid, end, radius, uvRange) {
    const geom = mesh.geometry;
    const pos = geom.attributes.position;
    const uv = geom.attributes.uv;

    const radial = 12;
    const vertical = 4;

    // Bone curve: start (shoulder) -> mid (elbow) -> end (wrist)
    const curve = [start, mid, end];

    let idx = 0;
    for (let r = 0; r <= vertical; r++) {
      const vT = r / vertical; // 0 to 1 along limb
      let center, boneDir;

      if (vT <= 0.5) {
        const t = vT / 0.5;
        center = new THREE.Vector3().lerpVectors(start, mid, t);
        boneDir = new THREE.Vector3().subVectors(mid, start).normalize();
      } else {
        const t = (vT - 0.5) / 0.5;
        center = new THREE.Vector3().lerpVectors(mid, end, t);
        boneDir = new THREE.Vector3().subVectors(end, mid).normalize();
      }

      // Perpendicular 3D vectors
      const up = Math.abs(boneDir.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
      const right = new THREE.Vector3().crossVectors(boneDir, up).normalize();
      const forward = new THREE.Vector3().crossVectors(right, boneDir).normalize();

      const rad = radius * (1.1 - vT * 0.3); // tapers toward wrist/ankle

      for (let c = 0; c <= radial; c++) {
        const theta = -Math.PI * 0.5 + (c / radial) * Math.PI;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);

        const px = center.x + (cosT * rad * right.x) + (sinT * rad * forward.x);
        const py = center.y + (cosT * rad * right.y) + (sinT * rad * forward.y);
        const pz = center.z + (cosT * rad * right.z) + (sinT * rad * forward.z);

        pos.setXYZ(idx, px, py, pz);

        const u = uvRange.uMin + (c / radial) * (uvRange.uMax - uvRange.uMin);
        const v = (1.0 - uvRange.vMin) - vT * (uvRange.vMax - uvRange.vMin);
        uv.setXY(idx, u, v);

        idx++;
      }
    }

    pos.needsUpdate = true;
    uv.needsUpdate = true;
    geom.computeVertexNormals();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
