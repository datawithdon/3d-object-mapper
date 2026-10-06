// 3D Object Mapper - frontend logic: file picking, upload, status, and
// a Three.js point-cloud viewer with orbit controls.

(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const fileListEl = document.getElementById("file-list");
  const form = document.getElementById("upload-form");
  const submitBtn = document.getElementById("submit-btn");
  const statusEl = document.getElementById("status");
  const viewerToolbar = document.getElementById("viewer-toolbar");
  const viewerInfo = document.getElementById("viewer-info");
  const downloadLink = document.getElementById("download-link");
  const placeholder = document.getElementById("viewer-placeholder");
  const container = document.getElementById("viewer-container");

  let selectedFiles = [];

  function renderFileList() {
    fileListEl.innerHTML = "";
    selectedFiles.forEach((f) => {
      const row = document.createElement("div");
      row.className = "file-row";
      row.innerHTML = `<span>${f.name}</span><span>${(f.size / 1024).toFixed(0)} KB</span>`;
      fileListEl.appendChild(row);
    });
    submitBtn.disabled = selectedFiles.length < 3;
  }

  function setStatus(message, kind) {
    statusEl.hidden = !message;
    statusEl.textContent = message || "";
    statusEl.className = "status" + (kind ? " " + kind : "");
  }

  fileInput.addEventListener("change", () => {
    selectedFiles = Array.from(fileInput.files);
    renderFileList();
  });

  ["dragover", "dragenter"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "dragend", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, () => dropzone.classList.remove("dragover"))
  );
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter((f) =>
      /\.(jpe?g|png|bmp)$/i.test(f.name)
    );
    selectedFiles = dropped;
    renderFileList();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (selectedFiles.length < 3) return;

    submitBtn.disabled = true;
    setStatus(`Reconstructing from ${selectedFiles.length} photos... this can take a minute.`);

    const formData = new FormData();
    selectedFiles.forEach((f) => formData.append("images", f));

    try {
      const res = await fetch("/api/reconstruct", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        setStatus(data.error || "Reconstruction failed.", "error");
        if (data.detail) setStatus((data.error || "") + " " + data.detail, "error");
        return;
      }

      setStatus(
        `Done: ${data.num_points} points from ${data.num_images_used} photos.`,
        "success"
      );
      if (data.warnings && data.warnings.length) {
        console.warn("Reconstruction warnings:", data.warnings);
      }

      viewerToolbar.hidden = false;
      viewerInfo.textContent = `${data.num_points} points · ${data.num_images_used} photos used`;
      downloadLink.href = data.download_url;

      placeholder.hidden = true;
      renderPointCloud(data.points);
    } catch (err) {
      setStatus("Network or server error: " + err.message, "error");
    } finally {
      submitBtn.disabled = selectedFiles.length < 3;
    }
  });

  // ---- Three.js point cloud viewer ----
  let scene, camera, renderer, controls, pointsObject, animId;

  function initViewer() {
    if (renderer) return; // already initialized

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(
      55,
      container.clientWidth / container.clientHeight,
      0.01,
      100
    );
    camera.position.set(0, 0, 4);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    window.addEventListener("resize", onResize);
    animate();
  }

  function onResize() {
    if (!renderer) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }

  function animate() {
    animId = requestAnimationFrame(animate);
    if (controls) controls.update();
    if (pointsObject) pointsObject.rotation.y += 0.0015;
    renderer.render(scene, camera);
  }

  function renderPointCloud(points) {
    initViewer();

    if (pointsObject) {
      scene.remove(pointsObject);
      pointsObject.geometry.dispose();
      pointsObject.material.dispose();
    }

    const n = points.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);

    for (let i = 0; i < n; i++) {
      const [x, y, z, r, g, b] = points[i];
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
      colors[i * 3] = r / 255;
      colors[i * 3 + 1] = g / 255;
      colors[i * 3 + 2] = b / 255;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({ size: 0.035, vertexColors: true });
    pointsObject = new THREE.Points(geometry, material);
    scene.add(pointsObject);
  }
})();
