# NBody Visualizer — Unity Project

Real-time 3D visualizer for the `frames.bin` binary output produced by
**nbody-sim** (the CUDA N-body protoplanetary disk simulation).

![Screenshot placeholder](docs/screenshot.png)

---

## Requirements

| Tool | Version |
|------|---------|
| Unity Editor | 2021.3 LTS or newer (tested on 2022.3.22f1) |
| Render Pipeline | Built-in (no URP/HDRP required) |
| GPU | Any — geometry shaders require Shader Model 4.0 (DX11 / OpenGL 4.1) |

---

## Quick Start

### 1. Open the project in Unity

1. Open **Unity Hub**.
2. Click **Add → Add project from disk**.
3. Select `C:\Projects\nbody-sim\unity-visualizer`.
4. Unity will import all packages and compile scripts (may take a minute).

### 2. Create the scene

**File → New Scene → Basic (Built-in)** → Save as `Assets/Scenes/Visualizer.unity`.

### 3. Create the particle material

1. **Assets → Create → Material** — name it `ParticleMaterial`.
2. In the Inspector, set **Shader** to `NBody/Particle`.
3. Adjust `Glow Falloff` (1.8) and `Brightness` (1.0) as desired.

### 4. Set up the Simulation GameObject

1. Create an empty `GameObject` — name it `Simulation`.
2. Add component **NBodyPlayer**:
   - Set **File Path** to the absolute path of your `frames.bin`, e.g.:
     `C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data\20260524_232513\frames.bin`
   - **Playback Speed**: `20` (T₀ per second — one orbit of the inner disk ≈ 2.2 T₀)
   - Check **Auto Play**.
3. Add component **NBodyRenderer**:
   - Drag `ParticleMaterial` into **Particle Material**.
   - **Particle Size**: `0.06` AU (increase for visibility, decrease for accuracy).
   - **Color Radius Max**: `5` AU (matches `disk_r_max` in the simulation config).
   - Check **Show Star** to display the central star.

### 5. Configure the camera

1. Select the **Main Camera**.
2. Add component **OrbitCamera**.
   - **Initial Distance**: `18` AU.
   - **Initial Pitch**: `55°` (slight bird's-eye view of the disk).
3. In the Camera component:
   - **Background**: black (`#000000`).
   - **Clear Flags**: Solid Color.

### 6. (Optional) Add the UI

1. **GameObject → UI → Canvas** → set Canvas Scaler to **Scale With Screen Size**.
2. Add UI elements (Button, Slider, TMP_Text, TMP_InputField) as described in `NBodyUI.cs`.
3. Add the **NBodyUI** component to the Canvas root.
4. Wire up all Inspector references from the Canvas hierarchy.

**Minimum working UI** (skip the rest for a quick start):
- Just press Play — the simulation plays automatically if **File Path** is set.
- Use keyboard shortcut mapping (see below) instead of UI buttons.

### 7. Press Play ▶

The disk should appear as a glowing particle cloud. Use the mouse to orbit:
- **Left drag** — rotate around the disk
- **Right drag** — pan
- **Scroll wheel** — zoom in/out

---

## Keyboard shortcuts (add to a MonoBehaviour if desired)

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| ← / → | Step one frame back / forward |
| +  / - | Increase / decrease playback speed |

---

## frames.bin format reference

```
Header (32 bytes):
  magic       char[4]   "NBOD"
  version     uint32    1
  n_particles uint32
  n_frames    uint32    (patched by finalize())
  dt          double    simulation timestep (T0)
  output_dt   double    time between written frames (T0)

Per frame:
  time_T0     double    simulation time (T0)
  x[n]        float32   x positions in AU   } SoA layout —
  y[n]        float32   y positions in AU   } all X first,
  z[n]        float32   z positions in AU   } then all Y, then all Z
```

**Coordinate system:** The disk lies in the **XZ plane** (Y is vertical, same as Unity Y-up).
The central star is at the origin.

**Time unit:** 1 T₀ = 1/(2π) years ≈ 58.1 days (natural units where G=1, M_star=1 M☉, r=1 AU).

---

## File sizes

| N particles | Frames | frames.bin size |
|-------------|--------|-----------------|
| 1,000       | 20     | ~240 KB |
| 100,000     | 2      | ~2.4 MB |
| 100,000     | 1,000  | ~1.2 GB |

---

## Performance notes

| Particles | Expected FPS (RTX 3060) |
|-----------|------------------------|
| 1,000     | 500+ fps |
| 100,000   | ~60 fps |
| 1,000,000 | ~10 fps (mesh upload bottleneck) |

For particle counts above ~500k, consider switching the renderer to a
`ComputeBuffer` + `Graphics.DrawProceduralIndirect` approach to avoid
the CPU-side mesh vertex upload bottleneck.

---

## Project structure

```
Assets/
├── Scripts/
│   ├── NBodyReader.cs     Binary file reader (streaming, frame random-access)
│   ├── NBodyPlayer.cs     Playback controller MonoBehaviour
│   ├── NBodyRenderer.cs   Point-cloud mesh renderer MonoBehaviour
│   ├── NBodyUI.cs         HUD controller (optional)
│   └── OrbitCamera.cs     Mouse orbit/pan/zoom camera
├── Shaders/
│   └── NBodyParticle.shader  Geometry-shader particle with additive glow
└── Materials/
    └── ParticleMaterial.mat  (create manually — see Quick Start)
```
