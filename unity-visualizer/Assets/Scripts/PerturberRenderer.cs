using System;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace NBodyVisualizer
{
    /// <summary>
    /// Renders the fixed outer perturber (a single massive body on a circular
    /// Keplerian orbit) as a glowing green sphere. Listens to the
    /// <see cref="NBodyPlayer"/> on the same GameObject and updates the
    /// perturber's position each frame from <see cref="NBodyPlayer.CurrentTime"/>.
    ///
    /// Parameters are loaded from a sidecar JSON file <b>perturber.json</b>
    /// placed next to the corresponding frames.bin:
    ///   {
    ///     "enabled":      true,
    ///     "mass_msun":    0.000955,
    ///     "r_AU":         20.0,
    ///     "phase0_rad":   0.0,
    ///     "omega_T0_inv": 0.0111803
    ///   }
    ///
    /// If the file is absent or enabled=false, this component is silent.
    /// </summary>
    [RequireComponent(typeof(NBodyPlayer))]
    public class PerturberRenderer : MonoBehaviour
    {
        [Header("Material")]
        [Tooltip("Material using the NBody/Perturber shader.")]
        [SerializeField] private Material _perturberMaterial;

        [Header("Appearance")]
        [Tooltip("Diameter scale (AU). Final size scales with sqrt(mass_msun).")]
        [SerializeField] private float _baseDiameter = 1.5f;

        [Tooltip("Sidecar JSON filename next to frames.bin. " +
                 "Leave empty for default 'perturber.json'.")]
        [SerializeField] private string _sidecarFilename = "perturber.json";

        [Tooltip("Override perturber parameters in the Inspector instead of " +
                 "reading the sidecar file. Useful for quick tests.")]
        [SerializeField] private bool _useInspectorOverride = false;

        [SerializeField] private bool   _overrideEnabled    = false;
        [SerializeField] private float  _overrideMassMsun   = 9.55e-4f;
        [SerializeField] private float  _overrideRadiusAU   = 20f;
        [SerializeField] private float  _overridePhase0     = 0f;
        [SerializeField] private float  _overrideOmega      = 0.011f;

        // ----------------------------------------------------------------
        // Internal state
        // ----------------------------------------------------------------

        private NBodyPlayer _player;
        private Mesh        _mesh;

        private bool   _enabled;
        private float  _massMsun;
        private float  _r;
        private float  _phase0;
        private float  _omega;

        private static readonly int _propDiameter = Shader.PropertyToID("_Diameter");

        [Serializable]
        private class PerturberSidecar
        {
            public bool  enabled      = false;
            public float mass_msun    = 0f;
            public float r_AU         = 0f;
            public float phase0_rad   = 0f;
            public float omega_T0_inv = 0f;
        }

        // ----------------------------------------------------------------
        // Unity messages
        // ----------------------------------------------------------------

        private void Awake()
        {
            _player = GetComponent<NBodyPlayer>();
            _player.OnFileLoaded += OnFileLoaded;
        }

        private void OnFileLoaded()
        {
            // Tear down any leftover perturber from the previous simulation —
            // critical when the user switches from a perturber run to control,
            // otherwise the green sphere from the previous run keeps rendering.
            if (_mesh != null) { Destroy(_mesh); _mesh = null; }
            _enabled = false;

            string framesPath = _player.FilePath;
            if (string.IsNullOrEmpty(framesPath))
            {
                Debug.LogWarning("[PerturberRenderer] No file path on player — perturber disabled");
                return;
            }

            if (_useInspectorOverride)
            {
                _enabled  = _overrideEnabled;
                _massMsun = _overrideMassMsun;
                _r        = _overrideRadiusAU;
                _phase0   = _overridePhase0;
                _omega    = _overrideOmega;
            }
            else
            {
                string sidecarPath = Path.Combine(
                    Path.GetDirectoryName(framesPath),
                    _sidecarFilename);

                if (!File.Exists(sidecarPath))
                {
                    Debug.Log($"[PerturberRenderer] No sidecar at {sidecarPath} — perturber disabled");
                    _enabled = false;
                    return;
                }

                try
                {
                    string json = File.ReadAllText(sidecarPath);
                    var sc = JsonUtility.FromJson<PerturberSidecar>(json);
                    _enabled  = sc.enabled;
                    _massMsun = sc.mass_msun;
                    _r        = sc.r_AU;
                    _phase0   = sc.phase0_rad;
                    _omega    = sc.omega_T0_inv;
                    Debug.Log($"[PerturberRenderer] Loaded sidecar: " +
                              $"{_massMsun:E3} M☉ at r={_r:F1} AU, omega={_omega:F4}, " +
                              $"period={(2 * Mathf.PI / _omega):F1} T0");
                }
                catch (Exception e)
                {
                    Debug.LogError($"[PerturberRenderer] Failed to read {sidecarPath}: {e.Message}");
                    _enabled = false;
                    return;
                }
            }

            if (!_enabled) { Debug.Log("[PerturberRenderer] Perturber disabled"); return; }

            // Build a single-vertex mesh
            if (_mesh != null) Destroy(_mesh);
            _mesh = new Mesh { name = "PerturberPoint" };
            _mesh.vertices = new[] { Vector3.zero };
            _mesh.SetIndices(new[] { 0 }, MeshTopology.Points, 0);
            // Large bounds — never frustum-cull this
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 80f);
        }

        private void LateUpdate()
        {
            if (!_enabled || _mesh == null || _perturberMaterial == null) return;

            // Keplerian position: phi(t) = phase0 + omega * t   (t in T0 units)
            float phi = _phase0 + _omega * (float)_player.CurrentTime;
            Vector3 pos = new Vector3(
                _r * Mathf.Cos(phi),
                0f,
                _r * Mathf.Sin(phi));

            // Size scales with cube root of mass — visually nice spread
            // mass = 1 M☉  -> ~3x scale, mass = 1e-3 -> 0.3x scale.
            float diameter = _baseDiameter * Mathf.Pow(_massMsun * 1000f, 1f / 3f);
            _perturberMaterial.SetFloat(_propDiameter, diameter);

            // Draw at computed position
            Matrix4x4 m = Matrix4x4.TRS(pos, Quaternion.identity, Vector3.one);
            Graphics.DrawMesh(_mesh, m, _perturberMaterial, 0);
        }

        private void OnDestroy()
        {
            if (_mesh != null) Destroy(_mesh);
            if (_player != null) _player.OnFileLoaded -= OnFileLoaded;
        }

    }
}
