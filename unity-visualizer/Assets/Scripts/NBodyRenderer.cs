using UnityEngine;
using UnityEngine.Rendering;

namespace NBodyVisualizer
{
    /// <summary>
    /// Renders the N-body particle cloud each frame using a single <see cref="Mesh"/>
    /// with <see cref="MeshTopology.Points"/> and a custom geometry-shader material.
    ///
    /// The mesh vertices are updated in <c>LateUpdate</c> from
    /// <see cref="NBodyPlayer.Positions"/>.  Each vertex carries a vertex color that
    /// encodes the particle's orbital radius (distance from Y-axis), giving a
    /// warm-inner / cool-outer color gradient.
    ///
    /// Requires the <b>NBodyParticle</b> material (assigned in Inspector) which uses
    /// the <c>NBody/Particle</c> shader.
    /// </summary>
    [RequireComponent(typeof(NBodyPlayer))]
    public class NBodyRenderer : MonoBehaviour
    {
        // ----------------------------------------------------------------
        // Inspector
        // ----------------------------------------------------------------

        [Header("Material")]
        [Tooltip("Material using the NBody/Particle shader.")]
        [SerializeField] private Material _particleMaterial;

        [Tooltip("Material using the NBody/Star shader. If null, the star is " +
                 "drawn using the particle material (legacy behaviour).")]
        [SerializeField] private Material _starMaterial;

        [Header("Appearance")]
        [Tooltip("Particle diameter in world-space AU.")]
        [Range(0.001f, 2f)]
        [SerializeField] private float _particleSize = 0.06f;

        [Tooltip("Color for particles near the disk centre (small r).")]
        [SerializeField] private Color _innerColor = new Color(1.0f, 0.75f, 0.2f, 1f); // warm gold

        [Tooltip("Color for particles at the outer disk edge.")]
        [SerializeField] private Color _outerColor = new Color(0.2f, 0.5f, 1.0f, 1f);  // cool blue

        [Tooltip("Orbital radius (AU) mapped to outerColor.  Typically disk_r_max = 5.")]
        [SerializeField] private float _colorRadiusMax = 5f;

        [Tooltip("Draw the central star as a bright sphere.")]
        [SerializeField] private bool _showStar = true;

        [SerializeField] private Color _starColor = new Color(2f, 1.8f, 0.8f, 1f);

        [Header("Aggregate-size brightness modulation (v2 frames only)")]
        [Tooltip("Brightness for singleton particles (size = 1). " +
                 "Lower = more 'transparent' look for noise. Set to 0 to hide singletons completely.")]
        [Range(0.0f, 0.5f)]
        [SerializeField] private float _minBrightness = 0.03f;

        [Tooltip("Aggregate size at which a particle reaches full brightness. " +
                 "Larger value = wider dynamic range, smaller bodies stay dim longer.")]
        [Range(8f, 2000f)]
        [SerializeField] private float _maxBrightnessAggSize = 300f;

        [Tooltip("Brightness ramp steepness. 1 = log curve (gentle), " +
                 "2 = log² (steep — small clumps stay dim until big), " +
                 "3 = log³ (very steep).")]
        [Range(1.0f, 4.0f)]
        [SerializeField] private float _brightnessGamma = 2.0f;

        // ----------------------------------------------------------------
        // Private
        // ----------------------------------------------------------------

        private NBodyPlayer _player;
        private Mesh        _mesh;
        private Vector3[]   _vertices;
        private Color[]     _colors;
        private int[]       _indices;

        // Star mesh (a single bright point at origin)
        private Mesh        _starMesh;

        private static readonly int _propSize         = Shader.PropertyToID("_ParticleSize");
        private static readonly int _propInnerColor   = Shader.PropertyToID("_InnerColor");
        private static readonly int _propOuterColor   = Shader.PropertyToID("_OuterColor");
        private static readonly int _propColorRadMax  = Shader.PropertyToID("_ColorRadiusMax");

        // ----------------------------------------------------------------
        // Unity messages
        // ----------------------------------------------------------------

        private void Awake()
        {
            _player = GetComponent<NBodyPlayer>();
            _player.OnFileLoaded += BuildMesh;
        }

        private void LateUpdate()
        {
            if (_mesh == null || !_player.IsLoaded) return;

            UpdateMesh();
            DrawMesh();
        }

        private void OnDestroy()
        {
            if (_mesh     != null) Destroy(_mesh);
            if (_starMesh != null) Destroy(_starMesh);
        }

        // ----------------------------------------------------------------
        // Mesh management
        // ----------------------------------------------------------------

        private void BuildMesh()
        {
            int n = _player.Header.NParticles;

            _vertices = new Vector3[n];
            _colors   = new Color[n];
            _indices  = new int[n];
            for (int i = 0; i < n; i++) _indices[i] = i;

            if (_mesh != null) Destroy(_mesh);
            _mesh = new Mesh { name = "NBodyParticleCloud" };
            _mesh.indexFormat = IndexFormat.UInt32; // required for n > 65535
            _mesh.vertices    = _vertices;
            _mesh.colors      = _colors;
            _mesh.SetIndices(_indices, MeshTopology.Points, 0);
            // Large bounds so the mesh is never culled
            _mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 30f);

            // Star mesh: single point at origin. Set alpha = 1 explicitly so the
            // shader's brightness modulation does not dim the star.
            if (_starMesh != null) Destroy(_starMesh);
            _starMesh = new Mesh { name = "NBodyStar" };
            _starMesh.vertices = new[] { Vector3.zero };
            _starMesh.colors   = new[] { new Color(_starColor.r, _starColor.g, _starColor.b, 1f) };
            _starMesh.SetIndices(new[] { 0 }, MeshTopology.Points, 0);
            _starMesh.bounds   = new Bounds(Vector3.zero, Vector3.one * 0.1f);
        }

        private void UpdateMesh()
        {
            var positions = _player.Positions;
            var aggSizes  = _player.AggregateSizes;     // null for v1 files
            int n         = _player.Header.NParticles;

            // Brightness curve: log( aggSize ) / log( maxAggSize ), clamped, then
            // linearly remapped from [min, 1]. Singletons → _minBrightness, sizes
            // ≥ _maxBrightnessAggSize → 1.0. log keeps a soft gradient so a
            // 10-particle clump (medium) is visibly brighter than a pair without
            // washing out the contrast on the big planets.
            float logMaxInv = 1.0f / Mathf.Log(Mathf.Max(_maxBrightnessAggSize, 2f));

            for (int i = 0; i < n; i++)
            {
                Vector3 p = positions[i];
                _vertices[i] = p;

                // Orbital radius in XZ plane (disk plane)
                float r = Mathf.Sqrt(p.x * p.x + p.z * p.z);
                float t = Mathf.Clamp01(r / _colorRadiusMax);
                Color baseColor = Color.Lerp(_innerColor, _outerColor, t);

                // Stash the aggregate brightness factor in the vertex color
                // ALPHA channel — the shader multiplies the final colour by it.
                //
                // The curve:
                //   normalized t = log(aggSize) / log(maxAggSize), clipped to [0,1]
                //   gamma t      = t^gamma   (gamma > 1 makes small clumps stay dim)
                //   factor       = lerp(minBrightness, 1, gamma t)
                //
                // Higher gamma -> dust is much darker, only large bodies pop.
                float factor;
                if (aggSizes != null && aggSizes[i] > 1)
                {
                    float logT  = Mathf.Clamp01(Mathf.Log((float)aggSizes[i]) * logMaxInv);
                    float gammaT = Mathf.Pow(logT, _brightnessGamma);
                    factor = Mathf.Lerp(_minBrightness, 1.0f, gammaT);
                }
                else
                {
                    factor = _minBrightness;
                }

                _colors[i] = new Color(baseColor.r, baseColor.g, baseColor.b, factor);
            }

            _mesh.vertices = _vertices;
            _mesh.colors   = _colors;
            // No need to recalculate bounds — we set large fixed bounds in BuildMesh
        }

        private void DrawMesh()
        {
            if (_particleMaterial == null) return;

            _particleMaterial.SetFloat(_propSize, _particleSize);
            _particleMaterial.SetColor(_propInnerColor,  _innerColor);
            _particleMaterial.SetColor(_propOuterColor,  _outerColor);
            _particleMaterial.SetFloat(_propColorRadMax, _colorRadiusMax);

            Graphics.DrawMesh(_mesh, Matrix4x4.identity, _particleMaterial, 0);

            if (_showStar && _starMesh != null)
            {
                if (_starMaterial != null)
                {
                    // Dedicated NBody/Star shader handles its own corona + pulse,
                    // sizing is baked into the material's _CoreSize / _CoronaSize.
                    Graphics.DrawMesh(_starMesh, Matrix4x4.identity, _starMaterial, 0);
                }
                else
                {
                    // Legacy fallback — draw the star with the particle material,
                    // scaled up by 6× so it's still recognisably the central body.
                    _particleMaterial.SetFloat(_propSize, _particleSize * 6f);
                    Graphics.DrawMesh(_starMesh, Matrix4x4.identity, _particleMaterial, 0);
                    _particleMaterial.SetFloat(_propSize, _particleSize);
                }
            }
        }
    }
}
