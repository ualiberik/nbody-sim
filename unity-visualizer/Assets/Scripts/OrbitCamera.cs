using UnityEngine;

namespace NBodyVisualizer
{
    /// <summary>
    /// Orbital camera controller.
    ///   • Left mouse drag  — rotate around target
    ///   • Right mouse drag — pan target point
    ///   • Scroll wheel     — zoom (dolly in/out)
    ///
    /// Attach to the Main Camera.  Set <see cref="Target"/> to the simulation
    /// origin or leave at world-space zero.
    /// </summary>
    public class OrbitCamera : MonoBehaviour
    {
        [Header("Target")]
        [Tooltip("World-space point the camera orbits around.")]
        public Vector3 Target = Vector3.zero;

        [Header("Initial pose")]
        [SerializeField] private float _initialDistance  = 18f;
        [SerializeField] private float _initialPitch     = 55f;   // degrees above horizon
        [SerializeField] private float _initialYaw       = 0f;

        [Header("Orbit")]
        [SerializeField] private float _orbitSensitivity = 0.4f;

        [Header("Pan")]
        [SerializeField] private float _panSensitivity   = 0.02f;

        [Header("Zoom")]
        [SerializeField] private float _zoomSensitivity  = 1.2f;
        [SerializeField] private float _minDistance      = 0.5f;
        [SerializeField] private float _maxDistance      = 80f;

        [Header("Smoothing")]
        [SerializeField] private float _smoothing        = 10f;

        // ----------------------------------------------------------------

        private float _yaw;
        private float _pitch;
        private float _distance;

        private float _targetYaw;
        private float _targetPitch;
        private float _targetDistance;
        private Vector3 _targetTarget;

        private Vector2 _prevMousePos;

        // ----------------------------------------------------------------

        private void Start()
        {
            _yaw      = _targetYaw      = _initialYaw;
            _pitch    = _targetPitch    = _initialPitch;
            _distance = _targetDistance = _initialDistance;
            _targetTarget = Target;
            ApplyPose();
        }

        private void LateUpdate()
        {
            HandleInput();
            SmoothApply();
        }

        // ----------------------------------------------------------------

        private void HandleInput()
        {
            Vector2 mouse    = new Vector2(Input.mousePosition.x, Input.mousePosition.y);
            Vector2 delta    = mouse - _prevMousePos;
            _prevMousePos    = mouse;

            bool overUI = UnityEngine.EventSystems.EventSystem.current != null &&
                          UnityEngine.EventSystems.EventSystem.current.IsPointerOverGameObject();

            if (!overUI)
            {
                // Left mouse: orbit
                if (Input.GetMouseButton(0))
                {
                    _targetYaw   += delta.x * _orbitSensitivity;
                    _targetPitch -= delta.y * _orbitSensitivity;
                    _targetPitch  = Mathf.Clamp(_targetPitch, 5f, 89f);
                }

                // Right mouse: pan (move target in camera's XY plane)
                if (Input.GetMouseButton(1))
                {
                    float scale = _distance * _panSensitivity;
                    Vector3 right = transform.right;
                    Vector3 up    = transform.up;
                    _targetTarget -= (right * delta.x + up * delta.y) * scale;
                }

                // Scroll: zoom
                float scroll = Input.GetAxis("Mouse ScrollWheel");
                if (Mathf.Abs(scroll) > 1e-5f)
                {
                    _targetDistance *= Mathf.Pow(0.85f, scroll * _zoomSensitivity * 10f);
                    _targetDistance  = Mathf.Clamp(_targetDistance, _minDistance, _maxDistance);
                }
            }
        }

        private void SmoothApply()
        {
            float t = Mathf.Min(1f, _smoothing * Time.deltaTime);

            _yaw      = Mathf.LerpAngle(_yaw,      _targetYaw,      t);
            _pitch    = Mathf.Lerp(_pitch,          _targetPitch,    t);
            _distance = Mathf.Lerp(_distance,       _targetDistance, t);
            Target    = Vector3.Lerp(Target,         _targetTarget,   t);

            ApplyPose();
        }

        private void ApplyPose()
        {
            Quaternion rot = Quaternion.Euler(_pitch, _yaw, 0f);
            Vector3 offset = rot * new Vector3(0f, 0f, -_distance);
            transform.position = Target + offset;
            transform.LookAt(Target, Vector3.up);
        }
    }
}
