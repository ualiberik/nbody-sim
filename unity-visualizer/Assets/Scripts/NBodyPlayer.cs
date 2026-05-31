using System;
using UnityEngine;

namespace NBodyVisualizer
{
    /// <summary>
    /// MonoBehaviour that loads a <b>frames.bin</b> file and drives frame-by-frame
    /// playback.  Attach to any GameObject; set <see cref="FilePath"/> in the
    /// Inspector or call <see cref="LoadFile"/> at runtime.
    ///
    /// Physical time units: T0 ≈ 58.1 days = 1/(2π) years (G=1, M_star=1 M☉, r=1 AU).
    /// </summary>
    public class NBodyPlayer : MonoBehaviour
    {
        // ----------------------------------------------------------------
        // Inspector
        // ----------------------------------------------------------------

        [Header("File")]
        [Tooltip("Absolute path to frames.bin. You can also call LoadFile() at runtime.")]
        [SerializeField] private string _filePath = "";

        [Header("Playback")]
        [Tooltip("Simulated T0 units advanced per real second.")]
        [SerializeField] private float _playbackSpeed = 20f;

        [Tooltip("Loop back to frame 0 when the last frame is reached.")]
        [SerializeField] private bool _loop = true;

        [Tooltip("Start playing immediately after the file is loaded.")]
        [SerializeField] private bool _autoPlay = true;

        // ----------------------------------------------------------------
        // Public properties
        // ----------------------------------------------------------------

        /// <summary>Current playback speed in T0/s.  Can be changed at runtime.</summary>
        public float PlaybackSpeed
        {
            get => _playbackSpeed;
            set => _playbackSpeed = Mathf.Max(0.001f, value);
        }

        public bool   IsPlaying    { get; private set; }
        public int    FrameCount   { get; private set; }
        public int    CurrentFrame { get; private set; }

        /// <summary>Simulation time of the current frame in T0.</summary>
        public double CurrentTime  { get; private set; }

        /// <summary>Current frame positions in AU (x/y/z).  Updated every frame.</summary>
        public Vector3[] Positions { get; private set; }

        /// <summary>
        /// Per-particle aggregate size for the current frame. Element i is the
        /// number of particles in the aggregate containing particle i (1 if
        /// the particle is a singleton). null if the file does not contain
        /// aggregate info (v1 format).
        /// </summary>
        public ushort[] AggregateSizes { get; private set; }

        /// <summary>File header info (valid after a file has been loaded).</summary>
        public NBodyReader.Header Header { get; private set; }

        /// <summary>True if a file has been successfully loaded.</summary>
        public bool IsLoaded { get; private set; }

        // ----------------------------------------------------------------
        // Events
        // ----------------------------------------------------------------

        /// <summary>Fired once when a file finishes loading.</summary>
        public event Action OnFileLoaded;

        /// <summary>Fired every time the displayed frame changes.</summary>
        public event Action<int> OnFrameChanged;

        // ----------------------------------------------------------------
        // Private
        // ----------------------------------------------------------------

        private NBodyReader _reader;

        // Fractional sim-time cursor in T0 units (drives frame selection)
        private double _simTimeCursor;

        // ----------------------------------------------------------------
        // Unity messages
        // ----------------------------------------------------------------

        private void Start()
        {
            if (!string.IsNullOrEmpty(_filePath))
                LoadFile(_filePath);
        }

        private void Update()
        {
            if (!IsLoaded || !IsPlaying || FrameCount <= 1) return;

            _simTimeCursor += _playbackSpeed * Time.deltaTime;

            double totalDuration = FrameCount * Header.OutputDt;
            if (_simTimeCursor >= totalDuration)
            {
                if (_loop) _simTimeCursor %= totalDuration;
                else       { _simTimeCursor = totalDuration - 1e-9; IsPlaying = false; }
            }

            int newFrame = Mathf.Clamp(
                (int)(_simTimeCursor / Header.OutputDt), 0, FrameCount - 1);

            if (newFrame != CurrentFrame)
                SetFrame(newFrame);
        }

        private void OnDestroy() => _reader?.Dispose();

        // ----------------------------------------------------------------
        // Public API
        // ----------------------------------------------------------------

        /// <summary>Absolute path of the most recently loaded frames.bin.</summary>
        public string FilePath => _filePath;

        /// <summary>Load a frames.bin file.  Safe to call multiple times.</summary>
        public void LoadFile(string path)
        {
            try
            {
                _reader?.Dispose();
                _reader = new NBodyReader(path);
            }
            catch (Exception e)
            {
                Debug.LogError($"[NBodyPlayer] Failed to open '{path}': {e.Message}");
                IsLoaded = false;
                return;
            }

            // Remember the active path so listeners (PerturberRenderer, UI)
            // can locate sidecar files next to it after a switch.
            _filePath = path;

            Header     = _reader.FileHeader;
            FrameCount = Header.NFrames;
            Positions  = new Vector3[Header.NParticles];
            AggregateSizes = _reader.HasAggregateSizes
                ? new ushort[Header.NParticles]
                : null;

            _simTimeCursor = 0;
            SetFrame(0);
            IsLoaded  = true;
            IsPlaying = _autoPlay;

            OnFileLoaded?.Invoke();

            double t0Years = Header.OutputDt / (2.0 * Math.PI);
            Debug.Log(
                $"[NBodyPlayer] Loaded '{path}'\n" +
                $"  Particles : {Header.NParticles:N0}\n" +
                $"  Frames    : {Header.NFrames:N0}\n" +
                $"  output_dt : {Header.OutputDt:F4} T0  ({t0Years:F4} yr)\n" +
                $"  Total time: {FrameCount * Header.OutputDt:F1} T0  " +
                $"({FrameCount * t0Years:F2} yr)"
            );
        }

        public void Play()       => IsPlaying = true;
        public void Pause()      => IsPlaying = false;
        public void TogglePlay() => IsPlaying = !IsPlaying;

        /// <summary>Seek to a specific frame index.</summary>
        public void SeekToFrame(int frame)
        {
            if (!IsLoaded) return;
            SetFrame(Mathf.Clamp(frame, 0, FrameCount - 1));
            _simTimeCursor = CurrentFrame * Header.OutputDt;
        }

        /// <summary>Seek to a normalized position [0, 1] along the timeline.</summary>
        public void SeekToNormalized(float t)
            => SeekToFrame((int)(t * (FrameCount - 1)));

        // ----------------------------------------------------------------
        // Helpers
        // ----------------------------------------------------------------

        private void SetFrame(int frame)
        {
            CurrentFrame = frame;
            CurrentTime  = _reader.ReadFrame(frame, Positions, AggregateSizes);
            OnFrameChanged?.Invoke(CurrentFrame);
        }
    }
}
