using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace NBodyVisualizer
{
    /// <summary>
    /// HUD controller: wires the <see cref="NBodyPlayer"/> to the on-screen UI.
    ///
    /// Required UI hierarchy (create manually or via the README setup instructions):
    ///
    ///   Canvas
    ///   └── Panel (NBodyUI attached here)
    ///       ├── TimeLabel        TMP_Text
    ///       ├── FrameLabel       TMP_Text
    ///       ├── ParticleLabel    TMP_Text
    ///       ├── PlayPauseButton  Button + TMP_Text child
    ///       ├── TimelineSlider   Slider   (value 0–1)
    ///       ├── SpeedSlider      Slider   (value 0.1–200)
    ///       ├── SpeedLabel       TMP_Text
    ///       ├── FilePathInput    TMP_InputField
    ///       └── LoadButton       Button
    /// </summary>
    public class NBodyUI : MonoBehaviour
    {
        // ----------------------------------------------------------------
        // Inspector references (drag in from Hierarchy)
        // ----------------------------------------------------------------

        [Header("Player")]
        [SerializeField] private NBodyPlayer _player;

        [Header("Labels")]
        [SerializeField] private TMP_Text _timeLabel;
        [SerializeField] private TMP_Text _frameLabel;
        [SerializeField] private TMP_Text _particleLabel;

        [Header("Transport")]
        [SerializeField] private Button   _playPauseButton;
        [SerializeField] private TMP_Text _playPauseLabel;   // child of button
        [SerializeField] private Slider   _timelineSlider;

        [Header("Speed")]
        [SerializeField] private Slider   _speedSlider;
        [SerializeField] private TMP_Text _speedLabel;

        [Header("File Loading")]
        [SerializeField] private TMP_InputField _filePathInput;
        [SerializeField] private Button         _loadButton;
        [SerializeField] private TMP_Text       _statusLabel;

        // ----------------------------------------------------------------
        // Private
        // ----------------------------------------------------------------

        private bool _sliderDrag;

        // Conversion: 1 T0 = 1/(2π) years
        private const double T0ToYears = 1.0 / (2.0 * Math.PI);

        // ----------------------------------------------------------------
        // Unity messages
        // ----------------------------------------------------------------

        private void Start()
        {
            // Transport
            if (_playPauseButton) _playPauseButton.onClick.AddListener(OnPlayPause);

            // Timeline slider
            if (_timelineSlider)
            {
                _timelineSlider.minValue = 0f;
                _timelineSlider.maxValue = 1f;
                _timelineSlider.onValueChanged.AddListener(OnTimelineChanged);
            }

            // Speed slider: range 0.1 – 200 T0/s (logarithmic feel)
            if (_speedSlider)
            {
                _speedSlider.minValue = 0.1f;
                _speedSlider.maxValue = 200f;
                _speedSlider.value    = _player ? _player.PlaybackSpeed : 20f;
                _speedSlider.onValueChanged.AddListener(OnSpeedChanged);
                UpdateSpeedLabel(_speedSlider.value);
            }

            // File loading
            if (_loadButton)  _loadButton.onClick.AddListener(OnLoad);

            // Subscribe to player events
            if (_player)
            {
                _player.OnFileLoaded    += OnFileLoaded;
                _player.OnFrameChanged  += _ => RefreshLabels();
            }

            RefreshLabels();
        }

        private void Update()
        {
            // Keep play/pause label in sync
            if (_playPauseLabel && _player)
                _playPauseLabel.text = _player.IsPlaying ? "⏸" : "▶";

            // Keep timeline slider in sync (skip when user is dragging)
            if (_timelineSlider && !_sliderDrag && _player != null && _player.FrameCount > 1)
                _timelineSlider.SetValueWithoutNotify(
                    (float)_player.CurrentFrame / (_player.FrameCount - 1));
        }

        // ----------------------------------------------------------------
        // Event handlers
        // ----------------------------------------------------------------

        private void OnPlayPause() => _player?.TogglePlay();

        private void OnTimelineChanged(float value)
        {
            if (_player == null) return;
            _sliderDrag = true;
            _player.SeekToNormalized(value);
            _sliderDrag = false;
        }

        private void OnSpeedChanged(float value)
        {
            if (_player) _player.PlaybackSpeed = value;
            UpdateSpeedLabel(value);
        }

        private void OnLoad()
        {
            string path = _filePathInput ? _filePathInput.text.Trim() : "";
            if (string.IsNullOrEmpty(path))
            {
                SetStatus("⚠ Enter a path to frames.bin first.", Color.yellow);
                return;
            }
            SetStatus("Loading...", Color.white);
            _player?.LoadFile(path);
        }

        private void OnFileLoaded()
        {
            if (_player == null) return;
            SetStatus($"✓ {_player.Header.NParticles:N0} particles · {_player.Header.NFrames:N0} frames",
                      new Color(0.4f, 1f, 0.4f));
            RefreshLabels();
        }

        // ----------------------------------------------------------------
        // Label updates
        // ----------------------------------------------------------------

        private void RefreshLabels()
        {
            if (_player == null || !_player.IsLoaded) return;

            double t0    = _player.CurrentTime;
            double years = t0 * T0ToYears;

            if (_timeLabel)
                _timeLabel.text = $"t = {t0:F2} T₀   ({years:F3} yr)";

            if (_frameLabel)
                _frameLabel.text = $"Frame {_player.CurrentFrame + 1:N0} / {_player.FrameCount:N0}";

            if (_particleLabel)
                _particleLabel.text = $"{_player.Header.NParticles:N0} particles";
        }

        private void UpdateSpeedLabel(float value)
        {
            if (_speedLabel)
                _speedLabel.text = $"Speed: {value:F1} T₀/s";
        }

        private void SetStatus(string msg, Color color)
        {
            if (_statusLabel)
            {
                _statusLabel.text  = msg;
                _statusLabel.color = color;
            }
        }
    }
}
