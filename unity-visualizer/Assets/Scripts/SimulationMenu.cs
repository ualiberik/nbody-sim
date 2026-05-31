using System;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

namespace NBodyVisualizer
{
    /// <summary>
    /// Runtime UI menu for switching between the six simulations. Generates
    /// its own Canvas + buttons in <see cref="Start"/> — drop this component
    /// on any GameObject that has an <see cref="NBodyPlayer"/> and it just
    /// works.
    ///
    /// Each button calls <see cref="NBodyPlayer.LoadFile"/> with the
    /// corresponding <c>frames.bin</c> path. The <see cref="PerturberRenderer"/>
    /// (if present) auto-reloads via the <c>OnFileLoaded</c> event.
    /// </summary>
    [RequireComponent(typeof(NBodyPlayer))]
    public class SimulationMenu : MonoBehaviour
    {
        // ------------------------------------------------------------------
        // Inspector
        // ------------------------------------------------------------------

        [Header("Data root")]
        [Tooltip("Folder that contains all run subfolders. Each entry below " +
                 "is joined to this path. Override here if you move the data.")]
        [SerializeField]
        private string _dataRoot =
            @"C:\Projects\nbody-sim\.worktrees\feature-nbody-simulation\data";

        [Header("Simulations to expose (label, subfolder)")]
        [SerializeField]
        private SimEntry[] _simulations = new SimEntry[]
        {
            new SimEntry { label = "Контроль (без гиганта)",  folder = "20260525_202643" },
            new SimEntry { label = "0.3 M_Jup на 20 а.е.",    folder = "20260526_000833" },
            new SimEntry { label = "1.0 M_Jup на 20 а.е.",    folder = "20260526_015429" },
            new SimEntry { label = "3.0 M_Jup на 20 а.е.",    folder = "20260526_034408" },
            new SimEntry { label = "5.0 M_Jup на 20 а.е.",    folder = "20260526_053325" },
            new SimEntry { label = "10  M_Jup на 20 а.е.",    folder = "20260526_073428" },
        };

        [Header("Appearance")]
        [SerializeField] private int    _fontSize    = 16;
        [SerializeField] private float  _buttonWidth = 240f;
        [SerializeField] private float  _buttonHeight = 36f;
        [SerializeField] private float  _gap         = 6f;
        [SerializeField] private Color  _normalColor    = new Color(0.10f, 0.13f, 0.22f, 0.85f);
        [SerializeField] private Color  _hoverColor     = new Color(0.20f, 0.30f, 0.55f, 0.95f);
        [SerializeField] private Color  _activeColor    = new Color(0.30f, 0.60f, 0.90f, 1.0f);
        [SerializeField] private Color  _textColor      = Color.white;

        // ------------------------------------------------------------------

        [Serializable]
        public class SimEntry
        {
            public string label;
            public string folder;
        }

        private NBodyPlayer _player;
        private Button[]    _buttons;
        private int         _activeIndex = -1;
        private Text        _statusText;

        // ------------------------------------------------------------------
        // Unity messages
        // ------------------------------------------------------------------

        private void Start()
        {
            _player = GetComponent<NBodyPlayer>();
            EnsureEventSystem();
            BuildUI();
        }

        // ------------------------------------------------------------------
        // UI generation
        // ------------------------------------------------------------------

        private void EnsureEventSystem()
        {
            if (FindObjectOfType<EventSystem>() == null)
            {
                var go = new GameObject("EventSystem");
                go.AddComponent<EventSystem>();
                go.AddComponent<StandaloneInputModule>();
            }
        }

        private void BuildUI()
        {
            // Canvas
            var canvasGo = new GameObject("SimMenuCanvas");
            canvasGo.transform.SetParent(transform);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 100;
            canvasGo.AddComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ConstantPixelSize;
            canvasGo.AddComponent<GraphicRaycaster>();

            // Panel — top-left vertical stack
            var panelGo = new GameObject("Panel");
            panelGo.transform.SetParent(canvasGo.transform, false);
            var panelImg = panelGo.AddComponent<Image>();
            panelImg.color = new Color(0, 0, 0, 0.35f);
            var panelRt = panelGo.GetComponent<RectTransform>();
            panelRt.anchorMin = new Vector2(0, 1);
            panelRt.anchorMax = new Vector2(0, 1);
            panelRt.pivot     = new Vector2(0, 1);
            panelRt.anchoredPosition = new Vector2(20, -20);

            float totalH = _simulations.Length * (_buttonHeight + _gap) + 40 + 24; // +title +status
            panelRt.sizeDelta = new Vector2(_buttonWidth + 20, totalH);

            // Title
            var titleGo = new GameObject("Title");
            titleGo.transform.SetParent(panelGo.transform, false);
            var titleTxt = titleGo.AddComponent<Text>();
            titleTxt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            titleTxt.fontSize = _fontSize + 2;
            titleTxt.color = Color.white;
            titleTxt.alignment = TextAnchor.MiddleCenter;
            titleTxt.text = "Симуляция:";
            titleTxt.fontStyle = FontStyle.Bold;
            var titleRt = titleGo.GetComponent<RectTransform>();
            titleRt.anchorMin = new Vector2(0, 1);
            titleRt.anchorMax = new Vector2(1, 1);
            titleRt.pivot     = new Vector2(0.5f, 1);
            titleRt.anchoredPosition = new Vector2(0, -8);
            titleRt.sizeDelta = new Vector2(0, 24);

            // Buttons
            _buttons = new Button[_simulations.Length];
            for (int i = 0; i < _simulations.Length; i++)
            {
                int idx = i;
                var btnGo = new GameObject($"Btn_{i}");
                btnGo.transform.SetParent(panelGo.transform, false);

                var img = btnGo.AddComponent<Image>();
                img.color = _normalColor;

                var btn = btnGo.AddComponent<Button>();
                var colors = btn.colors;
                colors.normalColor      = _normalColor;
                colors.highlightedColor = _hoverColor;
                colors.pressedColor     = _activeColor;
                colors.selectedColor    = _hoverColor;
                btn.colors = colors;
                btn.onClick.AddListener(() => OnButtonClicked(idx));

                // Label
                var lblGo = new GameObject("Label");
                lblGo.transform.SetParent(btnGo.transform, false);
                var lblTxt = lblGo.AddComponent<Text>();
                lblTxt.font  = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
                lblTxt.fontSize = _fontSize;
                lblTxt.color = _textColor;
                lblTxt.alignment = TextAnchor.MiddleCenter;
                lblTxt.text  = _simulations[i].label;
                var lblRt = lblGo.GetComponent<RectTransform>();
                lblRt.anchorMin = Vector2.zero;
                lblRt.anchorMax = Vector2.one;
                lblRt.offsetMin = Vector2.zero;
                lblRt.offsetMax = Vector2.zero;

                // Position
                var btnRt = btnGo.GetComponent<RectTransform>();
                btnRt.anchorMin = new Vector2(0, 1);
                btnRt.anchorMax = new Vector2(1, 1);
                btnRt.pivot     = new Vector2(0.5f, 1);
                float y = -36 - i * (_buttonHeight + _gap);
                btnRt.anchoredPosition = new Vector2(0, y);
                btnRt.sizeDelta = new Vector2(-10, _buttonHeight);

                _buttons[i] = btn;
            }

            // Status line at the bottom of panel
            var statusGo = new GameObject("Status");
            statusGo.transform.SetParent(panelGo.transform, false);
            _statusText = statusGo.AddComponent<Text>();
            _statusText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _statusText.fontSize = _fontSize - 2;
            _statusText.color = new Color(1, 1, 1, 0.7f);
            _statusText.alignment = TextAnchor.MiddleCenter;
            _statusText.text = "(ничего не загружено)";
            var sRt = statusGo.GetComponent<RectTransform>();
            sRt.anchorMin = new Vector2(0, 0);
            sRt.anchorMax = new Vector2(1, 0);
            sRt.pivot     = new Vector2(0.5f, 0);
            sRt.anchoredPosition = new Vector2(0, 5);
            sRt.sizeDelta = new Vector2(0, 18);
        }

        // ------------------------------------------------------------------
        // Click handling
        // ------------------------------------------------------------------

        private void OnButtonClicked(int idx)
        {
            if (idx < 0 || idx >= _simulations.Length) return;
            var entry = _simulations[idx];
            string path = Path.Combine(_dataRoot, entry.folder, "frames.bin");

            if (!File.Exists(path))
            {
                Debug.LogError($"[SimulationMenu] File not found: {path}");
                _statusText.text = "ОШИБКА: файл не найден";
                _statusText.color = new Color(1, 0.4f, 0.4f, 0.9f);
                return;
            }

            _player.LoadFile(path);
            _activeIndex = idx;
            UpdateButtonHighlights();
            _statusText.text = "Активно: " + entry.label;
            _statusText.color = new Color(0.5f, 1.0f, 0.5f, 0.9f);
        }

        private void UpdateButtonHighlights()
        {
            for (int i = 0; i < _buttons.Length; i++)
            {
                var img = _buttons[i].GetComponent<Image>();
                img.color = (i == _activeIndex) ? _activeColor : _normalColor;
            }
        }
    }
}
