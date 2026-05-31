// NBodySetupWizard.cs
// Automates the full scene setup for the NBody Visualizer.
//
// Interactive use:    Unity menu  NBody → Setup Scene
// Batch/CI use:       Unity.exe -executeMethod NBodySetupWizard.RunSetup -batchmode -quit

using System;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEditor;
using UnityEditor.SceneManagement;
using NBodyVisualizer;
using TMPro;

public static class NBodySetupWizard
{
    private const string ScenePath    = "Assets/Scenes/Visualizer.unity";
    private const string MaterialPath = "Assets/Materials/ParticleMaterial.mat";

    // ────────────────────────────────────────────────────────────────────
    // Entry points
    // ────────────────────────────────────────────────────────────────────

    /// <summary>Called via Unity menu NBody → Setup Scene.</summary>
    [MenuItem("NBody/Setup Scene")]
    public static void SetupSceneMenu()
    {
        if (EditorUtility.DisplayDialog(
            "NBody Visualizer Setup",
            "This will create:\n" +
            "  • Assets/Materials/ParticleMaterial.mat\n" +
            "  • Assets/Scenes/Visualizer.unity\n\n" +
            "Existing files with those names will be overwritten.",
            "OK — Set Up", "Cancel"))
        {
            RunSetup();
            EditorUtility.DisplayDialog("Done",
                "Scene ready!\n\n" +
                "1. In the Simulation GameObject, set File Path to your frames.bin.\n" +
                "2. Press ▶ Play.\n" +
                "3. Use Left-mouse to orbit, scroll to zoom.",
                "Got it");
        }
    }

    /// <summary>Static entry point for -executeMethod (batch mode).</summary>
    public static void RunSetup()
    {
        try
        {
            EnsureDirectories();
            var mat   = CreateMaterial();
            var scene = CreateScene();
            CreateCamera(scene);
            CreateSimulation(scene, mat);
            CreateUI(scene);
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("[NBodySetupWizard] ✓ Setup complete. Scene saved to " + ScenePath);
        }
        catch (Exception e)
        {
            Debug.LogError("[NBodySetupWizard] Setup failed: " + e);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 1 — Directories
    // ────────────────────────────────────────────────────────────────────

    private static void EnsureDirectories()
    {
        foreach (var dir in new[] { "Assets/Scenes", "Assets/Materials" })
        {
            if (!AssetDatabase.IsValidFolder(dir))
            {
                var parts = dir.Split('/');
                AssetDatabase.CreateFolder(parts[0], parts[1]);
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 2 — Material
    // ────────────────────────────────────────────────────────────────────

    private static Material CreateMaterial()
    {
        // Wait for shader to be compiled/imported
        AssetDatabase.Refresh();

        var shader = Shader.Find("NBody/Particle");
        if (shader == null)
        {
            Debug.LogWarning("[NBodySetupWizard] Shader 'NBody/Particle' not found yet. " +
                             "It will be assigned on the next domain reload.");
        }

        var mat = new Material(shader != null ? shader : Shader.Find("Unlit/Color"))
        {
            name = "ParticleMaterial"
        };

        if (shader != null)
        {
            mat.SetFloat("_Glow",       1.8f);
            mat.SetFloat("_Brightness", 1.0f);
        }

        // Overwrite if exists
        if (File.Exists(Path.Combine(Application.dataPath, "../", MaterialPath)))
            AssetDatabase.DeleteAsset(MaterialPath);

        AssetDatabase.CreateAsset(mat, MaterialPath);
        Debug.Log("[NBodySetupWizard] Material created: " + MaterialPath);
        return mat;
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 3 — Scene
    // ────────────────────────────────────────────────────────────────────

    private static UnityEngine.SceneManagement.Scene CreateScene()
    {
        var scene = EditorSceneManager.NewScene(
            NewSceneSetup.EmptyScene,
            NewSceneMode.Single);

        // Dark ambient light (space background feel)
        RenderSettings.ambientMode  = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.05f, 0.05f, 0.1f);
        RenderSettings.skybox       = null;
        RenderSettings.fog          = false;

        return scene;
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 4 — Camera
    // ────────────────────────────────────────────────────────────────────

    private static void CreateCamera(UnityEngine.SceneManagement.Scene scene)
    {
        var go  = new GameObject("Main Camera");
        go.tag  = "MainCamera";

        var cam = go.AddComponent<Camera>();
        cam.clearFlags       = CameraClearFlags.SolidColor;
        cam.backgroundColor  = Color.black;
        cam.fieldOfView      = 60f;
        cam.nearClipPlane    = 0.01f;
        cam.farClipPlane     = 1000f;

        go.AddComponent<AudioListener>();
        go.AddComponent<OrbitCamera>();

        // Position: 55° pitch, 18 AU back
        go.transform.position = new Vector3(0f, 12.5f, -14f);
        go.transform.LookAt(Vector3.zero);

        UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(go, scene);
        Debug.Log("[NBodySetupWizard] Camera created.");
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 5 — Simulation
    // ────────────────────────────────────────────────────────────────────

    private static void CreateSimulation(
        UnityEngine.SceneManagement.Scene scene, Material mat)
    {
        var go = new GameObject("Simulation");
        UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(go, scene);

        // ---- NBodyPlayer ----
        var player = go.AddComponent<NBodyPlayer>();
        // Leave FilePath empty; user sets it in Inspector or at runtime via UI
        SetPrivateField(player, "_playbackSpeed",  20f);
        SetPrivateField(player, "_loop",           true);
        SetPrivateField(player, "_autoPlay",        true);

        // ---- NBodyRenderer ----
        var renderer = go.AddComponent<NBodyRenderer>();
        SetPrivateField(renderer, "_particleMaterial", mat);
        SetPrivateField(renderer, "_particleSize",     0.06f);
        SetPrivateField(renderer, "_colorRadiusMax",   5f);
        SetPrivateField(renderer, "_showStar",         true);

        Debug.Log("[NBodySetupWizard] Simulation GameObject created.");
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 6 — UI Canvas
    // ────────────────────────────────────────────────────────────────────

    private static void CreateUI(UnityEngine.SceneManagement.Scene scene)
    {
        // ── EventSystem ──────────────────────────────────────────────
        var esGO = new GameObject("EventSystem",
            typeof(EventSystem), typeof(StandaloneInputModule));
        UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(esGO, scene);

        // ── Canvas ───────────────────────────────────────────────────
        var canvasGO = new GameObject("Canvas");
        UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(canvasGO, scene);

        var canvas = canvasGO.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 10;

        var scaler = canvasGO.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.matchWidthOrHeight  = 0.5f;

        canvasGO.AddComponent<GraphicRaycaster>();

        // ── HUD Panel (bottom of screen) ─────────────────────────────
        var panelGO = CreateUIPanel(canvasGO,  "HUD",
            new Vector2(0f, 0f), new Vector2(1f, 0f),
            new Vector2(0f, 0f), new Vector2(0f, 120f));
        var panelImg = panelGO.GetComponent<Image>();
        panelImg.color = new Color(0, 0, 0, 0.6f);

        // ── Time label ───────────────────────────────────────────────
        var timeLabelGO = CreateTMPText(panelGO, "TimeLabel",
            "t = 0.00 T₀   (0.000 yr)",
            new Vector2(0f, 0.5f), new Vector2(0.5f, 0.5f),
            new Vector2(10f, 10f), new Vector2(-10f, -10f),
            18, TextAlignmentOptions.BottomLeft);

        // ── Frame label ──────────────────────────────────────────────
        var frameLabelGO = CreateTMPText(panelGO, "FrameLabel",
            "Frame 0 / 0",
            new Vector2(0.5f, 0.5f), new Vector2(1f, 0.5f),
            new Vector2(10f, 10f), new Vector2(-10f, -10f),
            18, TextAlignmentOptions.BottomRight);

        // ── Timeline slider ──────────────────────────────────────────
        var sliderGO = CreateSlider(panelGO, "TimelineSlider",
            new Vector2(0.05f, 0.55f), new Vector2(0.95f, 0.55f),
            new Vector2(0f, 20f), new Vector2(0f, -20f), 0f, 1f, 0f);

        // ── Play/Pause button ────────────────────────────────────────
        var playBtnGO = CreateButton(panelGO, "PlayPauseButton", "▶",
            new Vector2(0.45f, 0f), new Vector2(0.55f, 0f),
            new Vector2(0f, 5f), new Vector2(0f, 55f));

        // ── Speed slider ─────────────────────────────────────────────
        var speedSliderGO = CreateSlider(panelGO, "SpeedSlider",
            new Vector2(0.65f, 0f), new Vector2(0.98f, 0f),
            new Vector2(0f, 5f), new Vector2(0f, 40f), 0.1f, 200f, 20f);

        var speedLabelGO = CreateTMPText(panelGO, "SpeedLabel",
            "Speed: 20 T₀/s",
            new Vector2(0.65f, 0f), new Vector2(0.98f, 0f),
            new Vector2(0f, 45f), new Vector2(0f, 70f),
            14, TextAlignmentOptions.Center);

        // ── File path row ────────────────────────────────────────────
        var fileInputGO   = CreateInputField(panelGO, "FilePathInput",
            "Path to frames.bin …",
            new Vector2(0.01f, 1f), new Vector2(0.82f, 1f),
            new Vector2(0f, -60f), new Vector2(0f, -10f));

        var loadBtnGO = CreateButton(panelGO, "LoadButton", "Load",
            new Vector2(0.83f, 1f), new Vector2(0.99f, 1f),
            new Vector2(0f, -60f), new Vector2(0f, -10f));

        // ── Status label ─────────────────────────────────────────────
        var statusLabelGO = CreateTMPText(panelGO, "StatusLabel",
            "Set File Path and click Load, or set it in the Simulation Inspector.",
            new Vector2(0f, 1f), new Vector2(1f, 1f),
            new Vector2(10f, -80f), new Vector2(-10f, -62f),
            12, TextAlignmentOptions.Center);

        var statusText = statusLabelGO.GetComponent<TMP_Text>();
        statusText.color = new Color(0.7f, 0.7f, 0.7f);

        // ── Wire up NBodyUI ──────────────────────────────────────────
        // Find the Simulation GO in the scene
        var simGO = GameObject.Find("Simulation");

        var ui = canvasGO.AddComponent<NBodyUI>();
        SetSerializedRef(ui, "_player",         simGO?.GetComponent<NBodyPlayer>());
        SetSerializedRef(ui, "_timeLabel",       timeLabelGO.GetComponent<TMP_Text>());
        SetSerializedRef(ui, "_frameLabel",      frameLabelGO.GetComponent<TMP_Text>());
        SetSerializedRef(ui, "_playPauseButton", playBtnGO.GetComponent<Button>());
        SetSerializedRef(ui, "_playPauseLabel",  playBtnGO.GetComponentInChildren<TMP_Text>());
        SetSerializedRef(ui, "_timelineSlider",  sliderGO.GetComponent<Slider>());
        SetSerializedRef(ui, "_speedSlider",     speedSliderGO.GetComponent<Slider>());
        SetSerializedRef(ui, "_speedLabel",      speedLabelGO.GetComponent<TMP_Text>());
        SetSerializedRef(ui, "_filePathInput",   fileInputGO.GetComponent<TMP_InputField>());
        SetSerializedRef(ui, "_loadButton",      loadBtnGO.GetComponent<Button>());
        SetSerializedRef(ui, "_statusLabel",     statusLabelGO.GetComponent<TMP_Text>());

        Debug.Log("[NBodySetupWizard] UI Canvas created.");
    }

    // ────────────────────────────────────────────────────────────────────
    // UI helpers
    // ────────────────────────────────────────────────────────────────────

    private static GameObject CreateUIPanel(GameObject parent, string name,
        Vector2 anchorMin, Vector2 anchorMax,
        Vector2 offsetMin, Vector2 offsetMax)
    {
        var go  = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        go.transform.SetParent(parent.transform, false);
        var rt  = go.GetComponent<RectTransform>();
        rt.anchorMin  = anchorMin;
        rt.anchorMax  = anchorMax;
        rt.offsetMin  = offsetMin;
        rt.offsetMax  = offsetMax;
        go.GetComponent<Image>().color = new Color(0, 0, 0, 0);
        return go;
    }

    private static GameObject CreateTMPText(GameObject parent, string name,
        string text, Vector2 anchorMin, Vector2 anchorMax,
        Vector2 offsetMin, Vector2 offsetMax,
        float fontSize, TextAlignmentOptions alignment)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer));
        go.transform.SetParent(parent.transform, false);
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = anchorMin; rt.anchorMax = anchorMax;
        rt.offsetMin = offsetMin; rt.offsetMax = offsetMax;

        var tmp = go.AddComponent<TextMeshProUGUI>();
        tmp.text      = text;
        tmp.fontSize  = fontSize;
        tmp.color     = Color.white;
        tmp.alignment = alignment;
        return go;
    }

    private static GameObject CreateButton(GameObject parent, string name,
        string label, Vector2 anchorMin, Vector2 anchorMax,
        Vector2 offsetMin, Vector2 offsetMax)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        go.transform.SetParent(parent.transform, false);
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = anchorMin; rt.anchorMax = anchorMax;
        rt.offsetMin = offsetMin; rt.offsetMax = offsetMax;

        var img = go.GetComponent<Image>();
        img.color = new Color(0.2f, 0.2f, 0.3f, 0.9f);

        var btn = go.AddComponent<Button>();
        ColorBlock cb = btn.colors;
        cb.highlightedColor = new Color(0.35f, 0.35f, 0.5f);
        cb.pressedColor     = new Color(0.1f, 0.1f, 0.2f);
        btn.colors = cb;

        var textGO = new GameObject("Label", typeof(RectTransform), typeof(CanvasRenderer));
        textGO.transform.SetParent(go.transform, false);
        var textRT = textGO.GetComponent<RectTransform>();
        textRT.anchorMin = Vector2.zero; textRT.anchorMax = Vector2.one;
        textRT.offsetMin = textRT.offsetMax = Vector2.zero;

        var tmp = textGO.AddComponent<TextMeshProUGUI>();
        tmp.text      = label;
        tmp.fontSize  = 18;
        tmp.color     = Color.white;
        tmp.alignment = TextAlignmentOptions.Center;

        btn.targetGraphic = img;
        return go;
    }

    private static GameObject CreateSlider(GameObject parent, string name,
        Vector2 anchorMin, Vector2 anchorMax,
        Vector2 offsetMin, Vector2 offsetMax,
        float minVal, float maxVal, float defaultVal)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(parent.transform, false);
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = anchorMin; rt.anchorMax = anchorMax;
        rt.offsetMin = offsetMin; rt.offsetMax = offsetMax;

        // Background
        var bg = new GameObject("Background", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        bg.transform.SetParent(go.transform, false);
        var bgRT = bg.GetComponent<RectTransform>();
        bgRT.anchorMin = new Vector2(0, 0.25f); bgRT.anchorMax = new Vector2(1, 0.75f);
        bgRT.offsetMin = bgRT.offsetMax = Vector2.zero;
        bg.GetComponent<Image>().color = new Color(0.15f, 0.15f, 0.2f);

        // Fill area
        var fillArea = new GameObject("Fill Area", typeof(RectTransform));
        fillArea.transform.SetParent(go.transform, false);
        var faRT = fillArea.GetComponent<RectTransform>();
        faRT.anchorMin = new Vector2(0, 0.25f); faRT.anchorMax = new Vector2(1, 0.75f);
        faRT.offsetMin = new Vector2(5, 0); faRT.offsetMax = new Vector2(-15, 0);

        var fill = new GameObject("Fill", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        fill.transform.SetParent(fillArea.transform, false);
        var fillRT = fill.GetComponent<RectTransform>();
        fillRT.anchorMin = Vector2.zero; fillRT.anchorMax = new Vector2(0, 1);
        fillRT.offsetMin = fillRT.offsetMax = Vector2.zero;
        fill.GetComponent<Image>().color = new Color(0.3f, 0.5f, 1f);

        // Handle slide area
        var handleArea = new GameObject("Handle Slide Area", typeof(RectTransform));
        handleArea.transform.SetParent(go.transform, false);
        var haRT = handleArea.GetComponent<RectTransform>();
        haRT.anchorMin = Vector2.zero; haRT.anchorMax = Vector2.one;
        haRT.offsetMin = new Vector2(10, 0); haRT.offsetMax = new Vector2(-10, 0);

        var handle = new GameObject("Handle", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        handle.transform.SetParent(handleArea.transform, false);
        var handleRT = handle.GetComponent<RectTransform>();
        handleRT.anchorMin = new Vector2(0, 0); handleRT.anchorMax = new Vector2(0, 1);
        handleRT.sizeDelta = new Vector2(20, 0);
        handle.GetComponent<Image>().color = new Color(0.6f, 0.8f, 1f);

        var slider = go.AddComponent<Slider>();
        slider.fillRect   = fillRT;
        slider.handleRect = handleRT;
        slider.minValue   = minVal;
        slider.maxValue   = maxVal;
        slider.value      = defaultVal;
        slider.targetGraphic = handle.GetComponent<Image>();

        return go;
    }

    private static GameObject CreateInputField(GameObject parent, string name,
        string placeholder, Vector2 anchorMin, Vector2 anchorMax,
        Vector2 offsetMin, Vector2 offsetMax)
    {
        var go = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        go.transform.SetParent(parent.transform, false);
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = anchorMin; rt.anchorMax = anchorMax;
        rt.offsetMin = offsetMin; rt.offsetMax = offsetMax;
        go.GetComponent<Image>().color = new Color(0.1f, 0.1f, 0.15f, 0.9f);

        var textArea = new GameObject("Text Area", typeof(RectTransform));
        textArea.transform.SetParent(go.transform, false);
        var taRT = textArea.GetComponent<RectTransform>();
        taRT.anchorMin = Vector2.zero; taRT.anchorMax = Vector2.one;
        taRT.offsetMin = new Vector2(8, 4); taRT.offsetMax = new Vector2(-8, -4);

        var phGO = new GameObject("Placeholder", typeof(RectTransform), typeof(CanvasRenderer));
        phGO.transform.SetParent(textArea.transform, false);
        var phRT = phGO.GetComponent<RectTransform>();
        phRT.anchorMin = Vector2.zero; phRT.anchorMax = Vector2.one;
        phRT.offsetMin = phRT.offsetMax = Vector2.zero;
        var phTmp = phGO.AddComponent<TextMeshProUGUI>();
        phTmp.text      = placeholder;
        phTmp.fontSize  = 14;
        phTmp.color     = new Color(0.5f, 0.5f, 0.5f);
        phTmp.fontStyle = FontStyles.Italic;

        var textGO = new GameObject("Text", typeof(RectTransform), typeof(CanvasRenderer));
        textGO.transform.SetParent(textArea.transform, false);
        var textRT = textGO.GetComponent<RectTransform>();
        textRT.anchorMin = Vector2.zero; textRT.anchorMax = Vector2.one;
        textRT.offsetMin = textRT.offsetMax = Vector2.zero;
        var textTmp = textGO.AddComponent<TextMeshProUGUI>();
        textTmp.fontSize = 14;
        textTmp.color    = Color.white;

        var input = go.AddComponent<TMP_InputField>();
        input.textViewport   = taRT;
        input.textComponent  = textTmp;
        input.placeholder    = phTmp;
        input.targetGraphic  = go.GetComponent<Image>();

        return go;
    }

    // ────────────────────────────────────────────────────────────────────
    // Reflection helpers for private [SerializeField] fields
    // ────────────────────────────────────────────────────────────────────

    private static void SetPrivateField(object target, string fieldName, object value)
    {
        var so   = new SerializedObject(target as UnityEngine.Object);
        var prop = so.FindProperty(fieldName);
        if (prop == null)
        {
            Debug.LogWarning($"[NBodySetupWizard] Field not found: {fieldName}");
            return;
        }

        switch (prop.propertyType)
        {
            case SerializedPropertyType.Float:         prop.floatValue  = (float)value;   break;
            case SerializedPropertyType.Boolean:       prop.boolValue   = (bool)value;    break;
            case SerializedPropertyType.String:        prop.stringValue = (string)value;  break;
            case SerializedPropertyType.Integer:       prop.intValue    = (int)value;     break;
            default:
                Debug.LogWarning($"[NBodySetupWizard] Unhandled property type for {fieldName}: {prop.propertyType}");
                break;
        }

        so.ApplyModifiedPropertiesWithoutUndo();
    }

    private static void SetSerializedRef(object target, string fieldName, UnityEngine.Object value)
    {
        if (value == null) return;
        var so   = new SerializedObject(target as UnityEngine.Object);
        var prop = so.FindProperty(fieldName);
        if (prop == null) { Debug.LogWarning($"[NBodySetupWizard] Ref field not found: {fieldName}"); return; }
        prop.objectReferenceValue = value;
        so.ApplyModifiedPropertiesWithoutUndo();
    }
}
