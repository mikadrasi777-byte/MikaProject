using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class RunYOLO : MonoBehaviour
{
    [Header("Model")]
    public ModelAsset modelAsset;

    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    [Header("AR Camera")]
    public ARCameraManager arCameraManager;

    [Header("UI Overlay (on top of AR Background)")]
    [Tooltip("RectTransform that covers the whole screen (e.g., Canvas root/panel).")]
    public RectTransform uiRoot;

    [Tooltip("Border texture used for the bounding box (preferably 9-sliced sprite).")]
    public Texture2D borderTexture;

    const BackendType backend = BackendType.GPUCompute;

    const int imageWidth = 640;
    const int imageHeight = 640;

    Worker worker;
    Tensor<float> centersToCorners;

    Texture2D cameraTexture;
    RenderTexture modelRT;

    Tensor<float> inputTensor;

    // Box drawing
    Sprite borderSprite;
    readonly List<GameObject> boxPool = new();

    struct BoundingBox
    {
        public float cx, cy, w, h; // in model (640x640) pixel space
    }

    void Start()
    {
        if (arCameraManager == null || uiRoot == null || borderTexture == null || modelAsset == null)
        {
            Debug.LogError("RunYOLO_AR_Overlay: Missing references in Inspector.");
            enabled = false;
            return;
        }

        // 640x640 render target for the model
        modelRT = new RenderTexture(imageWidth, imageHeight, 0, RenderTextureFormat.ARGB32);
        modelRT.Create();

        // Input tensor (NCHW)
        inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));

        // UI sprite (make sure your texture import settings allow Sprite if you want slicing)
        borderSprite = Sprite.Create(
            borderTexture,
            new Rect(0, 0, borderTexture.width, borderTexture.height),
            new Vector2(0.5f, 0.5f)
        );

        LoadModel();
    }

    void LoadModel()
    {
        var model = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(
            new TensorShape(4, 4),
            new float[]
            {
                1,      0,      1,      0,
                0,      1,      0,      1,
                -0.5f,  0,      0.5f,   0,
                0,      -0.5f,  0,      0.5f
            }
        );

        // NMS postprocess graph
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var modelOutput = Functional.Forward(model, inputs)[0];

        // Assumes YOLO-like output: (1, 4+numClasses, numBoxes)
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1); // (numBoxes,4) = (cx,cy,w,h)
        var allScores = modelOutput[0, 4.., ..];                  // (numClasses,numBoxes)

        var scores = Functional.ReduceMax(allScores, 0);           // (numBoxes)
        var classIDs = Functional.ArgMax(allScores, 0);            // (numBoxes)  // not used for names

        var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners)); // (numBoxes,4)
        var indices = Functional.NMS(boxCorners, scores, iouThreshold, scoreThreshold);      // (N)

        var coords = Functional.IndexSelect(boxCoords, 0, indices); // (N,4)
        var labelIDs = Functional.IndexSelect(classIDs, 0, indices);// (N) unused (kept for compilation shape)

        worker = new Worker(graph.Compile(coords, labelIDs), backend);
    }

    void Update()
    {
        ExecuteML();
    }

    void ExecuteML()
    {
        ClearAnnotations();

        if (!TryGetARCameraTexture(out Texture src))
            return;

        // Feed the model with a 640x640 image.
        // Simple stretch into 640x640 (YOLO boxes are in that space).
        Graphics.Blit(src, modelRT);

        TextureConverter.ToTensor(modelRT, inputTensor, default);
        worker.Schedule(inputTensor);

        using var output = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        // output_1 label IDs exists but not needed since "no name"
        // using var labelIDs = (worker.PeekOutput("output_1") as Tensor<int>).ReadbackAndClone();

        int nBoxes = output.shape[0];
        for (int i = 0; i < Mathf.Min(nBoxes, 200); i++)
        {
            var b = new BoundingBox
            {
                cx = output[i, 0],
                cy = output[i, 1],
                w = output[i, 2],
                h = output[i, 3],
            };

            DrawBoxOnScreen(b, i);
        }
    }

    bool TryGetARCameraTexture(out Texture tex)
    {
        tex = null;

        if (!arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage cpuImage))
            return false;

        using (cpuImage)
        {
            if (cameraTexture == null || cameraTexture.width != cpuImage.width || cameraTexture.height != cpuImage.height)
                cameraTexture = new Texture2D(cpuImage.width, cpuImage.height, TextureFormat.RGBA32, false);

            var conversionParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
                outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
                outputFormat = TextureFormat.RGBA32,
                transformation = XRCpuImage.Transformation.MirrorY
            };

            int size = cpuImage.GetConvertedDataSize(conversionParams);
            var buffer = new NativeArray<byte>(size, Allocator.Temp);

            try
            {
                cpuImage.Convert(conversionParams, buffer);
                cameraTexture.LoadRawTextureData(buffer);
                cameraTexture.Apply();
            }
            finally
            {
                buffer.Dispose();
            }

            tex = cameraTexture;
            return true;
        }
    }

    // Map 640x640 model-space boxes to full-screen UI overlay (Fit to screen aspect)
    void DrawBoxOnScreen(BoundingBox box, int id)
    {
        // Get overlay rect in pixels (Canvas overlay usually matches Screen, but we derive from uiRoot)
        float screenW = uiRoot.rect.width;
        float screenH = uiRoot.rect.height;

        // We "fit" the square (640x640) into the screen rect without distortion:
        // scale = min(screenW/640, screenH/640)
        float scale = Mathf.Min(screenW / imageWidth, screenH / imageHeight);

        float fittedW = imageWidth * scale;
        float fittedH = imageHeight * scale;

        float offsetX = (screenW - fittedW) * 0.5f;
        float offsetY = (screenH - fittedH) * 0.5f;

        // Model coords are in 640 space. Convert to fitted space then to uiRoot local space (centered)
        float cx = offsetX + box.cx * scale;
        float cy = offsetY + box.cy * scale;
        float w = box.w * scale;
        float h = box.h * scale;

        // Convert top-left origin? Our UI local uses center pivot.
        // Here we treat (0,0) at bottom-left in fitted rect. If your boxes are flipped vertically, swap cy:
        // cy = offsetY + (imageHeight - box.cy) * scale;
        // Keep as-is first; if inverted, use the line above.

        GameObject go = GetOrCreateBox(id);
        RectTransform rt = go.GetComponent<RectTransform>();

        // UI local position where (0,0) is center of uiRoot
        float localX = cx - screenW * 0.5f;
        float localY = cy - screenH * 0.5f;

        rt.anchoredPosition = new Vector2(localX, localY);
        rt.sizeDelta = new Vector2(w, h);
    }

    GameObject GetOrCreateBox(int id)
    {
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
            return panel;
        }

        panel = new GameObject("Box");
        panel.AddComponent<CanvasRenderer>();

        var img = panel.AddComponent<Image>();
        img.color = Color.yellow;
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;

        var rt = panel.GetComponent<RectTransform>();
        rt.SetParent(uiRoot, false);
        rt.anchorMin = new Vector2(0.5f, 0.5f);
        rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.pivot = new Vector2(0.5f, 0.5f);

        boxPool.Add(panel);
        return panel;
    }

    void ClearAnnotations()
    {
        for (int i = 0; i < boxPool.Count; i++)
            boxPool[i].SetActive(false);
    }

    void OnDestroy()
    {
        centersToCorners?.Dispose();
        worker?.Dispose();
        inputTensor?.Dispose();

        if (modelRT != null)
        {
            modelRT.Release();
            modelRT = null;
        }
    }
}
