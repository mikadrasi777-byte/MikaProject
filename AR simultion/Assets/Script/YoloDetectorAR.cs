using System;
using Unity.Collections;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class YoloDetectorAR : MonoBehaviour
{
    [Header("YOLO")]
    public ModelAsset modelAsset;

    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    [Header("AR Camera")]
    public ARCameraManager arCameraManager;

    [Tooltip("If detections look vertically flipped, toggle this.")]
    public bool flipYToScreen = false;

    public const int InputSize = 640;

    public struct Detection
    {
        public Vector2 screenCenter;  // in screen pixels
        public Rect screenRect;       // in screen pixels
        public float score;
    }

    public event Action<Detection> OnBestDetection;

    const BackendType backend = BackendType.GPUCompute;

    Worker worker;
    Tensor<float> centersToCorners;
    Tensor<float> inputTensor;

    Texture2D cameraTexture;
    RenderTexture modelRT;

    void Start()
    {
        if (modelAsset == null || arCameraManager == null)
        {
            Debug.LogError("YoloDetectorAR: Missing modelAsset or arCameraManager.");
            enabled = false;
            return;
        }

        modelRT = new RenderTexture(InputSize, InputSize, 0, RenderTextureFormat.ARGB32);
        modelRT.Create();

        inputTensor = new Tensor<float>(new TensorShape(1, 3, InputSize, InputSize));

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

        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var modelOutput = Functional.Forward(model, inputs)[0];

        // Expected: (1, 4+numClasses, numBoxes)
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1); // (numBoxes,4) cx,cy,w,h
        var allScores = modelOutput[0, 4.., ..];                  // (numClasses,numBoxes)

        var scores = Functional.ReduceMax(allScores, 0);           // (numBoxes)
        var classIDs = Functional.ArgMax(allScores, 0);            // (numBoxes) unused

        var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners));
        var indices = Functional.NMS(boxCorners, scores, iouThreshold, scoreThreshold);

        var coords = Functional.IndexSelect(boxCoords, 0, indices); // (N,4)
        var selScores = Functional.IndexSelect(scores, 0, indices); // (N)
        var labelIDs = Functional.IndexSelect(classIDs, 0, indices);// (N) unused

        worker = new Worker(graph.Compile(coords, selScores, labelIDs), backend);
    }

    void Update()
    {
        if (!TryGetARCameraTexture(out Texture src)) return;

        // Feed model 640x640
        Graphics.Blit(src, modelRT);

        TextureConverter.ToTensor(modelRT, inputTensor, default);
        worker.Schedule(inputTensor);

        using var coords = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone(); // (N,4)
        using var scores = (worker.PeekOutput("output_1") as Tensor<float>).ReadbackAndClone(); // (N)

        int n = coords.shape[0];
        if (n <= 0) return;

        // pick best by score
        int bestIdx = -1;
        float bestScore = -1f;
        for (int i = 0; i < n; i++)
        {
            float s = scores[i];
            if (s > bestScore)
            {
                bestScore = s;
                bestIdx = i;
            }
        }
        if (bestIdx < 0) return;

        float cx = coords[bestIdx, 0];
        float cy = coords[bestIdx, 1];
        float w = coords[bestIdx, 2];
        float h = coords[bestIdx, 3];

        // Convert from model pixels (0..640) to screen pixels
        Vector2 center = ModelToScreenPoint(cx, cy);
        Rect rect = ModelRectToScreenRect(cx, cy, w, h);

        OnBestDetection?.Invoke(new Detection
        {
            screenCenter = center,
            screenRect = rect,
            score = bestScore
        });
    }

    Vector2 ModelToScreenPoint(float modelX, float modelY)
    {
        float x = (modelX / InputSize) * Screen.width;
        float y01 = (modelY / InputSize);
        float y = (flipYToScreen ? (1f - y01) : y01) * Screen.height;
        return new Vector2(x, y);
    }

    Rect ModelRectToScreenRect(float cx, float cy, float w, float h)
    {
        Vector2 c = ModelToScreenPoint(cx, cy);

        float ww = (w / InputSize) * Screen.width;
        float hh = (h / InputSize) * Screen.height;

        return new Rect(c.x - ww * 0.5f, c.y - hh * 0.5f, ww, hh);
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
