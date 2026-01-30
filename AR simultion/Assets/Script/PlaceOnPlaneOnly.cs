using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class PlaceOnPlaneOnly : MonoBehaviour
{
    [Header("AR")]
    public ARRaycastManager raycastManager;
    public ARAnchorManager anchorManager;

    [Header("UI")]
    public GraphicRaycaster raycaster;
    public Toggle editToggle; // ON = אפשר לערוך, OFF = נעול

    [Header("Prefabs (Random per Game)")]
    public GameObject[] prefabs;

    [Header("Selection")]
    [Tooltip("איזה Layer נמצא ה-Prefab שלך בזמן ריצה. ברירת מחדל: Default.")]
    public LayerMask placedObjectLayerMask = ~0; // Everything by default

    private static readonly List<ARRaycastHit> _hits = new List<ARRaycastHit>();

    private int _chosenIndex = -1;
    private GameObject _spawnedObject;
    private ARAnchor _anchor;

    void Start()
    {
        ChooseRandomPrefabForThisGame();

        if (editToggle != null)
            editToggle.onValueChanged.AddListener(_ => ApplyEditState());
    }

    void OnDestroy()
    {
        if (editToggle != null)
            editToggle.onValueChanged.RemoveAllListeners();
    }

    void Update()
    {
        if (!Input.GetMouseButtonDown(0))
            return;

        if (IsClickOverUI())
            return;

        // 1) אם עוד לא הוצב אובייקט – הצבה ראשונה על Plane
        if (_spawnedObject == null)
        {
            TryPlaceFirstTime();
            ApplyEditState(); // יכבה/ידליק את רכיבי העריכה לפי הטוגל
            return;
        }

        // מכאן והלאה: יש כבר אובייקט בסצנה

        // אם מצב עריכה כבוי - לא עושים כלום
        if (!IsEditEnabled())
            return;

        // 2) אם לחצנו על האובייקט עצמו - נותנים ל-XRI/ARGestureInteractor לטפל (Selection + Gestures)
        //    כלומר: לא מזיזים עוגן ולא עושים Raycast לרצפה, כדי לא "לגנוב" את המחווה/בחירה
        if (IsPointerOverSpawnedObject())
        {
            // לא עושים כאן כלום בכוונה.
            // ARSelectionInteractable + ARGestureInteractor יבחרו ויאפשרו Rotate/Scale/Translate.
            return;
        }

        // 3) אם לחצנו על הרצפה (ולא על האובייקט) - אפשר להזיז אותו למיקום חדש על ה-plane
        TryRepositionOnPlane();
    }

    private void TryPlaceFirstTime()
    {
        if (_chosenIndex < 0 || prefabs == null || prefabs.Length == 0)
            return;

        if (!raycastManager.Raycast(Input.mousePosition, _hits,
                TrackableType.PlaneEstimated | TrackableType.PlaneWithinPolygon))
            return;

        Pose pose = _hits[0].pose;

        _anchor = anchorManager != null ? anchorManager.AddAnchor(pose) : null;

        GameObject prefab = prefabs[_chosenIndex];

        if (_anchor != null)
        {
            _spawnedObject = Instantiate(prefab, _anchor.transform);
            _spawnedObject.transform.localPosition = Vector3.zero;
            _spawnedObject.transform.localRotation = Quaternion.identity;
        }
        else
        {
            _spawnedObject = Instantiate(prefab, pose.position, pose.rotation);
        }
    }

    private void TryRepositionOnPlane()
    {
        if (!raycastManager.Raycast(Input.mousePosition, _hits,
                TrackableType.PlaneEstimated | TrackableType.PlaneWithinPolygon))
            return;

        Pose pose = _hits[0].pose;

        // עדיף להצמיד ע"י Anchor חדש
        if (anchorManager != null)
        {
            if (_anchor != null)
                Destroy(_anchor.gameObject);

            _anchor = anchorManager.AddAnchor(pose);

            if (_anchor != null)
            {
                _spawnedObject.transform.SetParent(_anchor.transform, false);
                _spawnedObject.transform.localPosition = Vector3.zero;
                _spawnedObject.transform.localRotation = Quaternion.identity;
                return;
            }
        }

        // fallback
        _spawnedObject.transform.SetParent(null);
        _spawnedObject.transform.SetPositionAndRotation(pose.position, pose.rotation);
    }

    private bool IsEditEnabled()
    {
        return editToggle != null && editToggle.isOn;
    }

    private void ApplyEditState()
    {
        if (_spawnedObject == null)
            return;

        bool enabled = IsEditEnabled();

        // מכבים/מדליקים את רכיבי העריכה על האובייקט
        // (שמות הקומפוננטות הללו קיימים כשיש XRI AR Interactions)
        var sel = _spawnedObject.GetComponent<UnityEngine.XR.Interaction.Toolkit.AR.ARSelectionInteractable>();
        if (sel != null) sel.enabled = enabled;

        var tr = _spawnedObject.GetComponent<UnityEngine.XR.Interaction.Toolkit.AR.ARTranslationInteractable>();
        if (tr != null) tr.enabled = enabled;

        var rot = _spawnedObject.GetComponent<UnityEngine.XR.Interaction.Toolkit.AR.ARRotationInteractable>();
        if (rot != null) rot.enabled = enabled;

        var sc = _spawnedObject.GetComponent<UnityEngine.XR.Interaction.Toolkit.AR.ARScaleInteractable>();
        if (sc != null) sc.enabled = enabled;
    }

    private bool IsPointerOverSpawnedObject()
    {
        if (_spawnedObject == null)
            return false;

        // בודקים Raycast פיזיקלי מול ה-Colliders של האובייקט
        var cam = Camera.main;
        if (cam == null)
            return false;

        Ray ray = cam.ScreenPointToRay(Input.mousePosition);
        if (Physics.Raycast(ray, out RaycastHit hit, 100f, placedObjectLayerMask))
        {
            // האם פגענו באובייקט שלנו או בילד שלו
            return hit.transform == _spawnedObject.transform || hit.transform.IsChildOf(_spawnedObject.transform);
        }

        return false;
    }

    public void StartNewGame()
    {
        if (_spawnedObject != null)
        {
            Destroy(_spawnedObject);
            _spawnedObject = null;
        }

        if (_anchor != null)
        {
            Destroy(_anchor.gameObject);
            _anchor = null;
        }

        ChooseRandomPrefabForThisGame();
    }

    private void ChooseRandomPrefabForThisGame()
    {
        if (prefabs == null || prefabs.Length == 0)
        {
            _chosenIndex = -1;
            return;
        }

        _chosenIndex = Random.Range(0, prefabs.Length);
    }

    private bool IsClickOverUI()
    {
        if (raycaster == null || EventSystem.current == null)
            return false;

        var data = new PointerEventData(EventSystem.current) { position = Input.mousePosition };
        var results = new List<RaycastResult>();
        raycaster.Raycast(data, results);
        return results.Count > 0;
    }
}
