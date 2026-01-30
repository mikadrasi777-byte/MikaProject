using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.UIElements;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class PlaceContent : MonoBehaviour
{
    public ARRaycastManager raycastManager;
    public GraphicRaycaster raycaster;

    public void Update()
    {
        if (Input.GetMouseButtonDown(0) && !IsClickOverUI()) { 
        List <ARRaycastHit> hitPoints = new List <ARRaycastHit>();
            raycastManager.Raycast(Input.mousePosition, hitPoints, TrackableType.Planes);

            if (hitPoints.Count > 0)
            {

                Pose pose = hitPoints[0].pose;
                transform.position = pose.position;
                transform.rotation = pose.rotation;
            }
        
        }
    }

    bool IsClickOverUI()
    {
        PointerEventData data = new PointerEventData(EventSystem.current) {
            position = Input.mousePosition
        };
        List<RaycastResult> results = new List<RaycastResult>();
        raycaster.Raycast(data, results);
        return results.Count > 0;
    }


}
