import torch
import cv2
import numpy as np
from skimage.segmentation import watershed
import copy

#
def Round(src_val):
    return int(round(src_val))

def Convert_Ellipse_BinaryMask_to_Circular_Marker(binary_mask_for_marker, inner_ratio):
    
    H, W = binary_mask_for_marker.shape
    ellipse_marker = np.zeros( (H,W), dtype= np.uint8 )
    circular_marker = np.zeros( (H,W), dtype= np.uint8 )
    inst_count = 0

    contours, _ = cv2.findContours(binary_mask_for_marker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx in range(len(contours)):
        tmp_mask = np.zeros( (H,W), dtype= np.uint8 )
        cv2.drawContours(tmp_mask, [contours[idx]], contourIdx=-1, color= 255, thickness= -1)
        non_zero_pixels = cv2.countNonZero(tmp_mask)
        if non_zero_pixels > (inner_ratio * 80):
            rect = cv2.minAreaRect(contours[idx])  # ((center_x, center_y), (width, height), angle)
            (cx, cy), (w, h), angle = rect
            cx = Round(cx); cy = Round(cy)
            w = Round(w); h = Round(h)

            shorter_side = int( min(w, h) )

            inst_count = inst_count + 1    

            cv2.circle( circular_marker, (int(cx),int(cy)), shorter_side, inst_count, thickness= -1 )
            cv2.drawContours(ellipse_marker, [contours[idx]], contourIdx=-1, color= inst_count, thickness= -1)

    return ellipse_marker, circular_marker

def ObtainContourFromInstanceMap_Smooth_Contour(instance_map):
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]

    H = instance_map.shape[0]
    W = instance_map.shape[1]
    list_of_contour_points = []

    smoothed_instance_map_by_contour = np.zeros( (H,W), dtype= np.uint8 )
    instance_count = 0
    #smoothed_instance_map_by_direct_fill = np.zeros( (H,W), dtype= np.uint8 )

    for cur_instance_id in instance_ids:
        binary_mask = np.zeros( (H, W), dtype= np.uint8 )
        binary_mask[ instance_map == cur_instance_id ] = 255
        #
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        else:
            instance_count = instance_count + 1
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour_points = [tuple(point[0]) for point in largest_contour]
            cv2.drawContours( smoothed_instance_map_by_contour, [largest_contour], 0, color= int(instance_count), thickness= -1 )
            list_of_contour_points.append(largest_contour_points)

            # debug mask
            tmp = np.zeros( (H, W), dtype= np.uint8 )
            cv2.drawContours( tmp, [largest_contour], 0, color= 255, thickness= -1 )

        #smoothed_instance_map_by_direct_fill[binary_mask == 255] = cur_instance_id

    return list_of_contour_points, smoothed_instance_map_by_contour


def Fit_Ellipse_for_A_Mask(src_mask):
    # smooth boundary
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #cleaned = cv2.morphologyEx(src_mask, cv2.MORPH_OPEN, kernel)
    #cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    # find contour
    #contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours, _ = cv2.findContours(src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ellipse = cv2.fitEllipse(contours[0])
    (cx, cy), (ma, mb), angle = ellipse

    return max(ma, mb)

def Debug_Marker_and_Instance_Ratio(ellipse_marker, circular_marker, instance_map, marker_to_instance_min_ratio):
    inst_id_list = np.unique(instance_map)[1:]  # exlcude background
    H, W = instance_map.shape

    refined_ellipse_marker  = np.zeros( (H,W), dtype= np.uint8 )
    refined_circular_marker = np.zeros( (H,W), dtype= np.uint8 )
    refined_instance_map    = np.zeros( (H,W), dtype= np.uint8 )

    refined_inst_count = 0

    for cur_inst_id in inst_id_list:
        tmp_marker_image = np.zeros((H,W), dtype= np.uint8)
        tmp_instance_image = np.zeros((H,W), dtype= np.uint8)

        tmp_marker_image[ellipse_marker == cur_inst_id] = 255
        tmp_instance_image[instance_map == cur_inst_id] = 255

        #marker_zero_pixels   = float( cv2.countNonZero(tmp_marker_image) )
        #instance_zero_pixels = float( cv2.countNonZero(tmp_instance_image) )

        marker_length = Fit_Ellipse_for_A_Mask(tmp_marker_image)
        instance_length = Fit_Ellipse_for_A_Mask(tmp_instance_image)
        marker_to_instance_length_ratio = marker_length / instance_length

        if(marker_to_instance_length_ratio > marker_to_instance_min_ratio):
            refined_inst_count = refined_inst_count + 1
            
            refined_ellipse_marker[ellipse_marker == cur_inst_id] = refined_inst_count
            refined_circular_marker[circular_marker == cur_inst_id] = refined_inst_count
            refined_instance_map[instance_map == cur_inst_id] = refined_inst_count

    return refined_ellipse_marker, refined_circular_marker, refined_instance_map


def Get_Instances_from_BinaryMask_DistMap(pred_binary_mask, pred_dist, inner_ratio= 0.4):
    # 2025_06_29 v2:
    threshold = inner_ratio * 2.0
    binary_mask_for_marker = (pred_dist <= threshold).astype(np.uint8)
    binary_mask_for_marker[pred_binary_mask < 1.0] = 0

    ellipse_marker, circular_marker = Convert_Ellipse_BinaryMask_to_Circular_Marker(binary_mask_for_marker, inner_ratio)

    watershed_labels = watershed(pred_dist, markers= circular_marker, mask= pred_binary_mask)

    # Optional
    _, watershed_labels = ObtainContourFromInstanceMap_Smooth_Contour(watershed_labels)

    refined_ellipse_marker, refined_circular_marker, refined_instance_map = Debug_Marker_and_Instance_Ratio(ellipse_marker, 
                                                                                                            circular_marker, 
                                                                                                            watershed_labels, 
                                                                                                            marker_to_instance_min_ratio= 0.25)

    return refined_instance_map.astype(np.uint8), refined_circular_marker


# prepare image input
def convert_Tensor_normalize_image(input_color_image):
    MEAN = [0.6777, 0.6777, 0.6777]
    STD  = [0.1592, 0.1592, 0.1592]
    float_color_image = input_color_image.astype(np.float32) / 255.0
    tensor_image = torch.tensor(float_color_image.transpose(2, 0, 1), dtype=torch.float32)
    mean_tensor = torch.tensor(MEAN).view(3, 1, 1)
    std_tensor = torch.tensor(STD).view(3, 1, 1)
    tensor_image = (tensor_image - mean_tensor) / std_tensor

    return tensor_image.float()


# Visualization
def ColorMapNumpyArray(src_array):
    
    norm_img = cv2.normalize(src_array, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to uint8
    norm_img_uint8 = norm_img.astype(np.uint8)
    # Apply JET colormap
    color_img = cv2.applyColorMap(norm_img_uint8, cv2.COLORMAP_JET)

    src_array_bool = src_array.astype(bool)
    color_img[~src_array_bool] = 0

    return color_img, src_array_bool

def Overlay(inst_color, mask, raw_img_color):
    gt_inst = cv2.addWeighted( raw_img_color, 0.2, inst_color, 0.8, gamma= 0.0); 
    raw_img_color_copy = copy.deepcopy(raw_img_color)

    raw_img_color_copy[mask] = gt_inst[mask]
    return raw_img_color_copy