#!/usr/bin/env python3
"""
This script builds and uses YOLOv3 for object detection.
"""

import tensorflow.keras as K
import numpy as np
import os
import cv2


class Yolo:
    """
    YOLO class for object detection.
    """
    model = None
    class_names = None
    class_t = None
    nms_t = None
    anchors = None

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize YOLO attributes.

        Parameters:
        - model_path: Path to the trained YOLO model.
        - classes_path: Path to the file containing class names.
        - class_t: Class score threshold for filtering boxes.
        - nms_t: IoU threshold for non-max suppression.
        - anchors: Array of anchor boxes.
        """
        # Load class names from the classes file
        with open(classes_path, 'r') as file:
            classes_list = [line.strip() for line in file if line.strip()]
        # Load the YOLO model
        self.model = K.models.load_model(model_path)
        # Assign the class names, threshold values, and anchors
        self.class_names = classes_list
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs from the YOLO model to get the bounding boxes.
        Using https://pjreddie.com/media/files/papers/YOLOv3.pdf as reference.

        Parameters:
        - outputs: Outputs from the YOLO model.
        - image_size: Size of the original image.

        Returns:
        - boxes: Bounding boxes for detected objects.
        - box_confidences: Confidences for the bounding boxes.
        - box_class_probs: Class probabilities for the bounding boxes.
        """
        def sigmoid(x):
            """Sigmoid activation function."""
            return 1 / (1 + np.exp(-x))
        
        boxes, box_confidences, box_class_probs = [], [], []
        
        for index, output in enumerate(outputs):
            grid_h, grid_w = output.shape[:2]
            # Extract confidence scores and class probabilities
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))
            box = np.zeros_like(output[..., :4])
            
            # Extract bounding box parameters
            t_x, t_y, t_w, t_h = output[..., 0], output[..., 1], output[..., 2], output[..., 3]
            pw, ph = self.anchors[index, :, 0], self.anchors[index, :, 1]
            
            # Create the grid for center coordinates
            cx = np.tile(np.arange(grid_w), grid_h).reshape(grid_h, grid_w, 1)
            cy = np.tile(np.arange(grid_h), grid_w).reshape(grid_h, grid_w).T.reshape(grid_h, grid_w, 1)
            
            # Calculate the bounding box positions and dimensions
            bx = sigmoid(t_x) + cx
            by = sigmoid(t_y) + cy
            bw = np.exp(t_w) * pw
            bh = np.exp(t_h) * ph
            
            # Normalize coordinates
            bx /= grid_w
            by /= grid_h
            bw /= self.model.input.shape[1]
            bh /= self.model.input.shape[2]
            
            # Convert to corner coordinates
            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]
            
            box[..., 0], box[..., 1] = x1, y1
            box[..., 2], box[..., 3] = x2, y2
            boxes.append(box)
        
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter the boxes based on the class score threshold.

        Parameters:
        - boxes: Bounding boxes for detected objects.
        - box_confidences: Confidences for the bounding boxes.
        - box_class_probs: Class probabilities for the bounding boxes.

        Returns:
        - filtered_boxes: Filtered bounding boxes.
        - box_classes: Classes for the filtered bounding boxes.
        - box_scores: Scores for the filtered bounding boxes.
        """
        filtered_boxes, box_classes, box_scores = [], [], []
        
        for i in range(len(boxes)):
            # Calculate class scores
            scores = box_confidences[i] * box_class_probs[i]
            box_class_scores = np.max(scores, axis=-1)
            positions = np.where(box_class_scores >= self.class_t)
            
            filtered_boxes.append(boxes[i][positions])
            box_classes.append(np.argmax(box_class_probs[i], axis=-1)[positions])
            box_scores.append(box_class_scores[positions])
        
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)
        
        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-Max Suppression to remove overlapping boxes.

        Parameters:
        - filtered_boxes: Filtered bounding boxes.
        - box_classes: Classes for the filtered bounding boxes.
        - box_scores: Scores for the filtered bounding boxes.

        Returns:
        - selected_boxes: Bounding boxes after non-max suppression.
        - selected_classes: Classes for the selected bounding boxes.
        - selected_scores: Scores for the selected bounding boxes.
        """
        def calculate_iou(box, boxes):
            """
            Calculate Intersection over Union (IoU) between a box and a list of boxes.

            Parameters:
            - box: Single bounding box.
            - boxes: List of bounding boxes to compare with.

            Returns:
            - iou: IoU values for the box with each box in boxes.
            """
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            inter_x1 = np.maximum(box[0], boxes[:, 0])
            inter_y1 = np.maximum(box[1], boxes[:, 1])
            inter_x2 = np.minimum(box[2], boxes[:, 2])
            inter_y2 = np.minimum(box[3], boxes[:, 3])
            
            inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
            iou = inter_area / (box_area + boxes_area - inter_area + 1e-9)
            
            return iou

        selected_boxes, selected_classes, selected_scores = [], [], []
        
        for class_idx in np.unique(box_classes):
            indices = np.where(box_classes == class_idx)
            class_boxes = filtered_boxes[indices]
            class_scores = box_scores[indices]
            sorted_indices = np.argsort(class_scores)[::-1]
            
            while len(sorted_indices) > 0:
                idx = sorted_indices[0]
                selected_boxes.append(class_boxes[idx])
                selected_classes.append(class_idx)
                selected_scores.append(class_scores[idx])
                
                sorted_indices = sorted_indices[1:]
                remaining_boxes = class_boxes[sorted_indices]
                iou = calculate_iou(class_boxes[idx], remaining_boxes)
                
                sorted_indices = sorted_indices[iou < self.nms_t]
        
        return np.array(selected_boxes), np.array(selected_classes), np.array(selected_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Load images from a specified folder.

        Parameters:
        - folder_path: Path to the folder containing images.

        Returns:
        - images: List of loaded images.
        - image_paths: List of image paths.
        """
        images, image_paths = [], []
        
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                image_paths.append(image_path)
        
        return images, image_paths
    
    def preprocess_images(self, images):
        """
        Preprocess the images to the required input shape for the YOLO model.

        Parameters:
        - images: List of images to preprocess.

        Returns:
        - pimages: Preprocessed images.
        - image_shapes: Original shapes of the images.
        """
        image_list, image_shapes = [], []
        
        for image in images:
            # Resize image to model input size
            resized_image = cv2.resize(image,
                                       (self.model.input.shape[1],
                                        self.model.input.shape[2]),
                                       interpolation=cv2.INTER_CUBIC)
            # Normalize image pixels
            rescaled_image = resized_image.astype(np.float32) / 255
            image_shapes.append(image.shape[:2])
            image_list.append(rescaled_image)
        
        return np.array(image_list), np.array(image_shapes)
