"""
the Non-Maximum Suppression (NMS) algorithm in Python. This code assumes that you're passing in a list of bounding boxes and their associated confidence scores, 
and that each bounding box is represented as a list or tuple in the format (x1, y1, x2, y2), 
where (x1, y1) are the coordinates of the top left corner, and (x2, y2) are the coordinates of the bottom right corner.
"""

import numpy as np

def nms(boxes, scores, threshold):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Convert the boxes to floats for division
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes	
    pick = []

    # Get the coordinates of the bounding boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Compute the area of the bounding boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    idxs = np.argsort(scores)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have an overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))

    # Return only the bounding boxes that were picked using the integer data type
    return boxes[pick].astype("int")

"""
To use the nms function, simply pass in your list of bounding boxes and their scores, 
as well as the overlap threshold you want to use:
"""
boxes = np.array([[12, 84, 140, 212], [24, 84, 152, 212], [36, 84, 164, 212], [12, 96, 140, 224], [24, 96, 152, 224], [24, 108, 152, 236]])
scores = np.array([0.9, 0.75, 0.6, 0.95, 0.8, 0.65])
threshold = 0.3

result = nms(boxes, scores, threshold)
# This will print the bounding box that reselected by NMS.
print(result)
