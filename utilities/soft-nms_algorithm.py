"""
the Soft-NMS algorithm. This code assumes that the bounding boxes are represented as a list of arrays, 
where each array contains the coordinates in the format [x1, y1, x2, y2, score], and (x1, y1) and (x2, y2) are the coordinates of the top left and bottom right corners of the bounding box, respectively.
"""

import numpy as np

def soft_nms(boxes: np.array, sigma=0.5, Nt=0.3, threshold=0.001, method=2):
    # N: number of boxes
    N = boxes.shape[0]
    
    # iterate through all the boxes
    for i in range(N):
        # initialize maximum score for the current box
        maxscore = boxes[i, 4]
        maxpos = i

        # find box with maximum score
        for pos in range(i+1, N):
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos

        # swap box with highest score to the top
        boxes[[i, maxpos]] = boxes[[maxpos, i]]
        
        # start soft-NMS iterations for boxes i+1,...,N
        for pos in range(i+1, N):
            # calculate intersection over union (IoU)
            ix1 = np.maximum(boxes[i, 0], boxes[pos, 0])
            iy1 = np.maximum(boxes[i, 1], boxes[pos, 1])
            ix2 = np.minimum(boxes[i, 2], boxes[pos, 2])
            iy2 = np.minimum(boxes[i, 3], boxes[pos, 3])
            
            iw = np.maximum(ix2 - ix1 + 1., 0.)
            ih = np.maximum(iy2 - iy1 + 1., 0.)
            
            inters = iw * ih

            # compute union
            uni = ((boxes[i, 2] - boxes[i, 0] + 1.) * (boxes[i, 3] - boxes[i, 1] + 1.) +
                   (boxes[pos, 2] - boxes[pos, 0] + 1.) *
                   (boxes[pos, 3] - boxes[pos, 1] + 1.) - inters)

            ov = inters / uni

            # apply soft-NMS based on method: 1=linear, 2=gaussian, other=traditional NMS
            if method == 1:  # linear
                if ov > Nt:
                    weight = 1 - ov
                else:
                    weight = 1
            elif method == 2:  # gaussian
                weight = np.exp(-(ov * ov) / sigma)
            else:  # original NMS
                if ov > Nt:
                    weight = 0
                else:
                    weight = 1
            boxes[pos, 4] *= weight

            # if box score falls below threshold, discard the box by swapping with the last box
            # and decrementing N, the number of boxes
            if boxes[pos, 4] < threshold:
                boxes[[pos, N - 1]] = boxes[[N - 1, pos]]
                N -= 1
                pos -= 1

    # return indices of the boxes that are kept
    keep = [i for i in range(N)]
    return keep

"""
You can call this function with your list of bounding boxes as a NumPy array. The output will be the indices of the boxes that were kept after Soft-NMS.
"""
boxes = np.array([[12, 84, 140, 212, 0.9], 
                  [24, 84, 152, 212, 0.75], 
                  [36, 84, 164, 212, 0.6], 
                  [12, 96, 140, 224, 0.95], 
                  [24, 96, 152, 224, 0.8], 
                  [24, 108, 152, 236, 0.65]])

result_indices = soft_nms(boxes)
print("Boxes kept:")
for i in result_indices:
    print(boxes[i])

