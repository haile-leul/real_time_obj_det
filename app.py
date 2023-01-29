import sys
import cv2
import yolov5
import numpy as np

if len(sys.argv)>1:
    s = sys.argv[1]

alive = True
source = cv2.VideoCapture(0)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Object detection model definition

# load model
model = yolov5.load('fcakyon/yolov5s-v7.0')
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = True  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# img = '/content/people-walking-through-business-district-in-the-city-at-sunset_bbxoqvaod_thumbnail-1080_09.png'

while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame,1)


    # perform inference
    results = model(frame)

    # inference with larger input size
    results = model(frame, size=640)

    # inference with test time augmentation
    results = model(frame, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # Get the indices of the bounding boxes with the highest scores
    best_boxes = np.argmax(scores)

    # Draw the bounding boxes on the image
    for i, box in enumerate(boxes):
        # box = boxes[i]
        score = scores[i]
        category = categories[i]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text = f"Score: {score:.2f}, Category: {category}"
        cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow(win_name, frame)

    key = cv2.waitKey(1)

    if key == 27:
        alive = False

source.release()
cv2.destroyWindow(win_name)