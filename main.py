import cv2
import numpy as np
import copy
import time

FPS = 30
videoCapture = cv2.VideoCapture("input.mp4")
outputVideo = "filtered4.mp4"
blurStrength = 141
dilationStrength = 61
downscaleFactor = 8
detectOnlyOnce = False
detectInterval = 10 # Frames

maskRCNNModel = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "coco.txt")
COCOLabels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball","kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket","bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

# https://docs.opencv.org/4.x/db/df6/tutorial_erosion_dilatation.html
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilationStrength, dilationStrength))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None
labels = None
lastDetect = detectInterval
while True:
    start = time.time()
    ret, frame = videoCapture.read()
    if ret == False:
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    if out == None:
        out = cv2.VideoWriter(outputVideo, fourcc, FPS, (frame.shape[1], frame.shape[0]))

    if (detectOnlyOnce == False or labels is None) and lastDetect >= detectInterval:
        lastDetect = 0

        # Downscale the image to get faster results from Mask RCNN (the obvious drawback
        # being accuracy)
        downscaledFrame = cv2.resize(frame, None, fx=(1/downscaleFactor), fy=(1/downscaleFactor), interpolation=cv2.INTER_CUBIC)
        imageBlob = cv2.dnn.blobFromImage(downscaledFrame, swapRB=True, crop=False)

        # Propagate and detect objects in the image
        maskRCNNModel.setInput(imageBlob)
        boundingBoxes, masks = maskRCNNModel.forward(["detection_out_final", "detection_masks"])

        boxesList = [boundingBoxes[0, 0, i, 3:7] for i in range(0, boundingBoxes.shape[2])]
        confidencesList = [boundingBoxes[0, 0, i, 2] for i in range(0, boundingBoxes.shape[2])]

        # NMSBoxes takes two special arguments - score_threshold and nms_threshold
        # score_threshold is used to filter out low confidence results (i.e. it's the minimum
        #   confidence necessary to keep a result)
        # nms_threshold is the maximum intersection allowed between two results
        maxValueIDs = cv2.dnn.NMSBoxes(boxesList, confidencesList, 0.5, 0.25)

        labels = np.zeros(frame.shape)

        # https://github.com/ewelborn/PERS-grant-tracking-vehicles/blob/main/main.py
        for i in maxValueIDs:
            predictedClassID = int(boundingBoxes[0, 0, i, 1])
            predictedClassLabel = COCOLabels[predictedClassID] if predictedClassID < len(COCOLabels) else ""
            predictionConfidence = boundingBoxes[0, 0, i, 2]

            if predictedClassLabel == "person":
                x, y, endX, endY = (boundingBoxes[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])).astype("int")

                # Find the pixel mask of the person
                mask = masks[i, predictedClassID]
                mask = cv2.resize(mask, (endX - x, endY - y), interpolation=cv2.INTER_CUBIC)
                mask = (mask > 0.2).astype("uint8") * 255

                # Convert the mask into a 3D array and pad it out
                # (to make this array compatible with the mask array)
                originalShape = mask.shape
                mask = np.repeat(mask, 3).reshape((originalShape[0], originalShape[1], 3))

                labels[y:y + mask.shape[0], x:x + mask.shape[1]] += mask[:min(mask.shape[0], frameHeight - y), :min(mask.shape[1], frameWidth - x)]

        # Dilate the mask to increase the unblurred area
        labels = cv2.dilate(labels.astype("uint8"), element)
    else:
        lastDetect += 1

    # Blur the entire frame
    blurredFrame = copy.deepcopy(frame)
    blurredFrame = cv2.GaussianBlur(blurredFrame, (blurStrength, blurStrength), 0)

    # Replace the foreground of the blurred frame with the original,
    # unblurred image, using the masks from Mask RCNN
    blurredFrame = np.where(labels == 0, blurredFrame, frame)
    
    out.write(blurredFrame)

    print("total",time.time() - start)

    cv2.imshow("Filtered Video", blurredFrame)
    key = cv2.waitKey(int((1/FPS)*1000)) & 0xFF

    if key == ord("q"):
        break

out.release()
