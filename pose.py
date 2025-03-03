from mmpose.apis import MMPoseInferencer
import cv2
img_path = 'videos/ae41spainpole.mp4'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
# cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(img_path)
while cap.isOpened():
    ret, frame = cap.read()
    resizedFrame = cv2.resize(frame, (640, 480))
    if not ret:
        break
    result_generator = inferencer(resizedFrame, show=True)
    result = next(result_generator)