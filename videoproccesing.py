import cv2
from circle import Circle
class VideoFile:
    def __init__(self, path):
        video = cv2.VideoCapture(path)
        self.path = path
        self.fps = int(video.get(cv2.CAP_PROP_FPS))
        self.duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) / self.fps
        self.file_name = path.split('/')[-1]
        video.release()
    def detectCircle(self):
        cap = cv2.VideoCapture(self.path)
        frame_count = -1
        circle_detected = False
        c = None
        while not circle_detected:
            ret, frame = cap.read()
            if not ret:
                break  # Exit when the video ends

            frame_count += 1

            # Convert to grayscale for circle detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)

            # Detect circles using Hough Circle Transform
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=40,
                param1=100,
                param2=50,
                minRadius=140,
                maxRadius=160
            )

            if circles is not None:
                # Assume the first circle detected is the desired one
                circles = circles[0, :].astype("int")
                x, y, r = circles[0]
                c = Circle(x,y,r,frame_count,self)
                # Display the frame with the detected circle
                # cv2.imshow("Detected Circle", frame)

                # # Wait for user to close the window or press any key
                # print(f"Circle detected at frame: {frame_count}")
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                circle_detected = True
        return c
    def showFrame(self,frame_number):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video.read()
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
    def getFrame(self,frame_number):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video.read()
        return frame
        




