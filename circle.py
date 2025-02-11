import cv2

class Circle:
    def __init__(self, x, y, r,firstframe, videofile):
        self.x = x
        self.y = y
        self.r = r
        self.firstFrame = firstframe
        self.videofile = videofile
    def showCenter(self):
        frame = self.videofile.getFrame(self.firstFrame)
        cv2.circle(frame, (self.x, self.y), 2, (255, 0, 0), 3)
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("frame",frame)
        cv2.waitKey(0)
    