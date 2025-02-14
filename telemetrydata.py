import cv2
import numpy as np
import time
import pytesseract
import config
import easyocr

class TelemetryData:
    def __init__(self,c):
        self.brakes = []
        self.throttle = []
        self.speed = []
        self.angulation = []
        self.gears = []
        self.windowLenght = 0
        self.circle = c
    def computeWindows(self):
        cap = cv2.VideoCapture(self.circle.videofile.path)
        first_frame = self.circle.firstFrame
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        minX = []
        maxX = []
        counter = 0
        while counter < (60*self.circle.videofile.fps):
        # while True:
            ret, frame = cap.read()
            if not ret:
                break

            cFrame = frame[self.circle.y-config.windowSize[1]:self.circle.y+config.windowSize[1],self.circle.x-config.windowSize[0]:self.circle.x+config.windowSize[0]]

            hsvFrame = cv2.cvtColor(cFrame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsvFrame,np.array([30,100,80]),np.array([50,255,255]))

            # mask = cv2.Canny(hsvFrame, 100, 200)
            # result = cv2.bitwise_and(cFrame,cFrame,mask=mask)
            try:
                # print(np.min(np.where(mask==255)[1]))
                
                # print(np.max(np.where(mask==255)[1]))
                minX.append(np.min(np.where(mask==255)[1]))
                maxX.append(np.max(np.where(mask==255)[1]))
                counter +=1
            except:
                pass
        cap.release()
        #get the mode of an array of integers
        modeMinX = max(set(minX), key = minX.count)
        modeMaxX = max(set(maxX), key = maxX.count)
        self.windowLenght = modeMaxX - modeMinX
        return modeMinX, modeMaxX-modeMinX

    def computeThrottle(self):
        cap = cv2.VideoCapture(self.circle.videofile.path)
        first_frame = self.circle.firstFrame
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        minX , windowlenght = self.computeWindows()
        # print(minX, windowlenght)
        while cap.isOpened():
        # while True:
            ret, frame = cap.read()
            if not ret:
                break

            cFrame = frame[self.circle.y-config.windowSize[1]:self.circle.y+config.windowSize[1],self.circle.x-config.windowSize[0]:self.circle.x+config.windowSize[0]]
            hsvFrame = cv2.cvtColor(cFrame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsvFrame,np.array([30,100,80]),np.array([50,255,255]))


            # mask = cv2.Canny(hsvFrame, 100, 200)
            # result = cv2.bitwise_and(cFrame,cFrame,mask=mask)
          
            try:
                X = np.max(np.where(mask==255)[1])
                Throttle = X - minX
                Throttle = Throttle/windowlenght
                if Throttle < 0 or Throttle > 1:
                    Throttle = self.throttle[-1]
                    X = self.throttle[-1]*(windowlenght) + minX
                    dab = False
            except:
                X = 0
                Throttle = 0
            self.throttle.append(Throttle)
            if config.DEBUG:
                cv2.circle(cFrame,(int(X),config.windowSize[1]),1,(0, 0, 255),-1)
                cv2.imshow("mask",mask)
                # cv2.imshow("result",result)
                cv2.imshow("cFrame",cFrame)
                cv2.waitKey(1)
                # time.sleep(0.02)
        cap.release()
    def computeBrakes(self):
        cap = cv2.VideoCapture(self.circle.videofile.path)
        first_frame = self.circle.firstFrame
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        minX , windowlenght = self.computeWindows()

        while True:
        # while True:
            ret, frame = cap.read()
            if not ret:
                break

            cFrame = frame[self.circle.y-config.windowSize[1]:self.circle.y+config.windowSize[1],self.circle.x-config.windowSize[0]:self.circle.x+config.windowSize[0]]
            hsvFrame = cv2.cvtColor(cFrame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsvFrame,np.array([175,200,150]),np.array([180,255,255]))
            # kernel = np.ones((3, 3), np.uint8)
            kernel = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], np.uint8)
            # kernel = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], np.uint8)  # Left-edge preserving

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            dab = False
            try:
                X = np.min(np.where(mask==255)[1])
                breaks = minX - X
                breaks = breaks/windowlenght
                if breaks < 0 or breaks > 1:
                    breaks = self.brakes[-1]
                    X = minX - (self.brakes[-1]*(windowlenght))
                    dab = False
            except:
                X = minX
                breaks = 0
            self.brakes.append(breaks)
            if config.DEBUG:
                cv2.circle(cFrame,(int(X),config.windowSize[1]),1,(255, 0, 0),-1)
                
                # cv2.imshow("result",result)
                cv2.imshow("mask",mask)
                cv2.imshow("cFrame",cFrame)
                cv2.waitKey(1)
                if dab:
                    time.sleep(2)
                # time.sleep(0.02)
        cap.release()
    def computeBrakeThrottle(self):
        cap = cv2.VideoCapture(self.circle.videofile.path)
        first_frame = self.circle.firstFrame
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        minX , windowlenght = self.computeWindows()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cFrame = frame[self.circle.y-config.windowSize[1]:self.circle.y+config.windowSize[1],self.circle.x-config.windowSize[0]:self.circle.x+config.windowSize[0]]
            hsvFrame = cv2.cvtColor(cFrame, cv2.COLOR_BGR2HSV)
            maskBrakes = cv2.inRange(hsvFrame,np.array([175,200,150]),np.array([180,255,255]))
            maskThrottle = cv2.inRange(hsvFrame,np.array([30,100,80]),np.array([50,255,255]))
            kernelBrakes = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]], np.uint8)
            maskBrakes = cv2.morphologyEx(maskBrakes, cv2.MORPH_OPEN, kernelBrakes)
            try:
                XThrottle = np.max(np.where(maskThrottle==255)[1])
                Throttle = XThrottle - minX
                Throttle = Throttle/windowlenght
                if Throttle < 0 or Throttle > 1:
                    Throttle = self.throttle[-1]
                    XThrottle = self.throttle[-1]*(windowlenght) + minX
            except:
                XThrottle = 0
                Throttle = 0
            try:
                XBrakes = np.min(np.where(maskBrakes==255)[1])
                Breaks = minX - XBrakes
                Breaks = Breaks/windowlenght
                if Breaks < 0 or Breaks > 1:
                    Breaks = self.brakes[-1]
                    XBrakes = minX - (self.brakes[-1]*(windowlenght))
            except:
                XBrakes = minX
                Breaks = 0
            self.throttle.append(Throttle)
            self.brakes.append(Breaks)
            if config.DEBUG:
                cv2.circle(cFrame,(int(XThrottle),config.windowSize[1]),1,(0, 0, 255),-1)
                cv2.circle(cFrame,(int(XBrakes),config.windowSize[1]),1,(255, 0, 0),-1)
                cv2.imshow("maskBrakes",maskBrakes)
                cv2.imshow("maskThrottle",maskThrottle)
                cv2.imshow("cFrame",cFrame)
                cv2.waitKey(1)
        cap.release()
    def readSpeed(self):
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        cap = cv2.VideoCapture(self.circle.videofile.path)
        first_frame = self.circle.firstFrame
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        # Initial bounding box
        y = self.circle.y - 60
        x = self.circle.x - 40
        h = 35
        w = 75

        ret, initFrame = cap.read()
        if not ret:
            print("Error: Cannot read video frame")
            exit()

        # Convert to grayscale for Optical Flow
        initGray = cv2.cvtColor(initFrame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert current frame to grayscale
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameBlur = cv2.GaussianBlur(frameGray, (5, 5), 0)
            _, frameGray = cv2.threshold(frameBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Compute dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(initGray, frameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Extract movement inside bounding box
            flowX = flow[y:y+h, x:x+w, 0]
            flowY = flow[y:y+h, x:x+w, 1]

            # Compute mean displacement of pixels
            dx = np.mean(flowX)
            dy = np.mean(flowY)

            # Update bounding box position
            x += int(dx)
            y += int(dy)

            # Ensure bounding box stays within frame limits
            x = max(0, min(frame.shape[1] - w, x))
            y = max(0, min(frame.shape[0] - h, y))


            # **Update the reference frame for optical flow**
            initGray = frameGray.copy()

            # Draw the updated bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show result
            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def readSpeed2(self):
        # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        reader = easyocr.Reader(['en'])
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        cap = cv2.VideoCapture(self.circle.videofile.path)
        first_frame = self.circle.firstFrame
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    
        ret, frame = cap.read()
        if not ret:
            print("Failed to read video")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        positionAspeed = (self.circle.x+config.windowSpeedUp[0],self.circle.y+config.windowSpeedUp[1],config.windowSpeedUp[2],config.windowSpeedUp[3])
        positionBspeed = (self.circle.x+config.windowSpeedDown[0],self.circle.y+config.windowSpeedDown[1],config.windowSpeedDown[2],config.windowSpeedDown[3])
        positions = [positionAspeed,positionBspeed]

        pos = self.checkSpeedPosition(frame,reader)
        roi = positions[pos]
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, roi)
        print(self.circle.y)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update tracker
            success, roi = tracker.update(frame)
            
            if success:
                # Draw bounding box
                x, y, w, h = map(int, roi)
                print(y)
                x,_,w,h = positions[pos]
                if y > self.circle.y:
                    pos = 1
                    x,y,w,h = positions[pos]
                elif y < self.circle.y-40:
                    pos = 0
                    x,y,w,h = positions[pos]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Extract digits
            roi = frame[y:y+h,x:x+w]
            t = self.preprocess_roi(roi)
            cv2.imshow("thresh",t)
            speed = self.recognize_digit_easyocr(t,reader)
            # Display speed number
            cv2.putText(frame, speed, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Display frame
            cv2.imshow("Tracking", frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    def preprocess_roi(self,roi):
        """Preprocess the extracted region for better OCR performance"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Binarization
        mask = np.zeros_like(thresh)
        border_size = 2  # Adjust based on your frame
        mask[border_size:-border_size, border_size:-border_size] = 255  # Keep the inner region

        # Apply the mask to the thresholded image
        thresh_masked = cv2.bitwise_and(thresh, mask)
        return thresh_masked

    def recognize_digit_tesseract(self,thresh):
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        digit = pytesseract.image_to_string(thresh, config=config)
        
        return digit.strip() if digit.strip().isdigit() else None
    def recognize_digit_easyocr(self,thresh,reader):
        result = reader.readtext(thresh)
        for res in result: 
            if res[1].isdigit():
                return res[1]
    def detectWindow(self,firstframe,reader):

        results = reader.readtext(firstframe)
        # Check if "km/h" is detected
        for res in results: 
            if "k" in res[1].lower():
                print(res[1])
                return True
        return False
    def extract_speed_if_kmh_easyocr(self,frame, speed_position, kmh_position,reader):
        """Extract speed only if 'km/h' is detected below it."""
        if self.detect_kmh_easyocr(frame, kmh_position,reader):  # Ensure 'km/h' is present
            return self.recognize_speed_easyocr(frame, speed_position,reader)
        return None
    def checkSpeedPosition(self,frame,reader):
        """Recognizes speed values using EasyOCR with digit filtering."""
        
        position_A_kmh = (self.circle.x+config.windowKMHUp[0],self.circle.y+config.windowKMHUp[1],config.windowKMHUp[2],config.windowKMHUp[3])
        position_B_kmh = (self.circle.x+config.windowKMHDown[0],self.circle.y+config.windowKMHDown[1],config.windowKMHDown[2],config.windowKMHDown[3])
        roiA = frame[position_A_kmh[1]:position_A_kmh[1]+position_A_kmh[3],position_A_kmh[0]:position_A_kmh[0]+position_A_kmh[2]]
        roiB = frame[position_B_kmh[1]:position_B_kmh[1]+position_B_kmh[3],position_B_kmh[0]:position_B_kmh[0]+position_B_kmh[2]]
        resultsA = reader.readtext(roiA)
        resultsB = reader.readtext(roiB)
        if resultsA == []:
            return 1
        for res in resultsA: 
            if "k" in res[1].lower():
                return 0
            elif "a" or "n" in res[1].lower():
                return 1
    # def tracker(self,)
        



    



                


                