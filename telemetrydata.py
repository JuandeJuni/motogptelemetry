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
        # Lucas-Kanade Optical Flow parameters
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Shi-Tomasi Feature Detection parameters
        feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Define fixed A & B positions
        position_A = (self.circle.x+config.windowSpeedUp[0],self.circle.y+config.windowSpeedUp[1],config.windowSpeedUp[2],config.windowSpeedUp[3])  # Adjust based on actual video
        position_B = (self.circle.x+config.windowSpeedDown[0],self.circle.y+config.windowSpeedDown[1],config.windowSpeedDown[2],config.windowSpeedDown[3])  # Adjust based on actual video

        # Tracking status
        current_position = "B"  # Start at A
        tracking_moving = True  # Optical Flow is active only when digits are moving
        transition_log = []

        # Read first frame
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read video")
            cap.release()
            exit()

        # Convert to grayscale
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Start with the fixed ROI at A
        x, y, w, h = position_B
        roi_gray = first_gray[y:y+h, x:x+w]
        p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

        # Adjust keypoints to absolute coordinates
        if p0 is not None:
            for i in range(p0.shape[0]):
                p0[i][0][0] += x
                p0[i][0][1] += y

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if tracking_moving:
                # Compute optical flow when digits are moving
                if p0 is not None and len(p0) > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(first_gray, gray, p0, None, **lk_params)
                    good_new = p1[st == 1]  # Keep only valid keypoints

                    if len(good_new) > 0:
                        centroid_x = int(np.mean(good_new[:, 0]))
                        centroid_y = int(np.mean(good_new[:, 1]))

                        # If movement reaches B, switch to B
                        x_B, y_B, w_B, h_B = position_B
                        if x_B <= centroid_x <= x_B + w_B and y_B <= centroid_y <= y_B + h_B:
                            if current_position == "A":
                                print(f"Transition detected: A → B at frame {frame_count}")
                                transition_log.append({"Frame": frame_count, "Transition": "A → B"})
                            current_position = "B"
                            tracking_moving = False  # Stop Optical Flow, switch to fixed ROI

                        # If movement reaches A, switch to A
                        x_A, y_A, w_A, h_A = position_A
                        if x_A <= centroid_x <= x_A + w_A and y_A <= centroid_y <= y_A + h_A:
                            if current_position == "B":
                                print(f"Transition detected: B → A at frame {frame_count}")
                                transition_log.append({"Frame": frame_count, "Transition": "B → A"})
                            current_position = "A"
                            tracking_moving = False  # Stop Optical Flow, switch to fixed ROI

                        # Update tracking ROI while moving
                        x, y = centroid_x - w // 2, centroid_y - h // 2
                        x = max(0, min(x, frame.shape[1] - w))
                        y = max(0, min(y, frame.shape[0] - h))

                    # Update previous frame and keypoints
                    first_gray = gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)

                else:
                    # If keypoints are lost, reset to fixed A or B position
                    tracking_moving = False

            else:
                # When digits are static, use fixed ROI at A or B
                if current_position == "A":
                    x, y, w, h = position_A
                else:
                    x, y, w, h = position_B

                # Detect motion: Try finding new keypoints
                roi_gray = gray[y:y+h, x:x+w]
                p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

                # Convert keypoints to absolute coordinates
                if p0 is not None:
                    for i in range(p0.shape[0]):
                        p0[i][0][0] += x
                        p0[i][0][1] += y

                # If enough keypoints are detected, start tracking movement
                if p0 is not None and len(p0) > 10:
                    tracking_moving = True

            # Draw the ROI
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Tracking: {current_position}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show result
            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

            cv2.imshow("Tracking", frame)

            # Quit with 'Q'
            if cv2.waitKey(25) & 0xFF == ord('q'):
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
        position_A_speed = (self.circle.x+config.windowSpeedUp[0],self.circle.y+config.windowSpeedUp[1],config.windowSpeedUp[2],config.windowSpeedUp[3])
        position_A_kmh = (self.circle.x+config.windowKMHUp[0],self.circle.y+config.windowKMHUp[1],config.windowKMHUp[2],config.windowKMHUp[3])
        position_B_speed = (self.circle.x+config.windowSpeedDown[0],self.circle.y+config.windowSpeedDown[1],config.windowSpeedDown[2],config.windowSpeedDown[3])
        position_B_kmh = (self.circle.x+config.windowKMHDown[0],self.circle.y+config.windowKMHDown[1],config.windowKMHDown[2],config.windowKMHDown[3])

        frame_count = 0
        speed_tracking = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Extract speed only if "km/h" is detected
            speed_A = self.extract_speed_if_kmh_easyocr(frame, position_A_speed, position_A_kmh,reader)
            # speed_A = None
            speed_B = self.extract_speed_if_kmh_easyocr(frame, position_B_speed, position_B_kmh,reader)
            # Determine which position has the speed
            if speed_A:
                position = "A"
                speed = speed_A
            elif speed_B:
                position = "B"
                speed = speed_B
            else:
                position = "None"
                speed = None

            # Store tracking data
            speed_tracking.append({"Frame": frame_count, "Speed": speed, "Position": position})

            # Draw bounding boxes
            cv2.rectangle(frame, position_A_speed[:2], 
                        (position_A_speed[0]+position_A_speed[2], position_A_speed[1]+position_A_speed[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, position_A_kmh[:2], 
                        (position_A_kmh[0]+position_A_kmh[2], position_A_kmh[1]+position_A_kmh[3]), (0, 255, 0), 2)

            cv2.rectangle(frame, position_B_speed[:2], 
                        (position_B_speed[0]+position_B_speed[2], position_B_speed[1]+position_B_speed[3]), (0, 0, 255), 2)
            cv2.rectangle(frame, position_B_kmh[:2], 
                        (position_B_kmh[0]+position_B_kmh[2], position_B_kmh[1]+position_B_kmh[3]), (0, 0, 255), 2)

            # Display results
            cv2.putText(frame, f"A Speed: {speed_A}", (position_A_speed[0], position_A_speed[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"B Speed: {speed_B}", (position_B_speed[0], position_B_speed[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
            cv2.imshow("Tracking", frame)

            # Quit with 'Q'
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    def preprocess_roi(self,frame, position):
        x, y, w, h = position
        roi = frame[y:y+h, x:x+w]
        frameGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # frameBlur = cv2.GaussianBlur(frameGray, (3, 3), 0)
        # _, thresh = cv2.threshold(frameGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return frameGray
    def recognize_digit_tesseract(self,thresh):
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        digit = pytesseract.image_to_string(thresh, config=config)
        
        return digit.strip() if digit.strip().isdigit() else None
    def recognize_digit_easyocr(self,thresh,reader):
        result = reader.readtext(thresh)
        return result[0][1] if result and result[0][1].isdigit() else None
    def detect_kmh_easyocr(self,frame, position,reader):
        """Detects if 'km/h' text exists in the given region."""
        thresh = self.preprocess_roi(frame, position)
        cv2.imshow("gray",thresh)
        # Run OCR
        results = reader.readtext(thresh)
        print(results)
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
    def recognize_speed_easyocr(self,frame, position,reader):
        """Recognizes speed values using EasyOCR with digit filtering."""
        thresh = self.preprocess_roi(frame, position)

        # Run OCR
        results = reader.readtext(thresh)

        # Filter for valid speeds (2-3 digit numbers between 10-300)
        for res in results:
            text = res[1].strip()
            if text.isdigit():
                num = int(text)
                return num  # Return as integer
        return None



    



                


                