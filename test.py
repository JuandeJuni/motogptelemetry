import cv2
from videoproccesing import VideoFile
from telemetrydata import TelemetryData
import plotting
# input = VideoFile("videos/jm89malasyapole.mp4")
input = VideoFile("videos/ae41spainpole.mp4")
# input = VideoFile("videos/mm93portimaopole.mp4")
# frame = input.showFrame(30)


c = input.detectCircle()
t = TelemetryData(c)
t.readSpeed2()
# brakes = plotting.removeOutliers(t.brakes, 0.9)
# brakes = plotting.smoothData(brakes, 5)
# throttle = plotting.removeOutliers(t.throttle, 0.9)
# throttle = plotting.smoothData(throttle, 5)
# plotting.plotUnitData(brakes, "Brakes Smooth", "Frame", "Brakes", "blue")
# plotting.plotUnitData(t.brakes, "Throttle", "Frame", "Throttle", "red")
# plotting.plotUnitData(t.throttle, "Throttle", "Frame", "Speed", "green")
# plotting.plotUnitData(throttle, "Throttle Smooth", "Frame", "Throttle", "blue")

# print(t.throttle)
