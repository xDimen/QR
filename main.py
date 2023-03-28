
import cv2 as cv
import numpy as np
import math
import pyzbar.pyzbar as pyzbar


camID = 0  
KNOWN_DISTANCE = 30  # cm
KNOWN_WIDTH = 4  # cm

fonts = cv.FONT_HERSHEY_COMPLEX
Pos =(50,50)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
GOLD = (0, 255, 215)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 230)

def eucaldainDistance(x, y, x1, y1):

    eucaldainDist = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    return eucaldainDist

def focalLengthFinder(knowDistance, knownWidth, widthInImage):
    focalLength = ((widthInImage * knowDistance) / knownWidth)
    return focalLength

def distanceFinder(focalLength, knownWidth, widthInImage):
    distance = ((knownWidth * focalLength) / widthInImage)
    return distance

def DetectQRcode(image):
    codeWidth = 0
    x, y = 0, 0
    euclaDistance = 0
    global Pos 
    # gray 
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    objectQRcode = pyzbar.decode(Gray)
    for obDecoded in objectQRcode:

        points = obDecoded.polygon
        if len(points) > 4:
            hull = cv.convexHull(
                np.array([points for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        n = len(hull)


        for j in range(0, n):
            print(j, "      ", (j + 1) % n, "    ", n)

            cv.line(image, hull[j], hull[(j + 1) % n], MAGENTA, 2)
        x, x1 = hull[0][0], hull[1][0]
        y, y1 = hull[0][1], hull[1][1]
        Pos = hull[3]
        euclaDistance = eucaldainDistance(x, y, x1, y1)
        return euclaDistance

camera = cv.VideoCapture(camID)

refernce = cv.imread("reference2.png")
Rwidth= DetectQRcode(refernce)

focalLength = focalLengthFinder(KNOWN_DISTANCE, KNOWN_WIDTH, Rwidth)
print("Focal length:  ", focalLengthFinder)

counter =0

while True:
    ret, frame = camera.read()
 
    codeWidth= DetectQRcode(frame)
    
    if codeWidth is not None:
        
        Distance = distanceFinder(focalLength, KNOWN_WIDTH, codeWidth)

        cv.putText(frame, f"Distance: {Distance}", (50,50), fonts, 0.8, (GOLD), 2)
    

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord('s'):
        counter += 1
        print("frame saved")
        cv.imwrite(f"frames/frame{counter}.png", frame)
    if key == ord('q'):
        break
camera.release()
cv.destroyAllWindows()
