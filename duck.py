import cv2

img = cv2.imread('screen.png')
bnw = cv2.inRange(img, (0, 150, 170), (0, 220, 255))
contours, ret = cv2.findContours(bnw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
if contours:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[0]
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    (x, y, w, h) = cv2.boundingRect(contours)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    print(w, h)

    if h > 240:
        print("stop")

cv2.imshow('inRange', img)

if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()