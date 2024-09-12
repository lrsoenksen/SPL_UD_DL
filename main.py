import cv2


image = cv2.imread("data/examples/single_mole_images/2_P001_00007b.png")

cv2.imshow("Image", image)

cv2.waitKey(0)

cv2.destroyAllWindows()