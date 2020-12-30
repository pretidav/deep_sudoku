import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--filter", required=True, help="path to trained digit classifier",action='store_true')
ap.add_argument("--image", required=True, help="path to input Sudoku puzzle image")
args = vars(ap.parse_args())
filter = args['filter']
file_path = args['image']

img = cv2.imread(file_path)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 3)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.bitwise_not(thresh)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# initialize a contour that corresponds to the puzzle outline
puzzleCnt = None
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then we can
    # assume we have found the outline of the puzzle
    if len(approx) == 4:
        puzzleCnt = approx
        break
output = img.copy()
cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)

# apply a four point perspective transform to both the original
# image and grayscale image to obtain a top-down bird's eye view
# of the puzzle
puzzle = four_point_transform(img, puzzleCnt.reshape(4, 2))
warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
cv2.imwrite('perspective.jpg',puzzle)

filtered_lines = []
lower_lim = 90
while len(filtered_lines)<20:

    img = cv2.imread('perspective.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,lower_lim,100,apertureSize = 3)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.erode(edges,kernel,iterations = 1)
    cv2.imwrite('canny.jpg',edges)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)

    if not lines.any():
        print('No lines were found')
        exit()

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i : [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]: # and only if we have not disregarded them already
                    continue

                rho_i,theta_i = lines[indices[i]][0]
                rho_j,theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

    #print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)): # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines
    lower_lim-=10

for line in filtered_lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('grid.jpg',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
edged = cv2.Canny(gray, 30, 200) 

contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
print("Number of Contours found = " + str(len(contours))) 

count = 0
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    count+=1
    cv2.drawContours(img, cnt, -1, (0, 255, 0), 3) 
    #cv2.imwrite('crop'+str(count)+'.jpg',img[y:y+h, x:x+w])

print('countours {}'.format(count))
cv2.imwrite('contours.jpg',img)
