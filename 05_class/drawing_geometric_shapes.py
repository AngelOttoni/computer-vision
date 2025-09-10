import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)

# Draw green rectangle at the top-right corner of image 
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

# Draw a circle inside the rectangle drawn above
cv.circle(img,(447,63), 63, (0,0,255), -1)

# Draw a half ellipse at the center of the image
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

# Draw a small polygon of with four vertices in yellow color
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))

# Write 'OpenCV' on our image in white color
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

# Converter BGR para RGB
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Mostrar a imagem
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title('Linha Diagonal Azul')
plt.axis('off')
plt.tight_layout()
plt.show()