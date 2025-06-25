import pygame, sys
from pygame.locals import *
import numpy as np
from tensorflow.keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDARYINC = 5

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
MODEL = load_model("bestmodel.h5")
Predict = True
LABELS = {0:"Zero", 1:"One", 
          2:"Two", 3:"Three", 
          4:"Four", 5:"Five", 
          6:'Six', 7:"Seven", 
          8:"Eight", 9:"Nine"}
Img_save = False
Img_count = 1

pygame.init()

screen = pygame.display.set_mode((WINDOWSIZEX,  WINDOWSIZEY))
pygame.display.set_caption("Digit Recognition App")

FONT = pygame.font.Font("Poppins-Medium.ttf", 18)
iswriting = False
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(screen, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if not number_xcord or not number_ycord:
                continue

            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(number_ycord[-1] + BOUNDARYINC, WINDOWSIZEY)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(screen))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float64)

            pygame.draw.rect(screen, RED, (rect_min_x, rect_min_y,
                                           rect_max_x - rect_min_x,
                                           rect_max_y - rect_min_y), 2)

            if Img_save:
                cv2.imwrite("image.png")
                img_count += 1

            if Predict:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values = 0)
                image = cv2.resize(image, (28, 28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                text_surface = FONT.render(label, True, RED, WHITE)
                textRecObj = text_surface.get_rect()
                textRecObj.midbottom = (rect_min_x + (rect_max_x - rect_min_x) // 2, rect_min_y - 5)

                screen.blit(text_surface, textRecObj)

        if event.type == KEYDOWN:
            if event.unicode == "n":
                screen.fill(BLACK)

    pygame.display.update()
