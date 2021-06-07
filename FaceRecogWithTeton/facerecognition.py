#first import necessary libraries
import cv2
import numpy as np
import face_recognition as fr

#load original images using face_recognition
imgPolok=fr.load_image_file('ImageSimpleTest/polokbhai1.jpg')
imgPolok=cv2.cvtColor(imgPolok,cv2.COLOR_BGR2RGB)
face_loc=fr.face_locations(imgPolok)[0]
face_encod=fr.face_encodings(imgPolok)[0]

cv2.rectangle(imgPolok,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)

#load test images using face_recognition
imgPolok_Test=fr.load_image_file('ImageSimpleTest/razib_bhaia.jpg')
imgPolok_Test=cv2.cvtColor(imgPolok_Test,cv2.COLOR_BGR2RGB)
face_loc_test=fr.face_locations(imgPolok_Test)[0]
face_encod_test=fr.face_encodings(imgPolok_Test)[0]

cv2.rectangle(imgPolok_Test,(face_loc_test[3],face_loc_test[0]),(face_loc_test[1],face_loc_test[2]),(255,255,0),2)

result=fr.compare_faces([face_encod],face_encod_test)
dis_result=fr.face_distance([face_encod],face_encod_test)
print(result,dis_result)
cv2.putText(imgPolok_Test,f'{result} {round(dis_result[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,45,255),2)
cv2.imshow("Polok",imgPolok)
cv2.imshow("PolokTest",imgPolok_Test)




cv2.waitKey(0)