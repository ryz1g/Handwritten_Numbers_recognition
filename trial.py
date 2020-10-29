import cv2
import PIL
import helping_functions as hf
import numpy as np

v=cv2.VideoCapture(0)
_, frame=v.read()
while(True):
    ret, frame=v.read()
    #c1,c2=frame_prev.shape
    frame=hf.new_rectifier(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    #frame=cv2.line(frame,(0,84), (399,84), (255,0,0), 1)
    #frame=cv2.line(frame,(0,140), (399,140), (255,0,0), 1)
    #img=PIL.Image.fromarray(frame).crop((0, 85, 399, 140))
    img=PIL.Image.fromarray(frame)
    op=hf.output_multiple(img)
    frame=cv2.putText(frame,str(op),(0,105), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(150,150,150),2,cv2.LINE_AA)
    cv2.imshow('Testing: Press Q to quit', frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        #img.show()
        #op=hf.output_multiple(img)
        break

v.release()
cv2.destroyAllWindows()
