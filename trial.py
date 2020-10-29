import cv2
import PIL
import helping_functions as hf

v=cv2.VideoCapture(0)
_, frame=v.read()
flag=0
while(True):
    ret, frame=v.read()
    frame=hf.new_rectifier(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    img=PIL.Image.fromarray(frame)
    ke=cv2.waitKey(1)
    if ke==ord('q'):
        break
    if ke==ord('s'):
        cv2.destroyAllWindows()
        flag=1
    if ke==ord('d'):
        cv2.destroyAllWindows()
        flag=0
    if flag==1:
        op=hf.output_multiple(img)
        frame=cv2.putText(frame,str(op),(0,105), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(150,150,150),2,cv2.LINE_AA)
        cv2.imshow('Testing: S:Start Detection | D:Stop Detection| Q:Quit', frame)
    if flag==0:
        cv2.imshow('Testing: Press Q to quit', frame)


v.release()
cv2.destroyAllWindows()

