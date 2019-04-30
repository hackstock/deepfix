import cv2
import os
import uuid

class Recognizer(object):

    def __init__(self, src, haar_path, pos_prefix, neg_prefix, size):
        self.cap = cv2.VideoCapture(src)
        self.img_count = 0
        self.img_class = 1
        self.pos_count = 0
        self.neg_count = 0
        self.pos_prefix = pos_prefix
        self.neg_prefix = neg_prefix
        self.size = int(size)
        self.classifier = cv2.CascadeClassifier(haar_path)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def __save(self, img):
        if self.img_class == 0:
            self.neg_count += 1
            filename = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../images/{}.{}.png".format(self.neg_prefix,uuid.uuid1())))
        else:
            self.pos_count += 1
            filename = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../images/{}.{}.png".format(self.pos_prefix,uuid.uuid1())))

        print("saving file to: {}".format(filename))
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(filename, img)
        self.img_count += 1
    
    def run(self):
        while(True):
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.putText(frame, "Current Image Class : {} : Press '0' or '1' to change".format(str(self.img_class)), (10,30), self.font, 0.8, (0,0,255),3)
            cv2.putText(frame, "Instructions : Press 's' to save or 'q' to quit", (10,55), self.font, 0.8, (200,0,120),3)
            cv2.putText(frame, "Statistics : Pos = {} , Neg = {}".format(self.pos_count, self.neg_count), (10,80), self.font, 0.8, (150,45,120),3)

            faces = self.classifier.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                roi = gray[y:y+h,x:x+w]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("0"):
                self.img_class = 0
            elif key == ord("1"):
                self.img_class = 1
            elif key == ord("s"):
                self.__save(roi)
            elif key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
