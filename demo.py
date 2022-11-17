import cv2, time
from utils import ( 
    FpeNet
)

def face_detector(model, source):
    # Face Model
    detector = cv2.CascadeClassifier(model)

    cap = cv2.VideoCapture(source)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
        face = detector.detectMultiScale(gray)        # 擷取人臉區域
        for(x,y,w,h) in face:
            x1, y1, x2, y2 = x, y, x+w, y+h
            frame   = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2) 
            cv2.imshow(CV_WIN, frame)

        if cv2.waitKey(1) in [ord('q'), ord('Q'), 27]: break

    cv2.destroyAllWindows()
    cap.release()

def lm_detector(model, source):

    cap     = cv2.VideoCapture(source)
    engine  = FpeNet(model)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        landmark = engine.predict( frame )
        cv2.imshow(CV_WIN, engine.visualize( frame, landmark ) )
        if cv2.waitKey(1) in [ord('q'), ord('Q'), 27]: break

    cv2.destroyAllWindows()
    cap.release()

def face_lm_detector( facenet_path, fpenet_path, source):

    # Define Landmark Model
    landmark_engine = FpeNet(fpenet_path)

    # Load Face Detector
    detector = cv2.CascadeClassifier(facenet_path)
    
    # Capture and Wait for first frame
    cap = cv2.VideoCapture(source)
    
    if cap.isOpened():
        print('Opened')
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    else:
        print('Not Opened')

    fps_list = []

    while(cap.isOpened()):

        t1 = time.time()
        
        ret, frame = cap.read()
        if(not ret): 
            break
        
        # Capture Face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = detector.detectMultiScale(gray)        

        # Primary Model to detect face
        for (x,y,w,h) in faces:

            x1, y1, x2, y2 = x, y, x+w, y+h
            frame   = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2) 
            face    = frame[y1:y2, x1:x2]

            # Secondary Model to predict face landmark
            landmarks = landmark_engine.predict(face)
            mini_h, mini_w = face.shape[:2]
            for lx, ly in landmarks:
                lx = int(lx * mini_w / landmark_engine.input_size[0])
                ly = int(ly * mini_h / landmark_engine.input_size[1])
                cv2.circle(frame, (x+lx, y+ly), 1, (0,255,0), 1)

        t2 = time.time()

        # Calculate FPS
        fps_list.append(1/(t2-t1))
        fps = round(sum(fps_list)/len(fps_list), 3)
        cv2.putText(frame, f'FPS: {fps}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display
        cv2.imshow(CV_WIN, frame)
        if cv2.waitKey(1) in [ ord('q'), ord('Q'), 27]:
            break

    cv2.destroyAllWindows()
    cap.release()

def get_arguments():

    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(    '-m', '--mode', type=str, 
                                choices=[FACE, LM, ALL], 
                                default='both', required=True )
    arg_parser.add_argument(    '-f', '--face', type=str, 
                                default='./model/haarcascade_frontalface_default.xml')
    arg_parser.add_argument(    '-l', '--lm', type=str, 
                                default='./model/fpenet_b1_fp32.trt')
    arg_parser.add_argument(    '-s', '--source', type=str, 
                                default='/dev/video0')
    return arg_parser.parse_args()    

if __name__ == '__main__':

    FACE    = 'face'
    LM      = 'landmark'
    ALL     = 'all'

    CV_WIN  = 'Face + Landmark'

    args = get_arguments()
    if args.mode == FACE:
        face_detector(model=args.face, source=args.source)

    elif args.mode == LM:
        lm_detector(model=args.lm, source=args.source)

    else:
        face_lm_detector(
            facenet_path=args.face,
            fpenet_path=args.lm,
            source=args.source
        )