import cv2, time, os ,sys
from utils import ( 
    NumpyArrayEncoder,
    FpeNet,
    get_distance,
    read_json,
    write_json
)

def face_lm_detector( facenet_path, fpenet_path, source, employee_id=None):

    # Define Saving Folder
    save_folder = './face-data'
    if(not os.path.exists(save_folder)):
        os.mkdir(save_folder)

    # Read DB
    db_path = './db.json'
    if(os.path.exists(db_path)):
        db = read_json(db_path)
    else:
        db = {}

    if employee_id==None:
        employee_id = 0 if db=={} else max(db.keys())+1
            
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

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if(not ret): 
            break
        
        draw = frame.copy()

        # Capture Face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = detector.detectMultiScale(gray)        

        # Primary Model to detect face
        for (x,y,w,h) in faces:

            x1, y1, x2, y2 = x, y, x+w, y+h
            draw = cv2.rectangle(draw, (x1, y1), (x2, y2), (0,255,0), 2) 
            face = frame[y1:y2, x1:x2]

            # Save Image
            save_path = os.path.join(save_folder, f'{employee_id}.jpg' )
            cv2.imwrite(save_path, face)

            # Secondary Model to predict face landmark
            landmarks = landmark_engine.predict(face)
            mini_h, mini_w = face.shape[:2]
            for lx, ly in landmarks:
                lx = int(lx * mini_w / landmark_engine.input_size[0])
                ly = int(ly * mini_h / landmark_engine.input_size[1])
                draw = cv2.circle(draw, (x+lx, y+ly), 1, (0,255,0), 1)

            # Saving into DB
            db[employee_id] = landmarks
            write_json(db_path, db)

        draw = cv2.putText(draw, f'Employee ID: {employee_id}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display
        cv2.imshow(CV_WIN, draw)
        
        if cv2.waitKey(0) in [ ord('q'), ord('Q'), 27]:
            break
        else:
            continue

    cv2.destroyAllWindows()
    cap.release()

def get_arguments():

    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(    '-i', '--id', type=int, required=True )
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
    face_lm_detector(
        facenet_path=args.face,
        fpenet_path=args.lm,
        source=args.source,
        employee_id=args.id
    )