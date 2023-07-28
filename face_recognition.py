
import cv2, time, os, sys

from utils import ( 
    NumpyArrayEncoder,
    FpeNet,
    get_distance,
    read_json,
    write_json
)

def compare_db(db, cur_lm):
    id = None
    score = 0
    map = {}
    for trg_id, trg_lm in db.items():
        tmp_score = 0
        for lm_idx, lm in enumerate(trg_lm):
            tmp_score += get_distance(lm, cur_lm[lm_idx])
        map[trg_id]=tmp_score
        if tmp_score <=500 and (tmp_score < score or score==0):
            score, id = tmp_score, trg_id
    print(map)
    return id

def face_lm_detector( facenet_path, fpenet_path, source):

    # Read DB
    db_path = './db.json'
    if(os.path.exists(db_path)):
        db = read_json(db_path)
    else:
        db = {}

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
            
        draw = frame.copy()
        
        # Capture Face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = detector.detectMultiScale(gray)        

        # Primary Model to detect face
        for (x,y,w,h) in faces:

            x1, y1, x2, y2 = x, y, x+w, y+h
            draw   = cv2.rectangle(draw, (x1, y1), (x2, y2), (0,255,0), 2) 
            face    = frame[y1:y2, x1:x2]

            # Secondary Model to predict face landmark
            landmarks = landmark_engine.predict(face)
            employee_id = compare_db(db, landmarks)
            cv2.putText(draw, f'ID: {employee_id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 1, cv2.LINE_AA)
            
        t2 = time.time()

        # Calculate FPS
        fps_list.append(1/(t2-t1))
        fps = round(sum(fps_list)/len(fps_list), 3)
        cv2.putText(draw, f'FPS: {fps}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display
        cv2.imshow(CV_WIN, draw)
        if cv2.waitKey(0) in [ ord('q'), ord('Q'), 27]:
            break

    cv2.destroyAllWindows()
    cap.release()

def get_arguments():

    import argparse
    arg_parser = argparse.ArgumentParser()
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
        source='/dev/video0'
    )