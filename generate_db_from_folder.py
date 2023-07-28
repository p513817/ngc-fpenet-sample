
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
    tmp_score = 0
    for trg_id, trg_lm in db.items():
        for lm_idx, lm in enumerate(trg_lm):
            tmp_score += get_distance(lm, cur_lm[lm_idx])
        
        if tmp_score < score:
            score = tmp_score
            id = trg_id
    
    return id

def face_lm_detector( facenet_path, fpenet_path, data_path, mode):

    #
    if mode=='auto':
        t_key_wait = 1
    else:
        t_key_wait = 0

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

    # Define Landmark Model
    landmark_engine = FpeNet(fpenet_path)

    # Load Face Detector
    detector = cv2.CascadeClassifier(facenet_path)
    
    # Capture and Wait for first frame
    for img_name in os.listdir(data_path):
        
        employee_id = int(img_name.split('.')[0])

        img_path = os.path.join(data_path, img_name)

        frame = cv2.imread(img_path)
        draw = frame.copy()
        
        # Capture Face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = detector.detectMultiScale(gray)        

        # Primary Model to detect face
        # face- Chin: 1-17, Eyebrows: 18-27, Nose: 28-36, Eyes: 37-48, Mouth: 49-61, Inner Lips: 62-68, Pupil: 69-76, Ears: 77-80,additional eye landmarks: 81-104
        for (x,y,w,h) in faces:
            
            # Parse
            x1, y1, x2, y2 = x, y, x+w, y+h
            face    = frame[y1:y2, x1:x2]

            # Draw
            draw   = cv2.rectangle(draw, (x1, y1), (x2, y2), (0,255,0), 2) 

            # Save Image
            save_path = os.path.join(save_folder, f'{employee_id}.jpg' )
            cv2.imwrite(save_path, face)

            # Secondary Model to predict face landmark
            landmarks = landmark_engine.predict(face)
            mini_h, mini_w = face.shape[:2]

            # Draw
            for idx, (lx, ly) in enumerate(landmarks):
                lx = int(lx * mini_w / landmark_engine.input_size[0])
                ly = int(ly * mini_h / landmark_engine.input_size[1])
                draw = cv2.circle(draw, (x+lx, y+ly), 1, (0,255,0), 1)
            
        # Display
        cv2.imshow(CV_WIN, draw)
        if cv2.waitKey(t_key_wait) in [ ord('q'), ord('Q'), 27]: break

        # Saving into DB
        db[employee_id] = landmarks
        write_json(db_path, db)
    
    cv2.destroyAllWindows()
    print('Generate Finished')

def get_arguments():

    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(    '-f', '--face', type=str, 
                                default='./model/haarcascade_frontalface_default.xml')
    arg_parser.add_argument(    '-l', '--lm', type=str, 
                                default='./model/fpenet_b1_fp32.trt')
    arg_parser.add_argument(    '-s', '--source', type=str, 
                                default='/dev/video0')
    arg_parser.add_argument(    '-m', '--mode', type=str, 
                                choices=['auto', 'debug'],
                                default='auto')
    return arg_parser.parse_args()    

if __name__ == '__main__':

    FACE    = 'face'
    LM      = 'landmark'
    ALL     = 'all'

    CV_WIN  = 'Face + Landmark'

    args = get_arguments()

    data_path = './data'

    face_lm_detector(
        facenet_path=args.face,
        fpenet_path=args.lm,
        data_path=data_path,
        mode=args.mode
    )