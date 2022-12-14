import time, cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

# Helper Function
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# FPE Net
class FpeNet(object):
    
    def __init__(self, trt_path, input_size=(80, 80), batch_size=1):
        
        cuda.init()
        self.cfx    = cuda.Device(0).make_context()

        self.trt_path   = trt_path
        self.input_size = input_size
        self.batch_size = batch_size

        TRT_LOGGER      = trt.Logger(trt.Logger.WARNING)
        trt_runtime     = trt.Runtime(TRT_LOGGER)
        
        self.trt_engine = self._load_engine(trt_runtime, self.trt_path)
        self.inputs, self.outputs, self.bindings, self.stream = \
            self._allocate_buffers()

        self.context = self.trt_engine.create_execution_context()

        # Fix dynamic shape to 1
        self.context.set_binding_shape(0, (1,1,80,80) )

    def store_runtime(self):
        if self.cfx:
            self.cfx.push()

    def clear_runtime(self):
        if self.cfx:
            self.cfx.pop()

    def _load_engine(self, trt_runtime, engine_path):
        # Deserialize
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        return trt_runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self):
        inputs      = []
        outputs     = []
        bindings    = []
        stream      = cuda.Stream()

        binding_to_type = {
            "input_face_images:0": np.float32,
            "softargmax/strided_slice:0": np.float32,
            "softargmax/strided_slice_1:0": np.float32
        }

        # For dynamic shape
        self.batch_size = -1
        for binding in self.trt_engine:
            size = trt.volume(self.trt_engine.get_binding_shape(binding)) \
                   * self.batch_size

            dtype       = binding_to_type[str(binding)]
            host_mem    = cuda.pagelocked_empty(size, dtype)
            device_mem  = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.trt_engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def _do_inference(self, context, bindings, inputs,
                      outputs, stream):
        
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) \
         for inp in inputs]

        context.execute_async(
            batch_size=1, bindings=bindings,
            stream_handle=stream.handle)

        [cuda.memcpy_dtoh_async(out.host, out.device, stream) \
         for out in outputs]

        stream.synchronize()

        return [out.host for out in outputs]

    def _process_image(self, image):
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        w = self.input_size[0]
        h = self.input_size[1]
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        image_resized = Image.fromarray(np.uint8(image))
        image_resized = image_resized.resize(size=(w, h), resample=Image.BILINEAR)
        img_np = np.array(image_resized)
        img_np = img_np.astype(np.float32)

        img_np = np.expand_dims(img_np, axis=0)  # the shape would be 1x80x80

        return img_np, image

    def predict(self, source, is_path=False):

        image = cv2.imread(source) if is_path else source
        
        # Preprocess
        img_processed, image = self._process_image(image)
        np.copyto(self.inputs[0].host, img_processed.ravel())

        # Do inference
        landmarks = None
        self.store_runtime()
        for _ in range(1):
            landmarks, probs = self._do_inference(
                self.context, bindings=self.bindings, inputs=self.inputs,
                outputs=self.outputs, stream=self.stream)
        self.clear_runtime()
        
        # to make (x, y)s from the (160, ) output
        landmarks = landmarks.reshape(-1, 2)

        return landmarks

    @staticmethod
    def _postprocess(landmarks):
        landmarks = landmarks.reshape(-1, 2)
        return landmarks

    def visualize(self, frame, landmarks):
        # visualized = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        visualized = frame
        for x, y in landmarks:
            x = x * self.image_width / self.input_size[0]
            y = y * self.image_height / self.input_size[1]
            x = int(x)
            y = int(y)
            cv2.circle(visualized, (x, y), 1, (0, 255, 0), 1)
        return visualized

    def __del__(self):
        self.clear_runtime()

def face_detector(model, source):
    # Face Model
    detector = cv2.CascadeClassifier(model)

    cap = cv2.VideoCapture(source)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # ?????????????????????
        face = detector.detectMultiScale(gray)        # ??????????????????
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