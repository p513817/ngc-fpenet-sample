import time, cv2, math
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

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

def get_distance(pt, pt2):
    pt = ( float(pt[0]), float(pt[1]) )
    pt2 = ( float(pt2[0]), float(pt2[1]) )
    return math.hypot( pt2[0]-pt[0], pt2[1]-pt[1])

def get_coord_distance(p1 , p2):
    coordinate_distance = math.sqrt( ((int(p1[0])-int(p2[0]))**2)+((int(p1[1])-int(p2[1]))**2) )
    return coordinate_distance

def read_json(path):
    data = json.load(open(path))
    ret_data = {}
    for key, val in data.items():
        key = int(key) if key.isdigit() else key
        ret_data[key] = val
    return ret_data

def write_json(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data, cls=NumpyArrayEncoder))
