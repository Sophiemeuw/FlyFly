import numpy as np 
import logging
from collections import deque
from enum import IntEnum
import scipy.signal
from pathlib import Path
import imageio
from copy import deepcopy
from flygym.vision import Retina


class RawVideoHandler:
    def __init__(self, file_name_prefix):
        self.save_path = Path("debug")/f"{file_name_prefix}_raw_vision.mp4"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.retina = Retina()
        self.frame = None

    def __enter__(self): 
        self.writer = imageio.get_writer(self.save_path, fps=5) # unsure about fps

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()

    
    def add_frame(self, frame): 
        left_readable = self.retina.hex_pxls_to_human_readable(frame[0, :, :], color_8bit=True)
        right_readable = self.retina.hex_pxls_to_human_readable(frame[1, :, :], color_8bit=True)

        frame = np.hstack((left_readable, right_readable))
        if type(self.frame) != np.ndarray: 
            self.frame = frame
        else:
            self.frame = np.vstack((self.frame, frame))
        
    def commit_frame(self): 
        # This is to make imageio stop complaining:
        # input image is not divisible by macro_block_size=16, resizing from (900, 514) to (912, 528) to ensure video compatibility with most codecs 
        # and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 
        # (risking incompatibility).
        if self.frame.shape[0] % 16 != 0 or self.frame.shape[1] % 16 != 0: 
            new_height = (self.frame.shape[0] + 15) // 16 * 16
            new_width = (self.frame.shape[1] + 15) // 16 * 16
            self.frame = np.pad(
                self.frame,
                ((0, new_height - self.frame.shape[0]), (0, new_width - self.frame.shape[1]), (0, 0)),
                mode='constant',
                constant_values=0
            )


        self.writer.append_data(self.frame)
        self.frame = None



class EMDDirection(IntEnum): 
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class RectificationType(IntEnum): 
    ON = 1
    OFF = 2


class LoomDetector: 
    def __init__(self, input_size: tuple, debug=False): 
        TS = 0.001
        HPF_TAU = 0.250
        HPF_CORNER = 1/(2 * np.pi * HPF_TAU)
        BUF_SIZE = 10

        self.frames_recvd = 0
        self.logger = logging.getLogger("ld")
        

        self.frame_buffer = deque([np.zeros(input_size) for _ in range(BUF_SIZE)], BUF_SIZE)

        self.emd_signals = {dir: 0 for dir in EMDDirection}
        self.hpf_b, self.hpf_a = scipy.signal.butter(1, HPF_CORNER, 'high', output='ba', fs=1/TS)
        self.hpf_history = deque([np.zeros(input_size)], 1)

        self.initialized = False


        self.debug = debug
        if debug:
            self.rvh = RawVideoHandler("hpf")
    
    def __enter__(self):
        if self.debug: 
            self.rvh.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            self.rvh.__exit__(exc_type, exc_value, traceback)
        
    
    def edge_rectification(self, input: np.ndarray, threshold: float, type: RectificationType): 
        input = input.copy()
        pre_shape = input.shape
        flattened = input.ravel()

        if type == RectificationType.ON: 
            pattern = flattened > threshold
        elif type == RectificationType.OFF:
            pattern = flattened < threshold
        
        flattened[pattern] = np.abs(flattened[pattern] - threshold)
        flattened[~pattern] = np.zeros(np.count_nonzero(~pattern))

        return flattened.reshape(pre_shape)
        

    def process(self, images: np.ndarray) -> float:
        PROCESS_PERIOD = 1
        EMD_OFFSET = 5

        self.frames_recvd += 1

        # first, take the image and combine the colour channels (they are useless to us)
        
        if not self.initialized and self.frames_recvd < self.frame_buffer.maxlen: 
            # just add the frame to the buffer
            self.frame_buffer.appendleft(images)
            return 0
        
        self.initialized = True

        self.frames_recvd = 0
        self.frame_buffer.appendleft(frame)

        hpf = self.hpf_b[0]*images + self.hpf_b[1]*self.frame_buffer[1] - self.hpf_a[1] * self.hpf_history[0]

        # first order hpf in pipeline 
        self.hpf_history.appendleft(hpf)

        on_rect = self.edge_rectification(hpf, 0, RectificationType.ON)
        off_rect = self.edge_rectification(hpf, 1, RectificationType.OFF)
                
    


        if self.debug:
            self.rvh.add_frame(self.frame_buffer[0])
            self.rvh.add_frame(self.hpf_history[0])
            self.rvh.add_frame(on_rect)
            self.rvh.add_frame(off_rect)
            self.rvh.commit_frame()
        # else: 
        #     pass # maybe log
    
        

    
        




if __name__ == "__main__": 
    import pickle
    with open("/home/niel/Documents/repos/controlling-behaviour/flyfly/.stuff/vid_hist.pkl", "rb") as f:
        data = pickle.load(f)
    

    
    ld = LoomDetector((2, 721, 2), debug=True)
    with ld: 
        for i, frame in enumerate(data["vision"]):
            if i < 1000: 
                continue 
            ld.process(frame)
    


