"""
    @author: yee
    @date: 2021/1/20
    @description: 
"""
import argparse
import cv2
import torch.multiprocessing as multiprocessing
from datetime import datetime
from align_detect import AlignDetector


class VideoDetect(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.detector = AlignDetector()

    def capture_stream(self):
        """
        capture stream from camera or video file.
        """
        cap = cv2.VideoCapture(self.video_path)

        if cap.isOpened():
            print('[INFO] Capture stream successful')
            self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            return cap
        else:
            print('[INFO] Capture stream failed')
            return

    def receive_frame(self, q):
        """
        receive frame from stream.
        """
        cap = self.capture_stream()
        print('[INFO] Start receive frame...')
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            q.put(frame)
            # q.get() if q.full() else time.sleep(0.01)
            # print('[QUEUE SIZE] {}'.format(q.qsize()))
        cap.release()
        print('[INFO] Stream cut off')
    
    def process_frame(self, q):
        """
        process frame from stream
        """
        self.detector.init_model()

        print('[INFO] Start process frame...')
        while True:
            try:
                frame = q.get(timeout=5)
            except Exception:  # break if the waiting time exceeds 5 seconds
                print('[INFO] Queue is empty, exit')
                break
            
            _ = self.detector.detect(frame)
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (w // 2, h // 2))
            cv2.imshow("result", frame)
            cv2.waitKey(1)
        print('[INFO] Process done')
    
    def run(self):
        """[summary]
        """
        multiprocessing.set_start_method(method='spawn')
        q = multiprocessing.Queue(maxsize=100)
        p1 = multiprocessing.Process(target=self.receive_frame, args=(q,))
        p2 = multiprocessing.Process(target=self.process_frame, args=(q,))

        for p in [p1, p2]:
            p.daemon = True
            p.start()
        for p in [p1, p2]:
            p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", help="path of video file", default=None)
    args = parser.parse_args()

    detector = VideoDetect(args.video_path)
    start = datetime.now()
    detector.run()
    end = datetime.now()
    print('[INFO] All done. Time consuming: {}s'.format((end - start).seconds))
