import datetime
import logging as log
import os.path
import tempfile
from pathlib import Path
import cv2

from insg.engine import Engine


class VideoEngine(Engine):

    def __init__(self):
        super().__init__("Video Engine")

        self.input = str(self.c.input.video)
        inp = self.input
        log.info(f"reading from {inp}")
        cap = cv2.VideoCapture(inp)
        ret, frame = cap.read()
        if not ret:
            log.error("Capture error")
            raise EOFError("Capture error " + inp)

        shape = frame.shape
        self.size = (shape[1], shape[0])
        log.info(f"frame size {self.size}")

        self.exclusions = self.create_exclusions()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        tf = tempfile.NamedTemporaryFile(suffix=".avi")
        self.temp_video = tf.name
        self.output_file = self._output_filename()
        tf.close()
        log.debug(f"Creating VideoWriter: {self.temp_video}, {self.size}")
        self.video_out = cv2.VideoWriter(self.temp_video, fourcc, 20.0, self.size)

        return

    def run_pipeline(self):
        log.info("Pipeline start")
        preview = "true" in str(self.c.output.preview).lower()
        log.debug(f"preview: {preview}")
        inp = str(self.c.input.video)
        cap = cv2.VideoCapture(inp)
        while cap.isOpened():
            if cv2.waitKey(1) == ord('q'):
                break
            read_correctly, frame = cap.read()
            if not read_correctly or frame is None:
                break
            log.info(f"got frame")
            # todo
            results = None
            ok, frame2 = self.process_results(results, frame)
            if preview:
                if ok:
                    cv2.imshow("rgb", frame2)
                else:
                    cv2.imshow("rgb", frame)
            if self.video_out:
                if ok:
                    self.video_out.write(frame2)
                else:
                    # self.video_out.write(frame)
                    # ignore frames w/o results
                    pass

        if self.video_out:
            self.video_out.release()
        assert len(self.output_file) > 0

        log.info(f"Convert to {self.c.output.type}")
        self._convert_to_mp4()
        assert os.path.exists(self.output_file)
        log.info(f"Output file is ready: {self.output_file}")

        log.info("Pipeline end")

    def process_results(self, results, frame):
        # # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
        # bboxes = np.array(in_nn.getFirstLayerFp16())
        # # transform the 1D array into Nx7 matrix
        # bboxes = bboxes.reshape((bboxes.size // 7, 7))
        # # filter out the results which confidence less than a defined threshold
        # bboxes = bboxes[bboxes[:, 2] > self.confidence]
        # if len(bboxes) == 0:
        #     return False, None
        #
        # # Cut bboxes and labels
        # labels = bboxes[:, 1].astype(int)
        # confidences = bboxes[:, 2]
        # bboxes = bboxes[:, 3:7]
        # # log.info(f"process_results conf: {confidence}, {len(bboxes)} bboxes")
        # # todo: config
        # color_bgr = (0, 250, 250)
        # font = cv2.FONT_HERSHEY_TRIPLEX
        # font_size = 0.9
        # thickness = 4
        count = 0
        # for raw_bbox, label, conf in zip(bboxes, labels, confidences):
        #     if label in self.exclusions:
        #         continue
        #     log.debug(f"conf: {conf}, label: {label} - {self.labels[label]}")
        #     count += 1
        #     bbox = _frame_norm(frame, raw_bbox)
        #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, thickness)
        #     cv2.putText(frame, self.labels[label], (bbox[0] + 10, bbox[1] + 20),
        #                 font, font_size, color_bgr)
        #     cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40),
        #                 font, font_size, color_bgr)
        if count:
            return True, frame
        else:
            return False, None

    def _output_filename(self):
        dir = self.c.output.dir
        assert (os.path.isdir(dir))
        assert (os.path.exists(dir))

        ts = datetime.datetime.now().isoformat(timespec='seconds')
        ts = ts.replace(":", "-")
        if str(self.input).startswith("rtsp:"):
            name = f"rtsp"
        else:
            name = Path(self.input).resolve().name
            assert len(name) > 0
            if '.' in name:
                el = name.split('.')
                name = el[-2]

        fname = f"{name}_{ts}_{self.c.var.name}.{self.c.output.type}"

        self.output_file = os.path.join(dir, fname)
        print(f"generate_output_file {self.input} -> {self.output_file}")
        return self.output_file

    def create_exclusions(self):
        exclusions = []
        if self.c.network.exclude:
            for n in str(self.c.network.exclude).split(","):
                ok, i = self.safe_label_index(n)
                if ok:
                    exclusions.append(i)
        log.debug(f"exclusions: {exclusions}")
        return exclusions

    def create_labels(self):
        with open(self.labels, 'r') as file:
            labels = [line.split(sep=' ', maxsplit=1)[-1].strip() for line in file]
            # log.debug(f"{len(self.labels)} labels")
        return labels

    def safe_label_index(self, s: str) -> (bool, int):
        max_index = len(self.labels)
        try:
            n = int(s)
            if max_index > 0 and 0 <= n < max_index:
                return True, n
        except ValueError:
            return False, 0

    def _convert_to_mp4(self):
        import subprocess
        inp = self.temp_video
        if not os.path.exists(inp):
            log.info(f"{inp} not found")
            return

        outp = self.output_file
        log.info(f"converting {inp} to {outp}")
        # ffmpeg -i debug.avi -y a.mp4
        res = subprocess.run(["ffmpeg", "-analyzeduration", "1000000", "-i", inp, "-y", outp])
        log.info(str(res))
        if ("returncode=0" in str(res)) and os.path.exists(outp):
            log.debug(f"Deleting {inp}")
            os.remove(inp)
        return

if __name__ == '__main__':
    v = VideoEngine()
    v.run_pipeline()



