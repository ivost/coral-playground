import datetime
import logging as log
import os.path
import tempfile
from pathlib import Path
import cv2
from pycoral.adapters import common
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import run_inference

from engine import Engine


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

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
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
        inp = str(self.c.input.video)
        cap = cv2.VideoCapture(inp)
        conf = float(self.c.network.confidence)
        top_k = int(self.c.network.top_k)
        self.size = input_size(self.coral)
        log.info(f"preview: {preview}, conf: {conf}, top_k: {top_k}")

        while cap.isOpened():
            key = cv2.waitKey(1)
            if key == ord('q'):
                log.info(f"{key} pressed")
                break
            ok, frame = cap.read()
            if not ok:
                continue
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, self.size)
            run_inference(self.coral, cv2_im_rgb.tobytes())
            results = get_objects(self.coral, conf)[:top_k]
            log.info(f"{len(results)} result(s)")
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

        cap.release()
        if self.video_out:
            self.video_out.release()
        assert len(self.output_file) > 0
        cv2.destroyAllWindows()
        log.info(f"Convert to {self.c.output.type}")
        self._convert_to_mp4()
        assert os.path.exists(self.output_file)
        log.info(f"Output file is ready: {self.output_file}")

        log.info("Pipeline end")

    def process_results(self, results, frame):
        height, width, channels = frame.shape
        scale_x, scale_y = width / self.size[0], height / self.size[1]
        count = 0
        for obj in results:
            count += 1
            bbox = obj.bbox.scale(scale_x, scale_y)
            x0, y0 = int(bbox.xmin), int(bbox.ymin)
            x1, y1 = int(bbox.xmax), int(bbox.ymax)
            label = self.labels.get(obj.id, obj.id)
            label = f"{label} {obj.score:.2f}"
            color = (0, 255, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, label, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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



