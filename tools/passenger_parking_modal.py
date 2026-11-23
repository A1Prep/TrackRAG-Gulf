"""Deploy the authenticated on-demand passenger GPU backend with modal deploy -m tools.passenger_parking_modal."""
from pathlib import Path
import os
import modal
from tools.passenger_parking_call import APP

app = modal.App(APP)
root = Path(__file__).resolve().parents[1]
parking = modal.Volume.from_name("trackrag-parking-vistas-weights")
qwen = modal.Volume.from_name("trackrag-qwen35-9b-switch-trial")
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0","torchvision==0.21.0","numpy==1.26.4","Pillow==11.1.0")
    .pip_install("transformers==5.6.2","accelerate==1.12.0","safetensors==0.8.0","tokenizers==0.22.2",
                 "opencv-python-headless==4.10.0.84","scipy==1.15.2","cryptography==50.0.2","sentencepiece==0.2.1")
    .env({"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false","HF_HUB_DISABLE_TELEMETRY":"1",
          "OMP_NUM_THREADS":"4","HF_HUB_DISABLE_PROGRESS_BARS":"1"})
    .add_local_dir(root/"trackrag","/root/trackrag",ignore=lambda p:p.is_file() and p.suffix != ".py"))
for filename in ("dropoff_predict.py","passenger_parking_study.py","passenger_parking_anchors.py",
                 "passenger_parking_candidates.py","passenger_road_points.py"):
    image=image.add_local_file(root/"tools"/filename,"/root/tools/"+filename)


@app.cls(image=image,gpu="A100-40GB",cpu=4,memory=30000,timeout=180,startup_timeout=240,
         min_containers=0,max_containers=1,scaledown_window=30,retries=0,
         volumes={"/parking":parking.read_only(),"/qwen":qwen.read_only()},
         secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
class PassengerParking:
    @modal.enter()
    def load(self):
        from tools.passenger_parking_study import QWEN_REVISION, SEG_REVISION
        from trackrag.dropoff.parking_inference import ParkingModels
        self.models=ParkingModels("/parking/"+SEG_REVISION,"/qwen/"+QWEN_REVISION,os.environ.pop("PARKING_SIGNING_KEY"))

    @modal.method()
    def interpret(self, request, image, recipe=None):
        return self.models.run(request, image, recipe)
