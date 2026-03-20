
from roboflow import Roboflow
rf = Roboflow(api_key="r6ycInQeA8kfK0H4615x")
project = rf.workspace("thanhdinhs-workspace").project("trai-cay-myk0t")
version = project.version(3)
dataset = version.download("yolov8")
                