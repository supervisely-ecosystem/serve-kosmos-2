import supervisely as sly
from dotenv import load_dotenv
from src.kosmos2 import Kosmos2


if sly.is_development():
    load_dotenv("local.env")
    load_dotenv("supervisely.env")

model = Kosmos2(
    use_gui=True,
    use_serving_gui_template=True,
    sliding_window_mode="none",
)
model.serve()