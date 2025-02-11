import supervisely as sly
import os
from supervisely.nn.inference import CheckpointInfo
import torch
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
from supervisely.nn.inference.inference import (
    get_hardware_info,
    get_name_from_env,
    logger,
)
from PIL import Image
from supervisely.nn.prediction_dto import PredictionBBox


class Kosmos2(sly.nn.inference.PromptBasedObjectDetection):
    FRAMEWORK_NAME = "Kosmos 2"
    MODELS = "src/models.json"
    APP_OPTIONS = "src/app_options.yaml"
    INFERENCE_SETTINGS = "src/inference_settings.yaml"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # disable GUI widgets
        self.gui.set_project_meta = self.set_project_meta

    def load_model(
        self,
        model_files: dict,
        model_info: dict,
        model_source: str,
        device: str,
        runtime: str,
    ):
        checkpoint_path = model_files["checkpoint"]
        if sly.is_development():
            checkpoint_path = "." + checkpoint_path
        self.classes = []
        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=os.path.basename(checkpoint_path),
            model_name=model_info["meta"]["model_name"],
            architecture=self.FRAMEWORK_NAME,
            checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
            model_source=model_source,
        )
        self.device = torch.device(device)
        self.model = Kosmos2ForConditionalGeneration.from_pretrained(
            checkpoint_path,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.model = self.model.to(self.device)

    def predict(self, image_path, settings):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        text_prompt = settings.get("text_prompt", "<grounding> An image of")
        if not text_prompt.startswith("<grounding>"):
            text_prompt = "<grounding> " + text_prompt
        max_new_tokens = int(settings.get("max_new_tokens", 128))
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(
            self.device
        )
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        predictions = self.postprocess_generated_text(generated_text, image)
        return predictions

    def predict_batch(self, images_np, settings):
        text_prompt = settings.get("text_prompt", "<grounding> An image of")
        if not text_prompt.startswith("<grounding>"):
            text_prompt = "<grounding> " + text_prompt
        max_new_tokens = int(settings.get("max_new_tokens", 128))

        images = [Image.fromarray(img) for img in images_np]
        text_prompt = [text_prompt] * len(images)

        inputs = self.processor(
            text=text_prompt, images=images, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=64,
        )
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        batch_predictions = []
        for generated_text, image in zip(generated_texts, images):
            batch_predictions.append(
                self.postprocess_generated_text(generated_text, image)
            )
        return batch_predictions

    def postprocess_generated_text(self, generated_text, image):
        caption, entities = self.processor.post_process_generation(generated_text)
        predictions = []
        for entity in entities:
            labels = [entity[0]] * len(entity[2])
            bboxes = entity[2]
            for label, bbox in zip(labels, bboxes):
                x1, y1, x2, y2 = bbox
                x1 = round(x1 * image.width)
                y1 = round(y1 * image.height)
                x2 = round(x2 * image.width)
                y2 = round(y2 * image.height)
                bbox_yxyx = [y1, x1, y2, x2]
                sly_bbox = PredictionBBox(label, bbox_yxyx, None)
                predictions.append(sly_bbox)
        return predictions

    def set_project_meta(self, inference):
        """The model does not have predefined classes.
        In case of prompt-based models, the classes are defined by the user."""
        self.gui._model_classes_widget_container.hide()
        return

    def _load_model_headless(
        self,
        model_files: dict,
        model_source: str,
        model_info: dict,
        device: str,
        runtime: str,
        **kwargs,
    ):
        """
        Diff to :class:`Inference`:
           - _set_model_meta_from_classes() removed due to lack of classes
        """
        deploy_params = {
            "model_files": model_files,
            "model_source": model_source,
            "model_info": model_info,
            "device": device,
            "runtime": runtime,
            **kwargs,
        }
        self._load_model(deploy_params)

    def _create_label(self, dto):
        """
        Create a label from the prediction DTO.
        """
        class_name = dto.class_name + " (bbox)"
        obj_class = self.model_meta.get_obj_class(class_name)
        if obj_class is None:
            self._model_meta = self.model_meta.add_obj_class(
                sly.ObjClass(class_name, sly.Rectangle)
            )
            obj_class = self.model_meta.get_obj_class(class_name)
        geometry = sly.Rectangle(*dto.bbox_tlbr)
        tags = []
        if dto.score is not None:
            tags.append(sly.Tag(self._get_confidence_tag_meta(), dto.score))
        label = sly.Label(geometry, obj_class, tags)
        return label
