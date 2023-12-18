# prompt: Vision transformer pipeline with huggingface transformers
import torch
import requests
from PIL import Image, ImageDraw
from transformers import ViltModel, ViltConfig, AutoTokenizer, ViltForQuestionAnswering, ViltProcessor, AutoModelForZeroShotObjectDetection, AutoProcessor


#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://media.istockphoto.com/id/1308453727/photo/three-young-house-cats.jpg?s=612x612&w=0&k=20&c=_Un40uwvdnpMroivx0yOYKa9ptVOYSFAbo68_ERc_Pk="
image = Image.open(requests.get(url, stream=True).raw)
text = "Are there dogs in the picture?"

#processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
#model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")

text_queries = ["cat", "eyes", "nose", "tail"]
inputs = processor(text=text_queries, images=image, return_tensors="pt")

with torch.no_grad():
    
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

draw = ImageDraw.Draw(image)

scores = results["scores"].tolist()
labels = results["labels"].tolist()
boxes = results["boxes"].tolist()

for box, score, label in zip(boxes, scores, labels):
    xmin, ymin, xmax, ymax = box
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{text_queries[label]}: {round(score,2)}", fill="white")
    
image.show()