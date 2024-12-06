import pandas as pd
import requests
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import CLIPProcessor, CLIPModel


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()

df = pd.read_csv('captionsBlip2clips_new.csv', header=None, sep='\t')

count = 0

results_list = []
with torch.no_grad():
    for index, row in df.iterrows():
        image_path, text = row[0], row[1]
        image = Image.open("flickr30k_images_sample_2clip/" + image_path + ".jpg").convert('RGB')
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to('cuda')
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        results_list.append([image_path, logits_per_image.item()])

        count = count + 1
        print(count)

    output_df = pd.DataFrame(results_list, columns=['path', 'score'])
    output_df.to_csv('output_scores2clips.csv', index=False, sep='\t')

