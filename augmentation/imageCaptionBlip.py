from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import csv
import os
import torch

processor =  Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")

image_folder = 'flickr30k_images_sample_4clip'
output_csv = 'captionsBlip4clips.csv'

image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
               filename.endswith('.jpg')]

count = 0
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(['Image_ID', 'Caption'])

    for image_file in image_files:
        print(count)
        count = count + 1
        image_id = os.path.splitext(os.path.basename(image_file))[0]
        raw_image = Image.open(image_file).convert('RGB')

        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)

        output_ids = processor.decode(out[0], skip_special_tokens=True)
        writer.writerow([image_id, output_ids])

print(f"Generated caption saved to {output_csv}")
