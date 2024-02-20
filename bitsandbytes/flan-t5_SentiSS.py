import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig
import torch

model_folder = "/mnt/New/Data/Vbox_SF/HuggingFaceLocal/"
#model_folder = "/home/debbie/Dev/HF Finetuning/finetuned/"
model_name = "flan-t5-xl"
# model_folder = "/home/debbie/Dev/HF Finetuning/models/"
# model_name = "flan-t5-base"
model = model_folder + model_name
max_output_tokens = 200
tokenizer = T5Tokenizer.from_pretrained(model, local_files_only=True, legacy=True)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

start_time = time.perf_counter()

model = T5ForConditionalGeneration.from_pretrained(
    model, 
    device_map="auto",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
)
# model = T5ForConditionalGeneration.from_pretrained(model)

end_time = time.perf_counter()
total_time = end_time - start_time
print("Total model load time (seconds): " + str(total_time))

prompt_template  = """
Here is a product review from a customer, which is delimited with triple backticks.

Product Name: [[PRODUCT_NAME]]
Review text: 
```
[[REVIEW_TEXT]]
```

Overall sentiment must be one of the following options:
-Positive
-Negative
-Neutral

What is the overall sentiment of that product review? 

Answer:"""


reviews = [
    {"product_name": "Bliss Rocking Chair",
     "review_text": "I am absolutely in love with it! The chair is not only beautiful, but also incredibly comfortable. The rocking motion is smooth and soothing, making it perfect for relaxing after a long day. The craftsmanship is top-notch, with sturdy construction and high-quality materials. I also appreciate the attention to detail in the design, such as the curved armrests and the ergonomic backrest. Overall, I highly recommend it to anyone looking for a stylish and comfortable addition to their home. It truly brings joy and relaxation to my life!",
     "expected_answer": "Positive"},
    {"product_name": "Kyushu Calm Lounge Sofa",
     "review_text": "The quality of the fabric on this couch is okay, but it's not the most comfortable seating I've experienced. It looks nice in my living room, though.",
     "expected_answer": "Neutral"},
    {"product_name": "Vintage Elegance Vanity",
     "review_text": "I am extremely disappointed with it. The quality of the product is very poor and does not meet my expectations at all. The materials used feel cheap and the vanity arrived with several scratches and dents. The assembly instructions were also confusing and the whole process took much longer than anticipated. I would not recommend this product to anyone.",
     "expected_answer": "Negative"},
    {"product_name": "Sahara Sands Canopy Bed",
     "review_text": "I absolutely love my Sahara Sands Canopy Bed! It adds a touch of elegance to my bedroom and the quality is exceptional. The assembly was a bit time-consuming, but the end result is worth it. The design is beautiful and the bed is very sturdy. Definitely worth the investment.",
     "expected_answer": "Positive"},
    {"product_name": "MaxGrip Screwdriver",
     "review_text": "OK I guess. Handle is too small for XL hands. No PZ2 bit (everything uses PZ2s). Delivery was slow. Customer service was slow to reply. Screwdriver mechanism is fine, it was the biggest selling point of the driver. Magmatism is strong, holds screw well.",
     "expected_answer": "Neutral"},
    {"product_name": "Himalaya Summit Coffee Table",
     "review_text": "I was very disappointed with the quality of this coffee table. It arrived with scratches all over and the wood was chipped in several places. Definitely not worth the price. Would not recommend.",
     "expected_answer": "Negative"},
     {"product_name": "ProTech Precision Screwdriver",
      "review_text": "This ProTech Precision Screwdriver has been an absolute game-changer for my toolkit. I primarily use it for intricate electronics and gadget repairs, and it's proven to be incredibly efficient. The ergonomic design ensures a comfortable grip, even during long use periods, which is a significant plus for me. The precision and strength of the tips are noteworthy, allowing me to work on delicate components without fear of stripping screws. The addition of multiple bit sizes, including the less common ones, makes this screwdriver versatile for a variety of tasks. Although the magnetic tip strength could be better to hold screws more securely, it's a minor issue compared to the overall quality and functionality. I've recommended this to several colleagues, and they all share my high opinion. Excellent product!",
      "expected_answer": "Positive"},
     {"product_name": "Maracay Rattan Armchair",
      "review_text": "The Maracay Rattan Armchair is a decent choice for outdoor seating. The rattan material feels sturdy and the chair is comfortable to sit in. However, the assembly instructions were a bit confusing and the chair wobbles slightly on uneven ground. Overall, it's an average product.",
      "expected_answer": "Neutral"},
     {"product_name": "Ember Ottoman",
      "review_text": "The Ember Ottoman is not very sturdy and the fabric started to tear after just a few weeks of use. I was disappointed with the quality for the price.",
      "expected_answer": "Negative"},
     {"product_name": "Jaipur Tapestry Bookshelf",
      "review_text": "The Jaipur Tapestry Bookshelf looks nice but it was difficult to assemble and the quality is not as sturdy as I had hoped. It serves its purpose but I expected better for the price.",
      "expected_answer": "Neutral"},

]

start_time = time.perf_counter()
score = 0
max_score = 0
runs = 10
for i in range(runs):
    for review in reviews:
        input_text = prompt_template.replace("[[PRODUCT_NAME]]", review['product_name']).replace("[[REVIEW_TEXT]]", review['review_text'])
        
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        # input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        #outputs = model.generate(input_ids, max_new_tokens=max_output_tokens, do_sample=True, temperature=0.6)
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_output_tokens)
        llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if review['expected_answer'] == llm_answer: score = score + 1
        else:
            print("[" + review['product_name'] +  "] Expected vs LLM: " + review['expected_answer'] + "->" + llm_answer)
        max_score = max_score + 1

end_time = time.perf_counter()
total_time = end_time - start_time
print("Final score:" + str(score) + " / " + str(max_score))
print("Total inference time (seconds): " + str(total_time))