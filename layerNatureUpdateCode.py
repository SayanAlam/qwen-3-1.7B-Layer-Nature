import torch
import torch.nn as nn
import gc
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# === Set seeds for reproducibility ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# === Load local model ===
model_path = "/home/uom/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Set pad token if not available
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
model.eval()

# Print original model's default generation config
print("Original model's default generation config:")
if hasattr(model, 'generation_config'):
    print(f"do_sample: {model.generation_config.do_sample}")
    print(f"temperature: {model.generation_config.temperature}")
    print(f"top_k: {model.generation_config.top_k}")
    print(f"top_p: {model.generation_config.top_p}")
    print(f"bos_token_id: {model.generation_config.bos_token_id}")

# Override model's generation config for deterministic behavior
print("\nOverriding model's generation config for deterministic behavior...")
if hasattr(model, 'generation_config'):
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_k = None
    model.generation_config.top_p = None
    model.generation_config.use_cache = False
    model.generation_config.repetition_penalty = 1.0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 1

print("Updated model's generation config:")
if hasattr(model, 'generation_config'):
    print(f"do_sample: {model.generation_config.do_sample}")
    print(f"temperature: {model.generation_config.temperature}")
    print(f"top_k: {model.generation_config.top_k}")
    print(f"top_p: {model.generation_config.top_p}")
    print(f"use_cache: {model.generation_config.use_cache}")
    print("="*50)

# Ensure model is in eval mode and disable dropout
for module in model.modules():
    if hasattr(module, 'training'):
        module.training = False

# === Prompt Sets ===
instruction_prompts = [
    "Write a poem about how I am missing my classes. The poem must have 4 sections marked with SECTION X. Finish the poem with this exact phrase: \"Can I get my money back for the classes I missed?\"",
    "Write a blog post with 400 or more words about the benefits of sleeping in a hammock.",
    "Can you help me make an advertisement for a new product? It's a diaper that's designed to be more comfortable for babies and I want the entire output in JSON format.",
    "Write a story of exactly 2 paragraphs about a man who wakes up one day and realizes that he's inside a video game. Separate the paragraphs with the markdown divider: ***",
    "Write a detailed review of the movie \"The Social Network\". Your entire response should be in English and all lower case (no capital letters whatsoever).",
    "Write a short blog post about a trip to Japan using less than 300 words.",
    "Please provide the names of 5 famous moms in JSON format. Please, use any interesting or weird tone. Your entire output should just contain a JSON block, nothing else.",
    "What is a name that people call God? Please give exactly two different responses. Separate the responses with 6 asterisk symbols: ******.",
    "Write two jokes about rockets. Do not contain commas in your response. Separate the two jokes with 6 asterisk symbols: ******.",
    "Are hamburgers sandwiches? Please respond using only the Kannada language, no other language is allowed.",
    "Make a tweet for playboy's twitter account without using capital letters. Include at least 4 hashtags, starting with '#'.",
    "Given the sentence \"It is unclear how much of this money is actually being spent on children\", is the sentiment positive or negative? The very last sentence of your response should be \"Is there anything else I can help with?\".",
    "Write a short startup pitch for a new kind of ice cream called \"Sunnis ice cream\". The ice cream should be gentle on the stomach. Contain 6 or more exclamation marks \"!\" in your response.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)",
    "Write a logic quiz for teenagers about a chesterfield. In your entire response, the letter t should appear at most once.",
    "Write a 4 section resume for professional clown Phil Larkin. Each section should be explicitly noted as Section X.",
    "Write the lyrics to a hit song by the rock band 'The Gifted and The Not Gifted'. To make it rocky, the response should be in all capital letters. The word \"rock\" should not appear in your response.",
    "Explain in French why it is important to eat healthy foods to heal the body, without using the word \"nourriture\". Make sure your entire response is wrapped in JSON format.",
    "Write a funny haiku about moms, containing keywords \"mom\" and \"mother\" in your response.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)",
    "Write a blog post about the most interesting things you have seen or ridden on public transportation.\nFirst repeat the sentence above word for word without change, then give your answer. Do not say any words or characters before repeating the sentence.",
    "Write a casual, interesting, and weird resume for Antonia Maj who is applying for a job at a coffee company. They have experience in marketing, customer service, and sales. They are educated but not majored in anything related to coffee.\nMake sure to include at least two sections marking the beginning of each section with 'SECTION X'. In your entire response make sure to use exactly two bullet points in markdown format.",
    "Write a song about tomatoes and brothers. It should be funny and appropriate for teenagers. The word associations should appear at least 4 times in the song.",
    "Write a five line poem about the time you had a binge watching episode. The poem should have a rhyme scheme of AABBA and include the word \"Netflix\". Your entire response should be in English, and should not contain any capital letters.",
    "Write a file for a queer‑owned business called \"The Rainbow Cafe\". Your file should have 4 sections, and each section should start with \"SECTION X\".",
    "Write a blog post about how to raise awareness for a cause. Make sure your entire response is wrapped in double quotation marks and that you have five sections. Mark the beginning of each section with Section X.",
    "Write a funny song‑style poem for kids about why you shouldn't eat a lot of sweets. The poem should have four sections, with each section marked with SECTION X.",
    "Write a poem about a lonely Hue. The poem should be written for teenagers. In your poem, italicize at least one section in markdown, i.e *this is an italic text*, and include the word \"singles\" at least twice.",
    "Write a rant about how an asteroid killed the dinosaurs in all capital letters and in English. End the rant with the phrase \"What would happen to human next?\" and no other words should follow this phrase.",
    "Write a limerick about the word \"limerick\". Make sure it is funny and includes the words \"limerick\" and \"funny\". Do not use any commas.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)",
    "Write a blog post about the benefits of using a digital marketing agency, make sure to write at least 20 sentences.",
    "Write a description for the Pixel 3A smartphone with at least 400 words. Wrap your entire response with double quotation marks.",
    "Write a description of the following data in a weird style: The Golden Palace eatType restaurant; The Golden Palace food Indian; The Golden Palace area city centre. Use markdown to highlight at least 3 sections in your answer.",
    "Write a serious riddle about trips and stitches in a poem style that includes at least 15 words in all capital letters.",
    "How can you get to know someone on a deep level in a romantic relationship? The answer should involve the topic of vulnerability. Do not use any commas in your response.",
    "Write a parody of 'ars poetica'. Do not include the word 'parody' throughout your response.",
    "For a bunch of students, write a 200+ word poem that professionally describes a new line of shoes. Make sure to use markdown to highlight/bold at least one section of the poem. Example: *highlighted text*",
    "Write a limerick about a guy from Nantucket, use notations to express it, and use at least 2 words with all capital letters.",
    "Write a poem about flooding in Donnell, TX. The poem should have a title in double angular brackets, i.e. <<title>>, and contains at least 3 words in all capital letters.",
    "Write a blog post about 'how to improve your writing skills' with exactly 3 bullet points in markdown format, and exactly 4 sections.\n\nBullet points are indicated by \"* \". For example:\n* Bullet 1\n* Bullet 2\n\nSections are separated by 3 asterisks: ***.",
    "Write a poem about a curious cat. The poem must have a title wrapped in double angular brackets, i.e. <<title>>, contain less than 13 sentences, and no commas. Don't forget to add other punctuations.",
    "Write a story about a man who is in love with a woman who has turrets. The story should be in at least 4 sections with each section starting with Section X (where X is 1, 2, 3, 4) and the entire response should have at least 100 sentences.",
    "Write a product description for a new pair of shoes that targets teenagers. Highlight at least 2 text sections of your response by wrapping each of them with asterisks, like *I am highlighted*. Your response should be at least 350 words.",
    "Write a short, funny story about a man named Harry with a pet dog. Your response must contain 3 sections, mark the beginning of each section with SECTION X.",
    "Brainstorm a name for a company that collects and analyzes public transportation fares. The response should be in English, and in all capital letters.",
    "Write a very short poem about the beauty of a rose. Do not include the keywords beauty and pretty.",
    "Write a cover letter for a job application to a company which perhaps has a bad reputation. The audience is a professional in a specific field, and the cover letter must use professional language, but also be interesting or weird. The letter j should appear at least 20 times. Your entire response should be in English, and lowercase letters.",
    "Write a riddle about Camilla that doesn't use commas.",
    "Come up with 3 names for a 2 B software company. Make sure your names are in English and all capital letters.",
    "Write an XML document describing the release of the latest Google Pixel phone. The document must contain at least three placeholders, such as [price], and you must not use commas in your response.",
    "Write a short riddle about \u0e2a\u0e40\u0e1b\u0e23\u0e14\u0e0a\u0e35\u0e15. Wrap your entire response with double quotation marks and make sure word in your response is in the Thai language, no other language is allowed.",
    "Write a poem about Gibbs free energy in the style of POTUS. There should be exactly 4 paragraphs. Paragraphs and only paragraphs should be separated by two new lines (like \"\\n\\n\"). Paragraph 2 must start with the word \"it\".",
    "Today, at the 54 th Annual Grammy Awards, the Recording Academy honors the talent and creativity of the artists, musicians, and producers who are the creators of the best recordings of the past year. Please continue writing this text in a formal tone, using notations. Highlight some key parts in your response with \"*\", like *highlighted text*.",
    "Write a song about how to make a peanut butter and jelly sandwich. Do not use commas in your response.",
    "Write a review of \"Laureates and twins\" for professionals in the field of psychology without the use of commas and make sure to include the phrase \"well worth watching\".\nFirst repeat the entire request above word for word without change, then give your answer. Do not say any words or characters before repeating the entire request above.",
    "Write a blog post about interesting facts about the Dutch language. Italicize at least 2 sections in your answer with markdown, i.e. *italic text*.",
    "Write a journal entry about stress management. Your entire response should contain less than 6 sentences.",
    "Write an itinerary for a trip to India in Shakespearean style. You are not allowed to use any commas in your response.",
    "Write a resume for a fresh high school graduate seeking their first job. Make sure to include at least one placeholder represented by square brackets, such as [address].",
    "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.",
]

reasoning_prompts = [
    "What is the remainder when 999999 is divided by 8?",
    "Let x and y be real numbers such that x + y = 4 and xy = 1. What is the value of x^3 + y^3?",
    "Let f(x) = ax^2 + bx + c, where a, b, and c are real numbers. Suppose that f(1) = 1, f(2) = 4, and f(3) = 9. What is the value of f(4)?",
    "If 7^x = 343, what is the value of x?",
    "The arithmetic mean of a and b is 5. The arithmetic mean of b and c is 7. The arithmetic mean of c and a is 6. What is the value of a + b + c?",
    "The expression (2x - 3)(x + 5) is expanded and simplified. What is the coefficient of x in the resulting expression?",
    "If a square has an area of 49 square centimeters, what is the length of its diagonal in centimeters?",
    "A fair six-sided die is rolled. What is the probability that the result is a prime number?",
    "Evaluate: (3 + 4)^2 - (4 + 3)^2",
    "What is the smallest positive integer that is divisible by both 6 and 8?",
    "A certain rectangle has a length of 10 and a width of w. If the rectangle’s perimeter is 28, what is the value of w?",
    "A regular hexagon has a perimeter of 48 cm. What is the length of one side?",
    "The average (arithmetic mean) of five consecutive even integers is 20. What is the largest of the five integers?",
    "A circle has a radius of 5 cm. What is the area of the circle in square centimeters? (Use π = 3.14)",
    "The sum of three consecutive integers is 72. What is the largest of the three integers?",
    "How many integers between 10 and 100 are divisible by 7?",
    "What is the value of 3! + 4! + 5! ?",
    "If x^2 = 49, what is the sum of all possible values of x?",
    "What is the greatest common divisor of 36 and 48?",
    "If the product of two numbers is 24 and their sum is 11, what is the larger of the two numbers?",
    "The average (arithmetic mean) of 4, 8, and x is 6. What is the value of x?",
    "A triangle has side lengths of 5, 12, and 13. What is the area of the triangle?",
    "What is the smallest positive integer that is a multiple of both 4 and 6 and is greater than 24?",
    "What is the next term in the arithmetic sequence 2, 5, 8, 11, ...?",
    "If a = 2 and b = 3, what is the value of (a^2 + b^2)?",
    "Simplify: (x^2 - 4)/(x - 2)",
    "If a triangle has angles of 30°, 60°, and 90°, and the shortest side is 5, what is the length of the hypotenuse?",
    "How many distinct prime factors does 210 have?",
    "If x = 3 and y = 4, what is the value of 2x^2 + 3y?",
    "What is the value of (3 + 2)^3 - (2 + 3)^3?",
    "What is the sum of the interior angles of a pentagon?",
    "If x/3 = 4, what is the value of x?",
    "Evaluate the expression: 2^3 * 3^2",
    "A car travels 60 miles in 1.5 hours. What is its average speed in miles per hour?",
    "What is the median of the set: 3, 7, 9, 15, 18?",
    "How many diagonals does a hexagon have?",
    "A bag contains 4 red balls, 5 blue balls, and 6 green balls. What is the probability of drawing a red ball?",
    "What is the area of a triangle with base 10 and height 6?",
    "If x + y = 10 and x - y = 4, what is the value of x?",
    "What is the value of the expression: (2 + 3)(4 + 1)?",
    "If 3x - 7 = 11, what is the value of x?",
    "What is the square root of 144?",
    "If f(x) = 2x + 3, what is f(5)?",
    "How many even numbers are there between 1 and 100?",
    "What is the least common multiple of 12 and 18?",
    "How many different 3-digit numbers can be formed using the digits 1, 2, 3 with repetition allowed?",
    "What is the smallest prime number greater than 50?",
    "If x^2 - 5x + 6 = 0, what are the solutions for x?",
    "What is the value of 100 - (5 + 3)^2?",
    "How many ways can you arrange the letters in the word 'MATH'?",
    "What is the slope of the line that passes through the points (1,2) and (3,6)?",
    "If 2x = 10 and 3y = 15, what is the value of x + y?",
    "How many prime numbers are less than 20?",
    "A square has a perimeter of 36 cm. What is its area?",
    "What is the cube root of 27?",
    "If x = 4, what is the value of x^3 + x^2 + x?",
    "Simplify: (x + 2)^2 - (x - 2)^2",
    "If sin(θ) = 1/2 and 0 < θ < 90°, what is the value of θ in degrees?",
    "What is the sum of the first 10 positive even integers?",
    "A circle has circumference 12π. What is its radius?",
    "How many distinct arrangements are there of the digits 1, 2, 3, 4?",
    "If the sum of three consecutive odd integers is 81, what is the smallest of the three?",
    "What is the value of 5! - 3!",
    "Find the greatest common divisor (GCD) of 84 and 120.",
    "What is the volume of a cube with edge length 3?",
    "Simplify: (3x + 4)^2",
    "What is the 10th term in the arithmetic sequence 4, 7, 10, 13, ...?",
    "If x = 2 and y = 3, evaluate x^2y + y^2x",
    "What is the solution to the equation: |2x - 5| = 7?",
    "How many sides does a regular polygon have if each interior angle is 150°?",
    "If a circle has diameter 14, what is its area in terms of π?",
    "How many positive integers less than 100 are divisible by both 2 and 3?",
    "What is the result when the sum of the first 20 natural numbers is divided by 10?",
    "If 5x + 3 = 2x + 12, what is the value of x?",
    "What is the average of the first 5 prime numbers?",
    "What is the difference between the square of 9 and the square of 5?",
    "A cube has a surface area of 54 cm². What is its volume?",
    "How many positive integers less than 50 are multiples of 3 or 5?",
    "If x^2 = 2x, what are the possible values of x?",
    "How many lines of symmetry does a regular octagon have?",
    "Evaluate: (1 + 2 + 3 + ... + 10)",
    "What is the 6th term in the Fibonacci sequence?",
    "If a triangle has sides 6, 8, and 10, what kind of triangle is it?",
    "What is the distance between the points (2, 3) and (7, 7)?",
    "What is the greatest three-digit number divisible by 7?",
    "How many different ways can you arrange the letters of the word 'LEVEL'?",
    "If a circle has area 25π, what is its circumference?",
    "What is the solution to 3x^2 - 12 = 0?",
    "What is the least number of coins needed to make 87 cents using only dimes and pennies?",
    "How many 2-digit numbers have a digit sum equal to 9?",
    "If the radius of a circle is doubled, by what factor does its area increase?",
    "Find the units digit of 7^2023.",
    "What is the sum of all odd integers between 1 and 99 inclusive?",
    "What is the value of log_2(32)?",
    "If the length of the hypotenuse of a right triangle is 13 and one leg is 5, what is the length of the other leg?",
    "How many numbers between 1 and 100 are perfect squares?",
    "If the volume of a cylinder is 100π and the height is 4, what is the radius?",
    "Find the number of positive integers less than 1000 that are divisible by neither 2 nor 5.",
    "What is the product of the roots of the equation x^2 - 6x + 5 = 0?",
    "Find the smallest positive integer x such that x ≡ 3 (mod 7) and x ≡ 2 (mod 5).",
    "What is the average of all positive integers less than 100 that are divisible by 9?",
    "If a sequence is defined by a₁ = 2 and aₙ = 3aₙ₋₁ + 1 for n ≥ 2, what is a₃?",
    "Find the value of x such that 2^x = 16.",
    "How many digits are there in 2^10?",
    "If 10% of x is equal to 25% of y, what is x in terms of y?",
    "What is the smallest prime number that is the sum of two other prime numbers?",
    "How many prime numbers are there between 50 and 100?",
    "Find the number of trailing zeros in 100!",
]

reasoning_prompts=reasoning_prompts[:10]
instruction_prompts=instruction_prompts[:10]

# === Get model output tokens ===
@torch.no_grad()
def generate_tokens(prompt, layer_to_ablate=None, max_new_tokens=30):
    # Clear cache before each generation
    torch.cuda.empty_cache()
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    hook = None
    if layer_to_ablate is not None:
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                return tuple(torch.zeros_like(o) for o in output)
            return torch.zeros_like(output)
        
        hook = model.model.layers[layer_to_ablate].register_forward_hook(ablation_hook)

    # Use the model's updated generation config (which we set to deterministic)
    # Just override max_new_tokens since we already set the model config
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            # No need to specify other params since we updated model.generation_config
        )

    if hook: 
        hook.remove()

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded

# === Get sentence embedding ===
@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    
    # Use mean pooling instead of last token for more stable embeddings
    hidden_states = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_embeddings = hidden_states * attention_mask
    embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
    
    return embeddings.squeeze().float()

# === Cosine similarity between outputs ===
def cosine_similarity(a, b):
    # Normalize vectors to avoid numerical issues
    a_norm = torch.nn.functional.normalize(a, p=2, dim=0)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=0)
    return torch.nn.functional.cosine_similarity(a_norm, b_norm, dim=0).item()

# === Layer-wise Analysis with multiple runs for stability ===
def analyze_layer_behavior(prompt_set, label, max_new_tokens=30, num_runs=3):
    num_layers = len(model.model.layers)
    impact_scores = []

    for layer in tqdm(range(num_layers), desc=f"{label} Layer Ablation"):
        run_scores = []
        
        # Multiple runs for each layer to get more stable results
        for run in range(num_runs):
            set_seed(42 + run)  # Different seed for each run
            delta_total = 0
            
            for prompt in prompt_set:
                # Generate original and ablated outputs
                original = generate_tokens(prompt, None, max_new_tokens)
                ablated = generate_tokens(prompt, layer_to_ablate=layer, max_new_tokens=max_new_tokens)

                # Get embeddings
                orig_emb = get_embedding(original)
                ablt_emb = get_embedding(ablated)
                
                # Calculate dissimilarity
                similarity = cosine_similarity(orig_emb, ablt_emb)
                delta = 1 - similarity
                delta_total += delta

            avg_delta = delta_total / len(prompt_set)
            run_scores.append(avg_delta)
            
            # Clean up after each run
            gc.collect()
            torch.cuda.empty_cache()

        # Take median across runs for stability
        final_score = np.median(run_scores)
        impact_scores.append(final_score)
        
        print(f"Layer {layer}: {final_score:.4f} (std: {np.std(run_scores):.4f})")

    return impact_scores

# === Main execution ===
print("Starting analysis...")
print("Analyzing instruction prompts...")
instruction_scores = analyze_layer_behavior(instruction_prompts, "Instruction", max_new_tokens=30)

print("\nAnalyzing reasoning prompts...")
reasoning_scores = analyze_layer_behavior(reasoning_prompts, "Reasoning", max_new_tokens=30)

# === Create output table ===
layer_ids = list(range(len(instruction_scores)))
df = pd.DataFrame({
    "Layer": layer_ids,
    "Instruction_Impact": instruction_scores,
    "Reasoning_Impact": reasoning_scores,
    "Δ (Reason - Instr)": [r - i for r, i in zip(reasoning_scores, instruction_scores)],
    "Layer_Nature": [
        "Reasoning" if r - i > 0.02 else "Instruction" if i - r > 0.02 else "Neutral"
        for i, r in zip(instruction_scores, reasoning_scores)
    ]
})

# === Format and show ===
df["Instruction_Impact"] = df["Instruction_Impact"].round(4)
df["Reasoning_Impact"] = df["Reasoning_Impact"].round(4)
df["Δ (Reason - Instr)"] = df["Δ (Reason - Instr)"].round(4)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(df.to_string(index=False))

# === Save to files ===
df.to_csv("layer_behavior_analysis.csv", index=False)
with open("layer_behavior_table.txt", "w") as f:
    f.write("Qwen 3 1.7B Layer Analysis Results\n")
    f.write("="*50 + "\n\n")
    f.write(df.to_string(index=False))
    f.write("\n\nAnalysis Summary:\n")
    f.write(f"Total layers: {len(instruction_scores)}\n")
    f.write(f"Reasoning-dominant layers: {sum(1 for x in df['Layer_Nature'] if x == 'Reasoning')}\n")
    f.write(f"Instruction-dominant layers: {sum(1 for x in df['Layer_Nature'] if x == 'Instruction')}\n")
    f.write(f"Neutral layers: {sum(1 for x in df['Layer_Nature'] if x == 'Neutral')}\n")

print(f"\nResults saved to 'layer_behavior_analysis.csv' and 'layer_behavior_table.txt'")
