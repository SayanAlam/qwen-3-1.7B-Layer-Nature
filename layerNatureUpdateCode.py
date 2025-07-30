import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the model and tokenizer
# Make sure to specify the correct path to your downloaded model
model_name = "Qwen/Qwen1.5-1.8B"  # Adjust this to your specific model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}. Please ensure the model path is correct and dependencies are installed.")
    exit()

# If you have a GPU, move the model to it
if torch.cuda.is_available():
    model.to("cuda")
    print("Model moved to GPU.")
else:
    print("No GPU available. Running on CPU, which may be slow.")

# 2. Define the hook function and a helper to manage hooks
def get_activation_hook(activations_list):
    """
    Returns a hook function that saves the output of a layer to a list.
    """
    def hook_fn(module, input, output):
        # We save the output of the layer. Detach it from the graph.
        if isinstance(output, tuple):
            activations_list.append(output[0].detach().cpu())
        else:
            activations_list.append(output.detach().cpu())
    return hook_fn

def apply_hooks(model, activations_list):
    """
    Applies hooks to all transformer layers and returns a list of the hook handles.
    """
    hooks = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        hook = layer.register_forward_hook(get_activation_hook(activations_list))
        hooks.append(hook)
    return hooks

def remove_hooks(hooks):
    """
    Removes all the hooks in a list.
    """
    for hook in hooks:
        hook.remove()

# 3. Define the prompts
prompt_instruction = [
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

prompt_reasoning = [
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

prompt_reasoning=prompt_reasoning[:5]
prompt_instruction=prompt_instruction[:5]

# Use separate lists to store the activations for each task
activations_list_reasoning = []
activations_list_instruction = []

# Set model to evaluation mode
model.eval()

# 4. Run inference for each task and collect activations
with torch.no_grad():
    
    # --- Task 1: Reasoning ---
    print("\n--- Running Reasoning Task ---")
    
    # Apply hooks for the reasoning task
    reasoning_hooks = apply_hooks(model, activations_list_reasoning)
    
    inputs_reasoning = tokenizer(prompt_reasoning, return_tensors="pt").to(model.device)
    _ = model(**inputs_reasoning)
    
    # Remove the hooks to clean up
    remove_hooks(reasoning_hooks)
    
    # --- Task 2: Instruction-Following ---
    print("--- Running Instruction-Following Task ---")
    
    # Apply hooks for the instruction-following task
    instruction_hooks = apply_hooks(model, activations_list_instruction)
    
    inputs_instruction = tokenizer(prompt_instruction, return_tensors="pt").to(model.device)
    _ = model(**inputs_instruction)
    
    # Remove the hooks
    remove_hooks(instruction_hooks)

# 5. Convert lists of tensors to NumPy arrays
# We need to stack the tensors along a new dimension (the layer dimension)
try:
    activations_reasoning = torch.stack(activations_list_reasoning).numpy()
    activations_instruction = torch.stack(activations_list_instruction).numpy()
    
    print("\n--- Data successfully converted to NumPy arrays. ---")
    print("Shape of reasoning activations array:", activations_reasoning.shape)
    print("Shape of instruction activations array:", activations_instruction.shape)
    
except Exception as e:
    print(f"\nError converting to NumPy arrays: {e}")
    print("This might happen if the tokenization results in different sequence lengths.")
    print("You might need to pad the sequences or adjust your analysis.")
    # For now, we'll just proceed with the original lists for the analysis.
    activations_reasoning = activations_list_reasoning
    activations_instruction = activations_list_instruction

# 6. Analysis and Visualization (using the NumPy arrays)
num_layers = len(model.model.layers)
layer_indices = range(num_layers)

# Calculate L2 norms for each task using the numpy arrays
# We'll take the norm of the last token's activation for simplicity,
# or you could take the mean norm across all tokens.
norms_reasoning = [np.linalg.norm(activations_reasoning[i, -1, :]) for i in layer_indices]
norms_instruction = [np.linalg.norm(activations_instruction[i, -1, :]) for i in layer_indices]

# Calculate Cosine Similarities
cosine_similarities = []
# Ensure the dimensions match before computing similarity
if activations_reasoning.shape == activations_instruction.shape:
    for i in layer_indices:
        # Get the activation vectors for the last token of each sequence
        vec_reasoning = activations_reasoning[i, -1, :]
        vec_instruction = activations_instruction[i, -1, :]
        
        # Compute cosine similarity using numpy's dot product
        dot_product = np.dot(vec_reasoning, vec_instruction)
        norm_reasoning = np.linalg.norm(vec_reasoning)
        norm_instruction = np.linalg.norm(vec_instruction)
        
        if norm_reasoning > 0 and norm_instruction > 0:
            cos_sim = dot_product / (norm_reasoning * norm_instruction)
            cosine_similarities.append(cos_sim)
        else:
            cosine_similarities.append(0.0)
else:
    print("\nCannot compute cosine similarity directly due to mismatched array shapes.")
    print("Reasoning shape:", activations_reasoning.shape)
    print("Instruction shape:", activations_instruction.shape)
    cosine_similarities = [0] * num_layers

# 7. Visualization
plt.figure(figsize=(12, 6))
plt.plot(layer_indices, norms_reasoning, label="Reasoning Task (L2 Norm)", marker='o')
plt.plot(layer_indices, norms_instruction, label="Instruction-Following Task (L2 Norm)", marker='x')
plt.title("Layer Activation Norms for Reasoning vs. Instruction-Following Tasks")
plt.xlabel("Layer Index")
plt.ylabel("L2 Norm of Activations")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(layer_indices, cosine_similarities, label="Cosine Similarity", marker='o', color='green')
plt.title("Cosine Similarity Between Reasoning and Instruction-Following Activations")
plt.xlabel("Layer Index")
plt.ylabel("Cosine Similarity")
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Analysis Summary ---")
print(f"Number of layers analyzed: {num_layers}")
print(f"Reasoning Task Prompt: {prompt_reasoning}")
print(f"Instruction-Following Task Prompt: {prompt_instruction}")
