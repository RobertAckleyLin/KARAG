import cv2
import base64
from openai import OpenAI

def draw_chessboard(image_path, output_path, grid_size=10, alpha=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print("read image failed")
        return
    
    h, w = img.shape[:2]
    
    cell_width = w // grid_size
    cell_height = h // grid_size

    overlay = img.copy()
    for i in range(1, grid_size+1):
        y = (i-1) * cell_height
        cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(img, str(i), (1, y + cell_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    for i in range(1, grid_size+1):
        x = (i-1) * cell_width
        cv2.line(overlay, (x, 0), (x, h), (0, 255, 0), 1)
        letter = chr(ord('a') + i - 1)
        cv2.putText(img, letter, (x + cell_width//2 + 2, cell_height//2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.imwrite(output_path, img)
    print(f"save to {output_path}")


task_instruction = ""
path_to_image = ''
path_to_processed_image = ''

draw_chessboard(path_to_image, path_to_processed_image, grid_size=8, alpha=0.5)

with open('prompt/visual_prompt.txt', 'r', encoding='utf-8') as f:
    prompt_template = f.read()
prompt_text = prompt_template.format(task_instruction)
print(prompt_text)

client = OpenAI()
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Getting the Base64 string
base64_image = encode_image(path_to_processed_image)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(response.choices[0])
