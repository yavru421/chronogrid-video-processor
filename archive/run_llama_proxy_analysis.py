
 # strict_mode removed
import requests
import base64
import json
from pathlib import Path
from PIL import Image
import io

import sys
import requests
import base64
import json
from pathlib import Path
from PIL import Image
import io

def analyze_image(img_path):
    proxy_url = "https://llama-universal-netlify-project.netlify.app/.netlify/functions/llama-proxy?path=/chat/completions"
    print(f"\n--- Testing {img_path.name} ---")
    try:
        img = Image.open(img_path)
        width, height = img.size
        if width > 2000 or height > 2000:
            print(f"Image too large ({width}x{height}), splitting into 4 quadrants for analysis.")
            quadrant_width = width // 2
            quadrant_height = height // 2
            quadrants = [
                img.crop((0, 0, quadrant_width, quadrant_height)),  # top-left
                img.crop((quadrant_width, 0, width, quadrant_height)),  # top-right
                img.crop((0, quadrant_height, quadrant_width, height)),  # bottom-left
                img.crop((quadrant_width, quadrant_height, width, height)),  # bottom-right
            ]
            analyses = []
            for i, quad in enumerate(quadrants):
                buffer = io.BytesIO()
                quad.save(buffer, format='JPEG')
                quad_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                payload = {
                    "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"You are analyzing quadrant {i+1} of a split chronogrid image. Describe ONLY what is visually present in this quadrant. "
                                        "Identify the primary subject(s), their actions, tools, animals, and notable objects. "
                                        "Mention the environment, lighting, and any progression or sequence that can be inferred. "
                                        "Avoid speculation, metaphors, or emotional interpretation—stick to concrete facts. "
                                        "End with a concise timeline-style bullet list summarizing the sequence in this quadrant."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{quad_b64}"},
                                },
                            ],
                        }
                    ],
                    "max_completion_tokens": 800,
                    "temperature": 0.2,
                }
                resp = requests.post(proxy_url, json=payload, timeout=120)
                resp.raise_for_status()
                quad_data = resp.json()
                analyses.append(quad_data)
                print(f"Quadrant {i+1} analyzed successfully.")
            combined_text = f"Combined analysis of {img_path.name} (split into 4 quadrants):\n\n"
            for i, analysis in enumerate(analyses):
                content = analysis.get("completion_message", {}).get("content", {}).get("text", "")
                combined_text += f"Quadrant {i+1}:\n{content}\n\n"
            print(combined_text)
            print(f"\nSUCCESS: Endpoint responded for {img_path.name} (split analysis)")
        else:
            with img_path.open("rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            payload = {
                "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are analyzing a chronogrid image. Describe ONLY what is visually present. "
                                    "Identify the primary subject(s), their actions, tools, animals, and notable objects. "
                                    "Mention the environment, lighting, and any progression or sequence that can be inferred from the image. "
                                    "Avoid speculation, metaphors, or emotional interpretation—stick to concrete facts. "
                                    "End with a concise timeline-style bullet list summarizing the sequence in order."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                            },
                        ],
                    },
                ],
                "max_completion_tokens": 800,
                "temperature": 0.2,
            }
            resp = requests.post(proxy_url, json=payload, timeout=120)
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2))
            print("\nSUCCESS: Endpoint responded for", img_path.name)
    except Exception as exc:
        import traceback
        print(f"\nERROR for {img_path.name}: {type(exc).__name__} - {exc}")
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if not img_path.exists():
            print(f"Image file not found: {img_path}")
            sys.exit(1)
        analyze_image(img_path)
    else:
        # Fallback: scan for .jpg images in current directory
        image_dir = Path(__file__).parent
        image_files = sorted(image_dir.glob("*.jpg"))
        if not image_files:
            print("No .jpg images found in directory.")
            sys.exit(1)
        for img_path in image_files:
            analyze_image(img_path)