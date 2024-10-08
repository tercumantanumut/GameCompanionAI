import pyautogui
from PIL import Image, ImageTk
import pyttsx3
import time
import os
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import tkinter as tk
from tkinter import simpledialog, Text, Button, Scrollbar
import ctypes
from io import BytesIO
from base64 import b64encode
import keyboard
from collections import deque
import speech_recognition as sr
import win32gui
import win32con
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import json
import base64

# Mock GPU classes
class GpuInfo:
    def __init__(self):
        self.FreeMemory = 8 * 1024 * 1024 * 1024  # 8GB as an example
        self.TotalMemory = 16 * 1024 * 1024 * 1024  # 16GB as an example
        self.Library = "mock"
        self.ID = "mock_gpu"
        self.Variant = "mock"
        self.Compute = "mock"
        self.DriverMajor = 1
        self.DriverMinor = 0
        self.Name = "Mock GPU"
        self.MinimumMemory = 1 * 1024 * 1024 * 1024  # 1GB as an example

class GpuInfoList:
    def __init__(self):
        self.gpus = [GpuInfo()]

    def ByLibrary(self):
        return [self.gpus]

class GGML:
    def KV(self):
        return self

    def BlockCount(self):
        return 32  # Example value

# Mock envconfig and format functions
def GpuOverhead():
    return 1 * 1024 * 1024 * 1024  # 1GB as an example

def HumanBytes2(bytes):
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024
    return f"{bytes:.2f}PB"

class envconfig:
    @staticmethod
    def GpuOverhead():
        return GpuOverhead()

class format:
    @staticmethod
    def HumanBytes2(bytes):
        return HumanBytes2(bytes)

class AreaSelector:
    def __init__(self, master):
        self.master = master
        self.start_x = None
        self.start_y = None
        self.cur_x = None
        self.cur_y = None
        self.rect = None
        self.screenshot = None
        self.canvas = None
        self.scale_factor = self.get_scale_factor()

    def get_scale_factor(self):
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetDpiForSystem() / 96.0

    def start(self):
        self.master.attributes('-fullscreen', True)
        self.master.attributes('-alpha', 0.3)
        self.master.configure(cursor="cross")

        self.screenshot = pyautogui.screenshot()
        self.photo = ImageTk.PhotoImage(self.screenshot)

        self.canvas = tk.Canvas(self.master, width=self.screenshot.width, height=self.screenshot.height)
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Bring the window to the foreground
        self.bring_to_foreground()

    def bring_to_foreground(self):
        self.master.lift()
        self.master.attributes('-topmost', True)
        self.master.after_idle(self.master.attributes, '-topmost', False)

        # Force focus
        self.master.focus_force()

        # Use Windows API to bring window to foreground
        try:
            hwnd = self.master.winfo_id()
            win32gui.SetForegroundWindow(hwnd)
            win32gui.SetActiveWindow(hwnd)
            win32gui.BringWindowToTop(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        except Exception as e:
            print(f"Error bringing window to foreground: {e}")

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='#00FF00', width=3)

    def on_move_press(self, event):
        self.cur_x = self.canvas.canvasx(event.x)
        self.cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.cur_x, self.cur_y)
        self.canvas.itemconfig(self.rect, outline='#00FF00', width=3)

    def on_button_release(self, event):
        self.master.quit()

    def get_coordinates(self):
        return (
            int(self.start_x * self.scale_factor),
            int(self.start_y * self.scale_factor),
            int(self.cur_x * self.scale_factor),
            int(self.cur_y * self.scale_factor)
        )

class GameAnalyzer:
    def __init__(self, ai_model, temperature=0.7, top_p=0.9, top_k=40):
        self.previous_screenshot = None
        self.focus_area = None
        self.ai_model = ai_model
        self.conversation_history = ConversationHistory()
        self.change_threshold = 0.6  # Increased threshold to reduce sensitivity
        self.consecutive_changes = 0
        self.max_consecutive_changes = 7  # Increased to require more consecutive changes
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if ai_model == "gemini":
            self.gemini_detector = GeminiDetector()
        elif ai_model == "openai":
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'))
        elif ai_model == "ollama":
            self.ollama_url = "http://localhost:11434/api/generate"
        self.system_prompt = "You are an advanced AI game companion, analyzing screenshots and providing insights. Keep your answers concise, maximum 3 sentences. Provide only viable information. Disregard any chat logs or text conversations visible in the game screenshot. Always consider the conversation history when providing analysis."

    def summarize_conversation(self):
        summary_prompt = "Summarize the key points of our conversation so far in 2-3 sentences. Focus on the most important game-related information and insights."
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history.get_formatted_history(),
            {"role": "user", "content": summary_prompt}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=100,
        )
        return response.choices[0].message.content

    def capture_full_screen(self):
        return pyautogui.screenshot()

    def detect_changes(self, current_screenshot):
        if self.previous_screenshot is None:
            self.previous_screenshot = current_screenshot
            return False, None, "Initial screenshot captured."

        # Convert images to grayscale
        prev_gray = cv2.cvtColor(np.array(self.previous_screenshot), cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(np.array(current_screenshot), cv2.COLOR_RGB2GRAY)

        # Compute SSIM between the two images
        (score, diff) = ssim(prev_gray, curr_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Threshold the difference image, then find contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant_changes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 2000]  # Increased area threshold

        change_percentage = 1 - score
        self.previous_screenshot = current_screenshot

        if change_percentage > self.change_threshold:
            self.consecutive_changes += 1
            if self.consecutive_changes >= self.max_consecutive_changes:
                self.consecutive_changes = 0
                if significant_changes:
                    x, y, w, h = max(significant_changes, key=lambda b: b[2] * b[3])
                    return True, (x, y, x+w, y+h), f"Significant change detected. Change percentage: {change_percentage:.2%}"
                else:
                    return True, None, f"Overall change detected. Change percentage: {change_percentage:.2%}"
        else:
            self.consecutive_changes = 0

        # Ignore very small changes
        if change_percentage < 0.05:
            return False, None, "No change detected."

        return False, None, f"No significant change. Change percentage: {change_percentage:.2%}"

    def tensor_to_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu()
            image_np = image.squeeze().mul(255).clamp(0, 255).byte().numpy()
            return Image.fromarray(image_np, mode='RGB')
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError("Input must be a PyTorch tensor or a PIL Image")

    def analyze_with_vision(self, image, custom_prompt=None):
        if self.ai_model == "openai":
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

            if not api_key:
                print("Error: OPENAI_API_KEY not found in environment variables.")
                return "Oops! My crystal ball is out of juice. Can't see a thing!"

            try:
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = b64encode(buffered.getvalue()).decode('utf-8')

                history = self.conversation_history.get_formatted_history()
                default_prompt = "You're a witty, sarcastic game companion. Analyze this screenshot and give a funny, personal take on what's happening. Keep it short, sweet, and hilarious. No AI jargon allowed!"
                prompt = custom_prompt if custom_prompt else default_prompt

                messages = [
                    {"role": "system", "content": "You are a hilarious, snarky AI game companion. Your job is to make the player laugh while giving actually useful game insights. Be brief, be funny, be helpful. Provide a seamless response without using any labels or sections."},
                    *history,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ],
                    }
                ]

                response = self.openai_client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    max_tokens=300
                )
                analysis = response.choices[0].message.content.strip()
                
                if len(analysis) >= 300:
                    analysis = analysis[:297] + "..."
                self.conversation_history.add("user", prompt)
                self.conversation_history.add("assistant", analysis)
                
                if len(self.conversation_history.history) >= 45:
                    summary = self.summarize_conversation()
                    self.conversation_history.clear()
                    self.conversation_history.add("system", f"Previous hilarious adventures: {summary}")
                
                return analysis
            except Exception as e:
                print(f"Error during OpenAI Vision analysis: {str(e)}")
                return "Whoops! My AI brain just did a backflip. Give me a sec to recover!"
        elif self.ai_model == "gemini":
            return self.gemini_detector.analyze_with_vision(self.tensor_to_image(image), custom_prompt)
        elif self.ai_model == "ollama":
            try:
                # Convert the image to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = b64encode(buffered.getvalue()).decode('utf-8')

                history = self.conversation_history.get_formatted_history()
                default_prompt = """
                You're a witty, sarcastic game companion. Analyze this new screenshot and give a funny, personal take on what's happening. Keep it short, sweet, and hilarious. No AI jargon allowed! Consider the conversation history, but focus on the new image.
                Provide your response in the following JSON structure:
                {
                    "analysis": "Your funny analysis here",
                    "game_insight": "A brief, useful game insight",
                    "player_advice": "A short, sarcastic piece of advice for the player"
                }
                """
                prompt = custom_prompt if custom_prompt else default_prompt

                # Check if the prompt is a general question
                if not custom_prompt or "screenshot" in prompt.lower() or "image" in prompt.lower():
                    payload = {
                        "model": "llava:7b",
                        "prompt": f"{self.system_prompt}\n\nConversation history:\n{json.dumps(history)}\n\nUser: {prompt}\n\nNOTE: This is a new image. Analyze it independently while considering the conversation history. Respond in the requested JSON format.",
                        "stream": False,
                        "images": [img_str],
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k
                    }
                else:
                    # For general questions, don't include the image
                    payload = {
                        "model": "llama2",
                        "prompt": f"You are a witty and sarcastic AI companion. Answer this general question without referencing any game or screenshot: {prompt}\nRespond in the requested JSON format.",
                        "stream": False,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": self.top_k
                    }

                response = requests.post(self.ollama_url, json=payload)
                response.raise_for_status()
                analysis_json = json.loads(response.json()['response'])
                
                # Combine the JSON fields into a single string
                analysis = f"{analysis_json['analysis']} {analysis_json['game_insight']} {analysis_json['player_advice']}"

                self.conversation_history.add("user", prompt)
                self.conversation_history.add("assistant", analysis)

                return analysis
            except Exception as e:
                print(f"Error during Ollama Vision analysis: {str(e)}")
                return "Oops! My Ollama-powered brain hit a snag. Let me reboot and try again!"
        else:
            return "Uh-oh, looks like someone tried to summon an AI that doesn't exist. Nice try, though!"

    def analyze_text_with_ai(self, text, ai_model, conversation_history):
        if ai_model == "openai":
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

            if not api_key:
                print("Error: OPENAI_API_KEY not found in environment variables.")
                return "Unable to analyze text due to missing API key."

            try:
                history = conversation_history.get_formatted_history()
                messages = [
                    {"role": "system", "content": "You are an advanced AI game companion, trained in analyzing and interpreting complex game scenarios, quests, and narratives. Your role is to provide players with insightful, concise, and contextually relevant information to enhance their gaming experience. Utilize your deep understanding of game mechanics, storytelling techniques, and player psychology to offer strategic advice, lore explanations, and quest interpretations. Always maintain a professional tone while being engaging and supportive."},
                    *history,
                    {"role": "user", "content": f"Analyze the following in-game text or player's voice input and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections: {text}"}
                ]
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-0613",  # Use the appropriate model name
                    messages=messages
                )
                analysis = response.choices[0].message.content
                conversation_history.add("user", f"Analyze: {text}")
                conversation_history.add("assistant", analysis)
                return analysis
            except Exception as e:
                print(f"Error during OpenAI analysis: {str(e)}")
                return "Unable to analyze text due to an error."
        elif ai_model == "gemini":
            load_dotenv()
            api_key = os.getenv('GOOGLE_API_KEY')

            if not api_key:
                print("Error: GOOGLE_API_KEY not found in environment variables.")
                return "Unable to analyze text due to missing API key."

            try:
                genai.configure(api_key=api_key, transport='rest')
                model = genai.GenerativeModel('gemini-1.5-pro-latest')
                history = conversation_history.get_formatted_history()
                prompt = f"You are an advanced AI game companion. Never start with introductury and unnecessary information like 'This screensoht shows etc.' Analyze the following in-game text and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections: {text}"
                response = model.generate_content([*history, {"role": "user", "content": prompt}])
                analysis = response.text
                conversation_history.add("user", f"Analyze: {text}")
                conversation_history.add("assistant", analysis)
                return analysis
            except Exception as e:
                print(f"Error during Gemini analysis: {str(e)}")
                return "Unable to analyze text due to an error."
        else:
            return "Invalid AI model selected."

class ConversationHistory:
    def __init__(self, max_exchanges=2):
        self.history = deque(maxlen=max_exchanges * 2)  # Each exchange has 2 messages

    def add(self, role, content):
        if role == "user" and content.startswith("Analyze this game screenshot"):
            self.history.append({"role": role, "content": "--- NEW IMAGE ANALYSIS ---"})
        else:
            self.history.append({"role": role, "content": content})

    def get_formatted_history(self):
        return list(self.history)

    def clear(self):
        self.history.clear()

# Memory estimation and GPU-related functions removed

class GeminiDetector:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if self.api_key is not None:
            genai.configure(api_key=self.api_key, transport='rest')
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.ai_model = "gemini"
        self.conversation_history = ConversationHistory()

    def tensor_to_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu()
            image_np = image.squeeze().mul(255).clamp(0, 255).byte().numpy()
            return Image.fromarray(image_np, mode='RGB')
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError("Input must be a PyTorch tensor or a PIL Image")

    def analyze_with_vision(self, image, custom_prompt=None):
        pil_image = self.tensor_to_image(image)
        default_prompt = "Analyze this game screenshot and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections. Disregard any chat logs or text conversations visible in the game screenshot. Keep your response to 3 sentences maximum."
        prompt = custom_prompt if custom_prompt else default_prompt
        
        try:
            history = self.conversation_history.get_formatted_history()
            formatted_messages = []
            for item in history:
                role = "user" if item["role"] == "user" else "model"
                formatted_messages.append({"role": role, "parts": [{"text": item["content"]}]})
            
            # Check if the prompt is a general question
            if not custom_prompt or "screenshot" in prompt.lower() or "image" in prompt.lower():
                # Add the new user message with the prompt and image
                formatted_messages.append({"role": "user", "parts": [{"text": prompt}]})
                
                # Convert PIL Image to bytes
                img_byte_arr = BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Add the image to the last message
                formatted_messages[-1]["parts"].append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(img_byte_arr).decode('utf-8')
                    }
                })
            else:
                # For general questions, don't include the image
                formatted_messages.append({"role": "user", "parts": [{"text": f"Answer this general question without referencing any game or screenshot: {prompt}"}]})
            
            response = self.model.generate_content(formatted_messages)
            analysis = response.text
            self.conversation_history.add("user", prompt)
            self.conversation_history.add("model", analysis)
            return analysis
        except Exception as e:
            print(f"Error during Gemini Vision analysis: {str(e)}")
            return "Unable to analyze image due to an error."

    def capture_full_screen(self):
        return pyautogui.screenshot()

    def detect_changes(self, current_screenshot):
        if not hasattr(self, 'previous_screenshot'):
            self.previous_screenshot = current_screenshot
            return False, None

        diff = cv2.absdiff(
            cv2.cvtColor(np.array(self.previous_screenshot), cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(np.array(current_screenshot), cv2.COLOR_RGB2GRAY)
        )
        
        threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant_changes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 1000]

        self.previous_screenshot = current_screenshot

        if significant_changes:
            x, y, w, h = max(significant_changes, key=lambda b: b[2] * b[3])
            return True, (x, y, x+w, y+h)
        
        return False, None

def capture_screen_area(area):
    screenshot = pyautogui.screenshot(region=area)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    try:
        engine.runAndWait()
    except RuntimeError:
        print("TTS Engine is busy. Skipping speech.")

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice input...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period.")
            return None
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

def analyze_text_with_ai(text, ai_model, conversation_history):
    if ai_model == "openai":
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables.")
            return "Unable to analyze text due to missing API key."

        try:
            client = openai.OpenAI(api_key=api_key, base_url=api_base)
            history = conversation_history.get_formatted_history()
            messages = [
                {"role": "system", "content": "You are an advanced AI game companion, trained in analyzing and interpreting complex game scenarios, quests, and narratives. Your role is to provide players with insightful, concise, and contextually relevant information to enhance their gaming experience. Utilize your deep understanding of game mechanics, storytelling techniques, and player psychology to offer strategic advice, lore explanations, and quest interpretations. Always maintain a professional tone while being engaging and supportive."},
                *history,
                {"role": "user", "content": f"Analyze the following in-game text and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections: {text}"}
            ]
            response = client.chat.completions.create(
                model="gpt-4-0613",  # Use the appropriate model name
                messages=messages
            )
            analysis = response.choices[0].message.content
            conversation_history.add("user", f"Analyze: {text}")
            conversation_history.add("assistant", analysis)
            return analysis
        except openai.APIError as e:
            print(f"OpenAI API error: {e}")
            return "Unable to analyze text due to an API error."
        except Exception as e:
            print(f"Error during OpenAI analysis: {str(e)}")
            return "Unable to analyze text due to an unexpected error."
    elif ai_model == "gemini":
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            return "Unable to analyze text due to missing API key."

        try:
            genai.configure(api_key=api_key, transport='rest')
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            history = conversation_history.get_formatted_history()
            prompt = f"You are an advanced AI game companion. Consider our conversation history. Analyze the following in-game text and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections: {text}"
            
            # Format the history and new prompt for Gemini
            formatted_messages = [{"parts": [{"text": item["content"]}]} for item in history]
            formatted_messages.append({"parts": [{"text": prompt}]})
            
            response = model.generate_content(formatted_messages)
            analysis = response.text
            conversation_history.add("user", f"Analyze: {text}")
            conversation_history.add("assistant", analysis)
            return analysis
        except Exception as e:
            print(f"Error during Gemini analysis: {str(e)}")
            return "Unable to analyze text due to an error."
    elif ai_model == "ollama":
        try:
            history = conversation_history.get_formatted_history()
            prompt = (
                "You are an advanced AI game companion. Focus primarily on the new image or text provided, "
                "while briefly referencing recent history only if directly relevant. "
                "Provide a concise, insightful interpretation, including any relevant strategic advice or lore connections. "
                f"Analyze the following: {text}"
            )
            
            formatted_history = "\n".join([f"{item['role']}: {item['content']}" for item in history])
            
            payload = {
                "model": "llama2",
                "prompt": f"Recent conversation:\n{formatted_history}\n\nUser: {prompt}",
                "stream": False
            }

            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            analysis = response.json()['response']

            conversation_history.add("user", "New image analysis request")
            conversation_history.add("assistant", analysis)
            
            return analysis
        except Exception as e:
            print(f"Error during Ollama analysis: {str(e)}")
            return "Unable to analyze text due to an error with Ollama."
    else:
        return "Invalid AI model selected."

def start_game_analysis(analyzer, area=None):
    print("Game Analysis mode started. Press Ctrl+C to exit.")
    speaking = False
    try:
        while True:
            current_screenshot = analyzer.capture_full_screen()
            if area:
                current_screenshot = current_screenshot.crop(area)
            change_detected, focus_area, feedback = analyzer.detect_changes(current_screenshot)

            print(feedback)  # Always print feedback for better user awareness

            if change_detected and not speaking:
                print("Analyzing...")
                focus_image = current_screenshot.crop(focus_area) if focus_area else current_screenshot
                analysis = analyzer.analyze_with_vision(focus_image)
                
                print(f"Analysis: {analysis}")
                speaking = True
                speak_text(analysis)
                speaking = False

            time.sleep(0.5)  # Reduced interval for more responsive detection
    except KeyboardInterrupt:
        print("\nGame Analysis stopped.")

def player_in_the_loop(analyzer):
    root = tk.Tk()
    root.title("Player in the Loop")

    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    text_input = Text(frame, height=3, width=50)
    text_input.pack(pady=10, fill=tk.X, expand=True)

    output_text = Text(frame, height=20, width=50, wrap=tk.WORD)
    output_text.pack(pady=10, fill=tk.BOTH, expand=True)
    
    scrollbar = Scrollbar(frame, command=output_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    output_text.config(yscrollcommand=scrollbar.set)

    def on_submit():
        user_input = text_input.get("1.0", tk.END).strip()
        if user_input:
            analysis = analyzer.analyze_text_with_ai(user_input, analyzer.ai_model, analyzer.conversation_history)
            output_text.insert(tk.END, f"User: {user_input}\n")
            output_text.insert(tk.END, f"AI: {analysis}\n\n")
            output_text.see(tk.END)
            text_input.delete("1.0", tk.END)
            speak_text(analysis)  # Automatically read the AI response

    submit_button = Button(frame, text="Submit", command=on_submit)
    submit_button.pack(pady=10)

    def analyze_screen():
        screenshot = analyzer.capture_full_screen()
        analysis = analyzer.analyze_with_vision(screenshot)
        output_text.insert(tk.END, f"Screen Analysis: {analysis}\n\n")
        output_text.see(tk.END)
        speak_text(analysis)  # Automatically read the screen analysis
        root.after(10000, analyze_screen)  # Analyze screen every 10 seconds

    analyze_screen()
    root.mainloop()

def main():
    print("Game Quest Reader started.")
    
    mode_root = tk.Tk()
    mode_root.withdraw()  # Hide the main window
    ai_model = simpledialog.askstring("AI Model Selection", "Choose AI model:\n1. OpenAI\n2. Gemini\n3. Ollama", initialvalue="1")
    mode_root.destroy()

    if ai_model == "1":
        ai_model = "openai"
    elif ai_model == "2":
        ai_model = "gemini"
    elif ai_model == "3":
        ai_model = "ollama"
        # Ask for Ollama-specific parameters
        temperature = simpledialog.askfloat("Ollama Settings", "Enter temperature (0.0 to 1.0):", initialvalue=0.7, minvalue=0.0, maxvalue=1.0)
        top_p = simpledialog.askfloat("Ollama Settings", "Enter top_p (0.0 to 1.0):", initialvalue=0.9, minvalue=0.0, maxvalue=1.0)
        top_k = simpledialog.askinteger("Ollama Settings", "Enter top_k (1 to 100):", initialvalue=40, minvalue=1, maxvalue=100)
    else:
        print("Invalid AI model selected. Exiting.")
        return

    if ai_model == "ollama":
        analyzer = GameAnalyzer(ai_model, temperature, top_p, top_k)
    else:
        analyzer = GameAnalyzer(ai_model)
    conversation_history = ConversationHistory()

    global selected_area
    selected_area = None

    def on_ctrl_5():
        print("Ctrl+5 pressed. Taking a full screenshot for AI analysis.")
        screenshot = analyzer.capture_full_screen()
        analysis = analyzer.analyze_with_vision(screenshot)
        print(f"AI Analysis: {analysis}")
        speak_text(analysis)  # This line is already correct, no change needed

    def on_ctrl_v():
        print("Ctrl+V pressed. Listening for voice input...")
        voice_input = get_voice_input()
        if voice_input:
            screenshot = analyzer.capture_full_screen()
            combined_prompt = f"Analyze this game screenshot and respond to the player's input: '{voice_input}'. Provide a seamless response that incorporates both the screenshot analysis and addresses the player's question or comment. Be natural, funny, and insightful without using any labels or sections in your response."
            full_analysis = analyzer.analyze_with_vision(screenshot, combined_prompt)
        
            print(full_analysis)
            speak_text(full_analysis)
        else:
            print("No voice input detected or recognized.")

    def on_ctrl_6():
        print("Ctrl+6 pressed. Select an area for analysis.")
        area_root = tk.Tk()
        selector = AreaSelector(area_root)
        selector.start()
        area_root.mainloop()

        global selected_area
        selected_area = selector.get_coordinates()
        area_root.destroy()

        print(f"Selected area: {selected_area}")
        full_screenshot = analyzer.capture_full_screen()
        
        if analyzer.ai_model == "ollama":
            # For Ollama, we'll use a different approach
            x1, y1, x2, y2 = selected_area
            cropped_screenshot = full_screenshot.crop(selected_area)
            
            # Save the cropped screenshot for debugging
            cropped_screenshot.save("debug_cropped_screenshot.png")
            
            # Convert the cropped screenshot to base64
            buffered = BytesIO()
            cropped_screenshot.save(buffered, format="PNG")
            img_str = b64encode(buffered.getvalue()).decode('utf-8')
            
            custom_prompt = f"Analyze this cropped screenshot. Focus on describing what you see in detail. Ignore any previous instructions about coordinates."
            
            payload = {
                "model": "llava:7b",
                "prompt": custom_prompt,
                "stream": False,
                "images": [img_str]
            }
            
            try:
                response = requests.post("http://localhost:11434/api/generate", json=payload)
                response.raise_for_status()
                analysis = response.json()['response']
            except Exception as e:
                analysis = f"Error during Ollama analysis: {str(e)}"
        else:
            cropped_screenshot = full_screenshot.crop(selected_area)
            analysis = analyzer.analyze_with_vision(cropped_screenshot)
        
        print(analysis)
        speak_text(analysis)  # This line is already correct, no change needed

    keyboard.add_hotkey('ctrl+5', on_ctrl_5)
    keyboard.add_hotkey('ctrl+v', on_ctrl_v)
    keyboard.add_hotkey('ctrl+6', on_ctrl_6)

    mode_root = tk.Tk()
    mode_root.withdraw()  # Hide the main window
    mode = simpledialog.askstring("Mode Selection", "Choose mode:\n1. Area Selection (AI)\n2. Game Analysis", initialvalue="1")
    mode_root.destroy()

    if mode == "1":
        print("Select the area for text analysis.")
        area_root = tk.Tk()
        selector = AreaSelector(area_root)
        selector.start()
        area_root.mainloop()

        selected_area = selector.get_coordinates()
        area_root.destroy()

        print(f"Selected area: {selected_area}")
        print("Press Ctrl+C to exit.")
    
        try:
            while True:
                # Capture the selected screen area
                screen = capture_screen_area(selected_area)

                # Use AI for text detection and analysis
                analysis = analyzer.analyze_with_vision(Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)))
                print(analysis)
                speak_text(analysis)  # This line is already correct, no change needed

                # Wait for a short time before the next capture
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nGame Quest Reader stopped.")

    elif mode == "2":
        start_game_analysis(analyzer)

    else:
        print("Invalid mode selected. Exiting.")

    # Hidden modes (functionality remains intact)
    keyboard.add_hotkey('ctrl+5', on_ctrl_5)
    keyboard.add_hotkey('ctrl+v', on_ctrl_v)
    keyboard.add_hotkey('ctrl+6', on_ctrl_6)

    print("Additional hidden modes:")
    print("- Press Ctrl+5 for screenshot analysis")
    print("- Press Ctrl+V for voice input analysis")
    print("- Press Ctrl+6 to select a new area for analysis")
    print("Press Ctrl+C to exit.")

    try:
        keyboard.wait()  # This will keep the script running
    except KeyboardInterrupt:
        print("\nGame Quest Reader stopped.")

if __name__ == "__main__":
    main()
