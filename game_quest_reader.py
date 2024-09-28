import cv2
import numpy as np
import pyautogui
import pytesseract
from PIL import Image, ImageTk
import pyttsx3
import time
import os
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox, simpledialog, Text, Button, Scrollbar
import ctypes
from io import BytesIO
from base64 import b64encode
import keyboard
import json
import torch
from collections import deque
from skimage.metrics import structural_similarity as ssim

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

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_move_press(self, event):
        self.cur_x = self.canvas.canvasx(event.x)
        self.cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.cur_x, self.cur_y)

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
    def __init__(self, ai_model):
        self.previous_screenshot = None
        self.focus_area = None
        self.ai_model = ai_model
        self.conversation_history = ConversationHistory()
        self.change_threshold = 0.5  # Increased threshold to reduce sensitivity
        self.consecutive_changes = 0
        self.max_consecutive_changes = 5  # Increased to require more consecutive changes
        if ai_model == "gemini":
            self.gemini_detector = GeminiDetector()
        elif ai_model == "openai":
            self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'))

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

    def analyze_with_vision(self, image):
        if self.ai_model == "openai":
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

            if not api_key:
                print("Error: OPENAI_API_KEY not found in environment variables.")
                return "Unable to analyze image due to missing API key."

            try:
                # Convert the image to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = b64encode(buffered.getvalue()).decode('utf-8')

                history = self.conversation_history.get_formatted_history()
                messages = [
                    {"role": "system", "content": "You are a game companion analyzing screenshots. Keep your answers to a maximum of 3 sentences. Provide only viable information. Disregard any chat logs or text conversations visible in the game screenshot."},
                    *history,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this game screenshot, considering our conversation history. Ignore any in-game chat logs or text conversations."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ],
                    }
                ]

                response = self.openai_client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    max_tokens=300,
                )
                analysis = response.choices[0].message.content
                self.conversation_history.add("user", "Analyze this game screenshot.")
                self.conversation_history.add("assistant", analysis)
                return analysis
            except Exception as e:
                print(f"Error during OpenAI Vision analysis: {str(e)}")
                return "Unable to analyze image due to an error."
        elif self.ai_model == "gemini":
            return self.gemini_detector.analyze_with_vision(self.tensor_to_image(image))
        else:
            return "Invalid AI model selected."

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
                    {"role": "user", "content": f"Analyze the following in-game text and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections: {text}"}
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
                prompt = f"You are an advanced AI game companion. Consider our conversation history. Analyze the following in-game text and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections: {text}"
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
    def __init__(self, max_length=10):
        self.history = deque(maxlen=max_length)

    def add(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_formatted_history(self):
        return list(self.history)

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

    def analyze_with_vision(self, image):
        pil_image = self.tensor_to_image(image)
        prompt = "Analyze this game screenshot and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections. Disregard any chat logs or text conversations visible in the game screenshot. Keep your response to 3 sentences maximum. Consider the conversation history for context."
        
        try:
            history = self.conversation_history.get_formatted_history()
            response = self.model.generate_content([*history, {"role": "user", "content": prompt}, pil_image])
            self.conversation_history.add("user", prompt)
            self.conversation_history.add("assistant", response.text)
            return response.text
        except Exception as e:
            print(f"Error during Gemini Vision analysis: {str(e)}")
            return "Unable to analyze image due to an error."

    def capture_full_screen(self):
        return pyautogui.screenshot()

    def detect_changes(self, current_screenshot):
        if self.previous_screenshot is None:
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

    def analyze_with_vision(self, image):
        pil_image = self.tensor_to_image(image)
        prompt = "Analyze this game screenshot and provide a concise, insightful interpretation, including any relevant strategic advice or lore connections. Keep your response to 3 sentences maximum."
        
        try:
            response = self.model.generate_content([prompt, pil_image])
            return response.text
        except Exception as e:
            print(f"Error during Gemini Vision analysis: {str(e)}")
            return "Unable to analyze image due to an error."

def set_tesseract_path():
    default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    if os.path.exists(default_path):
        return default_path
    
    print("Tesseract OCR not found in the default location.")
    print("Please download and install Tesseract OCR from:")
    print("https://github.com/UB-Mannheim/tesseract/wiki")
    
    custom_path = input("Enter the full path to tesseract.exe (or press Enter to exit): ")
    
    if custom_path:
        if os.path.exists(custom_path):
            return custom_path
        else:
            print("Invalid path. Exiting.")
    
    exit()

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = set_tesseract_path()

def capture_screen_area(area):
    screenshot = pyautogui.screenshot(region=area)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def extract_text(image):
    try:
        return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract is not installed or the path is incorrect.")
        print("Please install Tesseract OCR and set the correct path.")
        exit()

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

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
    else:
        return "Invalid AI model selected."

def start_game_analysis(analyzer, area=None):
    print("Game Analysis mode started. Press Ctrl+C to exit.")
    try:
        while True:
            current_screenshot = analyzer.capture_full_screen()
            if area:
                current_screenshot = current_screenshot.crop(area)
            change_detected, focus_area, feedback = analyzer.detect_changes(current_screenshot)

            print(feedback)  # Always print feedback for better user awareness

            if change_detected:
                print("Analyzing...")
                focus_image = current_screenshot.crop(focus_area) if focus_area else current_screenshot
                analysis = analyzer.analyze_with_vision(focus_image)
                print(f"Analysis: {analysis}")
                speak_text(analysis)

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
    ai_model = simpledialog.askstring("AI Model Selection", "Choose AI model:\n1. OpenAI\n2. Gemini", initialvalue="1")
    mode_root.destroy()

    if ai_model == "1":
        ai_model = "openai"
    elif ai_model == "2":
        ai_model = "gemini"
    else:
        print("Invalid AI model selected. Exiting.")
        return

    analyzer = GameAnalyzer(ai_model)
    conversation_history = ConversationHistory()

    def on_ctrl_5():
        global selected_area
        print("Ctrl+5 pressed. Select the area for Game Analysis.")
        root = tk.Tk()
        selector = AreaSelector(root)
        selector.start()
        root.mainloop()

        selected_area = selector.get_coordinates()
        root.destroy()

        print(f"Selected area: {selected_area}")
        start_game_analysis(analyzer, selected_area)

    keyboard.add_hotkey('ctrl+5', on_ctrl_5)

    mode_root = tk.Tk()
    mode_root.withdraw()  # Hide the main window
    mode = simpledialog.askstring("Mode Selection", "Choose mode:\n1. Area Selection (Tesseract)\n2. Area Selection (AI)\n3. Game Analysis\n4. Wait for Ctrl+5\n5. Player in the Loop (with TTS) - Limited functionality with Gemini, might not work as expected", initialvalue="1")
    mode_root.destroy()

    selected_area = None

    if mode in ["1", "2"]:
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

                if mode == "1":
                    # Use Tesseract for local text analysis
                    text = extract_text(screen)
                    if text.strip():
                        print(f"Detected text: {text}")
                        print(f"Tesseract analysis: {text}")
                        speak_text(text)
                else:
                    # Use AI for text detection and analysis
                    analysis = analyzer.analyze_with_vision(Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)))
                    print(f"AI Vision analysis: {analysis}")
                    speak_text(analysis)

                # Wait for a short time before the next capture
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nGame Quest Reader stopped.")

    elif mode == "3":
        start_game_analysis(analyzer)

    elif mode == "4":
        print("Waiting for Ctrl+5 to be pressed. Press Ctrl+C to exit.")
        try:
            while True:
                if selected_area:
                    start_game_analysis(analyzer, selected_area)
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nGame Quest Reader stopped.")

    elif mode == "5":
        if ai_model == "gemini":
            print("Warning: Player in the Loop mode may have limited functionality with Gemini AI model and might not work as expected.")
        print("Starting Player in the Loop mode with automatic TTS...")
        player_in_the_loop(analyzer)

    else:
        print("Invalid mode selected. Exiting.")

if __name__ == "__main__":
    main()
