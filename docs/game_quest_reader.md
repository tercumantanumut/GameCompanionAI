# Game Quest Reader Documentation

## Overview

Game Quest Reader is an advanced application designed to enhance the gaming experience by providing real-time analysis and interpretation of game content. It uses AI-powered vision and text analysis to offer insights, strategic advice, and lore connections based on the game's visual and textual elements.

## Features

1. **Multiple Analysis Modes:**
   - Area Selection (Tesseract): Uses OCR to read text from a selected screen area.
   - Area Selection (AI): Uses AI vision to analyze a selected screen area.
   - Game Analysis: Continuously monitors the full screen for changes and provides AI analysis.
   - Wait for Ctrl+5: Allows users to trigger analysis on-demand.
   - Player in the Loop (with TTS): Interactive mode with AI responses and automatic text-to-speech.

2. **AI Model Options:**
   - OpenAI (GPT-4)
   - Google Gemini

3. **Text-to-Speech (TTS) Integration:** Speaks analysis results aloud.

4. **Conversation History:** Maintains context for more relevant AI responses.

5. **Screen Change Detection:** Automatically detects significant changes in the game screen.

## Setup

1. Install required dependencies:
   ```
   pip install opencv-python numpy pyautogui pytesseract pillow pyttsx3 openai google-generativeai python-dotenv tkinter keyboard
   ```

2. Install Tesseract OCR:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Set the path to the Tesseract executable in the script.

3. Set up API keys:
   - Create a `.env` file in the project root.
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key
     GOOGLE_API_KEY=your_google_api_key
     ```

## Usage

1. Run the script:
   ```
   python game_quest_reader.py
   ```

2. Select the AI model (OpenAI or Gemini).

3. Choose an analysis mode:
   - For Area Selection modes, use the mouse to select the desired screen area.
   - For Game Analysis, the program will continuously monitor the full screen.
   - In Wait for Ctrl+5 mode, press Ctrl+5 to trigger analysis.
   - In Player in the Loop mode, interact with the AI through the provided UI. If using Gemini, some features may be limited.

4. To exit, press Ctrl+C in the terminal or close the application window.

## Customization

- Adjust the analysis interval in the `start_game_analysis` function.
- Modify the AI prompts in the `analyze_text_with_ai` function for different analysis styles.
- Customize the UI in the `player_in_the_loop` function.

## Troubleshooting

- Ensure all dependencies are correctly installed.
- Verify that API keys are set correctly in the `.env` file.
- Check that Tesseract OCR is installed and the path is set correctly for OCR functionality.

## Contributing

Contributions to the Game Quest Reader are welcome. Please ensure that your code adheres to the existing style and includes appropriate documentation.

## License

[Specify the license under which this software is released]
