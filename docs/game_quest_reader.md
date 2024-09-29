# Game Companion AI Documentation

## Overview

Game Companion AI is an advanced application designed to enhance the gaming experience by providing real-time analysis and interpretation of game content. It uses AI-powered vision and text analysis to offer insights, strategic advice, and lore connections based on the game's visual and textual elements.

## Features

1. **Multiple Analysis Modes:**
   - Area Selection (AI): Uses AI vision to analyze a selected screen area.
   - Game Analysis: Continuously monitors the full screen for changes and provides AI analysis.
   - Screenshot Analysis (Ctrl+5): Allows users to trigger full-screen analysis on-demand.
   - Voice Input Analysis (Ctrl+V): Listens for voice input and analyzes it along with a screenshot.
   - Custom Area Analysis (Ctrl+6): Lets users select a custom area for analysis.

2. **AI Model Options:**
   - OpenAI (GPT-4 with vision capabilities)
   - Google Gemini
   - Ollama (with customizable parameters)

3. **Text-to-Speech (TTS) Integration:** Speaks analysis results aloud.

4. **Conversation History:** Maintains context for more relevant AI responses.

5. **Screen Change Detection:** Automatically detects significant changes in the game screen.

6. **Voice Input:** Allows players to ask questions or give commands using their voice.

7. **Customizable AI Parameters:** For Ollama, users can adjust temperature, top_p, and top_k values.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/game-quest-reader.git
   cd game-quest-reader
   ```

2. Install required dependencies:
   ```
   pip install opencv-python numpy pyautogui pillow pyttsx3 openai google-generativeai python-dotenv keyboard requests SpeechRecognition pywin32 scikit-image torch
   ```


4. Set up API keys:
   - Create a `.env` file in the project root.
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key
     GOOGLE_API_KEY=your_google_api_key
     OPENAI_API_BASE=https://api.openai.com/v1  # Optional: for OpenAI API base URL
     ```

5. (Optional) Install and set up Ollama:
   - Follow the installation instructions at: https://github.com/jmorganca/ollama
   - Ensure Ollama is running and accessible at `http://localhost:11434`

## Usage

1. Run the script:
   ```
   python game_quest_reader.py
   ```

2. Select the AI model (OpenAI, Gemini, or Ollama).
   - If choosing Ollama, you'll be prompted to set temperature, top_p, and top_k values.

3. Choose an analysis mode:
   - Area Selection (AI): Select a screen area for continuous analysis.
   - Game Analysis: The program will continuously monitor the full screen.

4. Additional features:
   - Press Ctrl+5 for instant full-screen analysis.
   - Press Ctrl+V to use voice input for questions or commands.
   - Press Ctrl+6 to select a custom area for one-time analysis.

5. To exit, press Ctrl+C in the terminal or close the application window.

## Customization

- Adjust the analysis interval in the `start_game_analysis` function.
- Modify the AI prompts in the `analyze_with_vision` and `analyze_text_with_ai` functions for different analysis styles.
- Customize the Ollama model and parameters in the `main` function.

## Troubleshooting

- Ensure all dependencies are correctly installed.
- Verify that API keys are set correctly in the `.env` file.
- For Ollama issues, ensure the Ollama service is running and accessible.
- If voice input is not working, check your microphone settings and ensure you have an active internet connection for speech recognition.

## Contributing

Contributions to the Game Companion AI are welcome. Please ensure that your code adheres to the existing style and includes appropriate documentation. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
