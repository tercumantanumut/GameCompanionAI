# Game Companion AI

Game Companion AI is an advanced application designed to enhance the gaming experience by providing real-time analysis and interpretation of game content using AI-powered vision and text analysis. Whether you're navigating a complex quest, analyzing game environments, or interacting with characters, Game Companion AI aims to assist players with dynamic and customizable AI tools.

## Features

- **Multiple Analysis Modes**:
    - Area Selection
    - Game Analysis
    - Screenshot Analysis
    - Voice Input Analysis
    - Custom Area Analysis
- **Support for Multiple AI Models**:
    - OpenAI
    - Google Gemini
    - Ollama
- **Text-to-Speech Integration**:
    - Speak back game insights and recommendations
- **Conversation History**:
    - Keeps track of player interactions for context-aware responses
- **Screen Change Detection**:
    - Automatically updates based on in-game screen changes (not really working nice)
- **Voice Input**:
    - Players can ask questions or issue commands using their voice
- **Customizable AI Parameters**:
    - Fine-tune responses and behavior for Ollama AI models
 
Press Ctrl+5 for instant full-screen analysis.
Press Ctrl+V to use voice input for questions or commands.
Press Ctrl+6 to select a custom area for one-time analysis.

## Installation

To set up Game Companion AI, follow the steps below:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/game-companion-ai.git
    cd game-companion-ai
    ```

2. **Install Dependencies**:
    Ensure you have Python installed. Then, install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up API Keys**:
    Create a `.env` file in the root directory and add your API keys for the various AI models (OpenAI, Google Gemini, Ollama):
    ```env
    OPENAI_API_KEY=your_openai_key
    GOOGLE_GEMINI_API_KEY=your_google_gemini_key
    OLLAMA_API_KEY=your_ollama_key
    ```

    Alternatively, you can add API keys via the command line for different operating systems:

    ### How to Set API Keys via Command Window

    ### Windows

    #### Using Command Prompt:
    ```cmd
    set OPENAI_API_KEY=your_openai_key
    set GOOGLE_GEMINI_API_KEY=your_google_gemini_key
    set OLLAMA_API_KEY=your_ollama_key
    ```

    This sets the API key for the current session. To set it permanently:
    ```cmd
    setx OPENAI_API_KEY "your_openai_key"
    setx GOOGLE_GEMINI_API_KEY "your_google_gemini_key"
    setx OLLAMA_API_KEY "your_ollama_key"
    ```

    #### Using PowerShell:
    ```powershell
    $Env:OPENAI_API_KEY="your_openai_key"
    $Env:GOOGLE_GEMINI_API_KEY="your_google_gemini_key"
    $Env:OLLAMA_API_KEY="your_ollama_key"
    ```

    For a permanent change:
    ```powershell
    [System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your_openai_key', 'User')
    [System.Environment]::SetEnvironmentVariable('GOOGLE_GEMINI_API_KEY', 'your_google_gemini_key', 'User')
    [System.Environment]::SetEnvironmentVariable('OLLAMA_API_KEY', 'your_ollama_key', 'User')
    ```

    ### macOS/Linux

    #### Temporarily for the current session:
    ```bash
    export OPENAI_API_KEY=your_openai_key
    export GOOGLE_GEMINI_API_KEY=your_google_gemini_key
    export OLLAMA_API_KEY=your_ollama_key
    ```

    #### Permanently (add to shell configuration file):

    For `bash`, add the following to `~/.bashrc` (Linux) or `~/.bash_profile` (macOS):
    ```bash
    echo 'export OPENAI_API_KEY=your_openai_key' >> ~/.bashrc  # Linux
    echo 'export GOOGLE_GEMINI_API_KEY=your_google_gemini_key' >> ~/.bashrc  # Linux
    echo 'export OLLAMA_API_KEY=your_ollama_key' >> ~/.bashrc  # Linux

    echo 'export OPENAI_API_KEY=your_openai_key' >> ~/.bash_profile  # macOS
    echo 'export GOOGLE_GEMINI_API_KEY=your_google_gemini_key' >> ~/.bash_profile  # macOS
    echo 'export OLLAMA_API_KEY=your_ollama_key' >> ~/.bash_profile  # macOS
    ```

    For `zsh` (common on macOS):
    ```bash
    echo 'export OPENAI_API_KEY=your_openai_key' >> ~/.zshrc
    echo 'export GOOGLE_GEMINI_API_KEY=your_google_gemini_key' >> ~/.zshrc
    echo 'export OLLAMA_API_KEY=your_ollama_key' >> ~/.zshrc
    ```

    After editing, reload the shell configuration:
    ```bash
    source ~/.bashrc  # Linux
    source ~/.bash_profile  # macOS
    source ~/.zshrc  # macOS using zsh
    ```

4. **(Optional) Install and Set Up Ollama**:
    If you wish to use the Ollama AI model, follow the Ollama-specific installation instructions found [here](https://ollama.com/docs/setup).

5. **Run the Application**:
    Once everything is set up, you can start the application using:
    ```bash
    python game_quest_reader.py
    ```

For more detailed instructions, troubleshooting, or usage guidelines, please refer to the [documentation](docs/game_quest_reader.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
