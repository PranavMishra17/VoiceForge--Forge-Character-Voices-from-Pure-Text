@echo off
echo Creating CosyVoice API Project Structure...

REM Create main directories
mkdir cosyvoice_api 2>nul
mkdir cosyvoice_api\config 2>nul
mkdir cosyvoice_api\core 2>nul
mkdir cosyvoice_api\api 2>nul
mkdir cosyvoice_api\utils 2>nul
mkdir cosyvoice_api\models 2>nul

mkdir data 2>nul
mkdir data\input 2>nul
mkdir data\input\audio_samples 2>nul
mkdir data\input\metadata 2>nul
mkdir data\output 2>nul
mkdir data\output\generated_audio 2>nul
mkdir data\output\embeddings 2>nul
mkdir data\output\logs 2>nul
mkdir data\temp 2>nul

mkdir pretrained_models 2>nul
mkdir pretrained_models\CosyVoice2-0.5B 2>nul

mkdir scripts 2>nul
mkdir examples 2>nul
mkdir tests 2>nul
mkdir third_party 2>nul

REM Create main package files
echo. > cosyvoice_api\__init__.py
echo. > cosyvoice_api\config\__init__.py
echo. > cosyvoice_api\config\settings.py
echo. > cosyvoice_api\core\__init__.py
echo. > cosyvoice_api\core\speaker_extractor.py
echo. > cosyvoice_api\core\dialogue_generator.py
echo. > cosyvoice_api\core\audio_processor.py
echo. > cosyvoice_api\api\__init__.py
echo. > cosyvoice_api\api\endpoints.py
echo. > cosyvoice_api\api\schemas.py
echo. > cosyvoice_api\utils\__init__.py
echo. > cosyvoice_api\utils\logger.py
echo. > cosyvoice_api\utils\file_utils.py
echo. > cosyvoice_api\utils\validation.py
echo. > cosyvoice_api\models\__init__.py

REM Create script files
echo. > scripts\setup_environment.py
echo. > scripts\download_models.py
echo. > scripts\test_installation.py

REM Create example files
echo. > examples\extract_embeddings_example.py
echo. > examples\generate_dialogue_example.py
echo. > examples\api_usage_example.py

REM Create test files
echo. > tests\__init__.py
echo. > tests\test_speaker_extractor.py
echo. > tests\test_dialogue_generator.py
echo. > tests\test_api.py
echo. > tests\test_basic.py

REM Create root files
echo. > requirements.txt
echo. > setup.py
echo. > README.md
echo. > main.py

echo.
echo Project structure created successfully!
echo.
echo Directory structure:
echo ├── cosyvoice_api/
echo │   ├── config/
echo │   ├── core/
echo │   ├── api/
echo │   ├── utils/
echo │   └── models/
echo ├── data/
echo │   ├── input/
echo │   ├── output/
echo │   └── temp/
echo ├── pretrained_models/
echo ├── scripts/
echo ├── examples/
echo ├── tests/
echo └── third_party/
echo.
echo All files created as empty files.
echo Copy and paste the content from the artifacts into each file.
echo.
echo Next steps:
echo 1. Copy content into each file
echo 2. Run: python scripts/setup_environment.py
echo 3. Run: python main.py setup
echo.
pause