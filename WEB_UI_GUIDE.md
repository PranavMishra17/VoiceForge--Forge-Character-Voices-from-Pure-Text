# VoiceForge Web UI Guide

## 🎉 Your Web UI is Ready!

The web interface has been completely enhanced with full support for all VoiceForge parameters including language selection, emotions, tones, and custom instructions.

---

## 🚀 Quick Start

### Start the Server:
```bash
python ui_server.py
```

### Open in Browser:
```
http://localhost:8008
```

**Default Port:** 8008 (can be changed with `VOICEFORGE_UI_PORT` environment variable)

---

## ✨ Features

### 1. 🎤 Speech Synthesis (Full-Featured)

The synthesis section now includes:

#### **Language Selection** (40+ Languages)
- **Dropdown menu** organized by categories:
  - Primary: English, Chinese, Japanese, Korean
  - Chinese Dialects: Cantonese, Shanghainese, Min Nan, Sichuanese
  - European: German, Spanish, French, Italian, Russian, Portuguese
  - Other: Arabic, Hindi, Thai, Vietnamese, and more
  - Mixed: Chinese-English code-switching

#### **Emotion Control** (12 Emotions with Icons)
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 🤩 Excited
- 😌 Calm
- 😰 Nervous
- 😎 Confident
- 🤔 Mysterious
- 🎭 Dramatic
- 🌸 Gentle
- ⚡ Energetic
- 💕 Romantic

#### **Tone Control** (9 Styles)
- Formal
- Casual
- Professional
- Friendly
- Serious
- Playful
- Authoritative
- Whispering
- Shouting

#### **Additional Controls**
- **Speed**: Slider from 0.5x (slow) to 2.0x (fast)
- **Custom Instructions**: Natural language input for fine-grained control
- **Speaker ID**: Use saved speaker embeddings
- **Prompt Audio/Text**: For real-time voice cloning

### 2. 🎙️ Speaker Management

- **Extract Embeddings**: Upload audio + transcript to create speaker profiles
- **Add from File**: Import .npy or .json embedding files
- **View Speakers**: Dropdown list of all available speakers
- **Delete Speakers**: Remove unwanted profiles

### 3. 📜 Dialogue Processing

- **Script Editor**: Multi-line text input with examples
- **Tag Support**: Full support for all parameters in dialogue tags
  ```
  [speaker:wizard,emotion:happy,language:english] Hello!
  [tone:whispering,speed:0.8] Shh, quiet...
  [language:japanese,emotion:excited] すごい！
  ```

### 4. 🎵 Audio Playback

- **Instant Playback**: Generated audio plays immediately in browser
- **Audio List**: View and play all generated files
- **File Info**: Shows filename and size

### 5. ⚙️ Configuration

- **Live Updates**: Change output directory and speaker database without restart
- **Status Display**: Real-time feedback with success/error indicators

---

## 🎨 UI Design

### Modern Gradient Theme
- Beautiful purple gradient background
- Clean white content cards
- Smooth animations and hover effects
- Responsive design for all screen sizes

### User-Friendly Features
- **Helpful Hints**: Gray text under each field explains what it does
- **Organized Sections**: Each feature in its own collapsible section
- **Visual Feedback**: Color-coded status messages (green = success, red = error)
- **Error Handling**: Clear error messages with details
- **Examples**: Placeholder text shows proper format

---

## 📖 Usage Examples

### Example 1: English Speech with Emotion
1. Enter text: "Hello, I'm so excited to meet you!"
2. Select Language: `English`
3. Select Emotion: `😊 Happy`
4. Select Speaker or leave empty for voice cloning
5. Click "🎵 Synthesize Speech"

### Example 2: Chinese with Formal Tone
1. Enter text: "欢迎来到我们的公司"
2. Select Language: `Chinese (Mandarin)`
3. Select Tone: `Formal`
4. Select Speed: `1.0`
5. Click "🎵 Synthesize Speech"

### Example 3: Japanese with Custom Instruction
1. Enter text: "ようこそ、魔法の世界へ！"
2. Select Language: `Japanese`
3. Enter Custom Instruction: "speak like an anime character"
4. Click "🎵 Synthesize Speech"

### Example 4: Multilingual Dialogue
1. Go to "📜 Dialogue Processing"
2. Enter dialogue name: "multilingual_test"
3. Enter script:
   ```
   [language:english,speaker:narrator] Once upon a time...
   [language:chinese,emotion:excited] 这真是太棒了！
   [language:japanese,tone:formal] ようこそいらっしゃいました。
   ```
4. Click "▶️ Process Dialogue"

---

## 🔧 Technical Details

### Backend (ui_server.py)
- **Framework**: FastAPI
- **Port**: 8008 (configurable)
- **CORS**: Enabled for cross-origin requests
- **File Uploads**: Supports audio, embedding files
- **API Endpoints**:
  - `/api/speakers` - List speakers
  - `/api/extract` - Extract embeddings
  - `/api/synthesize` - Generate speech
  - `/api/dialogue` - Process dialogue
  - `/api/audio/{filename}` - Serve audio files
  - `/api/audio_list` - List generated audio

### Frontend (webui/index.html)
- **No Dependencies**: Pure HTML/CSS/JavaScript
- **Modern CSS**: Gradients, animations, responsive grid
- **Fetch API**: Async communication with backend
- **FormData**: File uploads and form submission

---

## 🎯 Parameter Mapping

The Web UI maps perfectly to the command-line interface:

| Web UI Field | CLI Argument | Description |
|-------------|-------------|-------------|
| Language dropdown | `--language` | Output language |
| Emotion dropdown | `--emotion` | Emotional expression |
| Tone dropdown | `--tone` | Speaking style |
| Speed slider | `--speed` | Speech rate |
| Custom Instruction | `--instruction` | Natural language modulation |
| Speaker ID | `--speaker_id` | Saved speaker embedding |

**All parameters work exactly the same as the CLI!**

---

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check if port is already in use
lsof -i :8008

# Use different port
VOICEFORGE_UI_PORT=8009 python ui_server.py
```

### Cannot Load Speakers
- Check that `voiceforge_output/speakers/speaker_database.json` exists
- Click "🔄 Refresh" button in Speakers section
- Check status panel for error messages

### Audio Not Playing
- Check browser console for errors (F12)
- Verify audio files are in `voiceforge_output/audio/`
- Try refreshing the audio list

### Language Not Working
- Make sure you selected a language from the dropdown (not "Auto-detect")
- Check the status panel - it shows which language was applied
- Verify the language tag is supported (see README.md)

---

## 💡 Tips & Best Practices

### For Best Results:
1. **Extract Quality Speakers**: Use clean, clear audio samples
2. **Combine Parameters**: Mix language + emotion + tone for best expressiveness
3. **Test Speed**: Start with 1.0, adjust based on output
4. **Use Custom Instructions**: For unique styles not in dropdowns
5. **Save Your Work**: Download generated audio files before clearing

### Language Selection:
- **Auto-detect** works well for single-language text
- **Specify language** for multilingual content or mixed-language scenarios
- **Use dialect tags** for regional variations (Cantonese, Shanghainese, etc.)

### Emotion + Tone:
- Emotions affect the feeling: happy, sad, angry
- Tones affect the style: formal, casual, whispering
- Combine both for nuanced expression!

---

## 🚀 Next Steps

### Try These:
1. Extract a speaker embedding from your own voice
2. Generate speech in multiple languages
3. Create a multilingual dialogue with different emotions
4. Experiment with custom instructions
5. Adjust speed for different effects

### Advanced Usage:
- Use dialogue processing for entire story scripts
- Batch process multiple character voices
- Create voice libraries for game characters
- Generate audiobook-style narration with emotion control

---

## 📚 More Information

- **Full Documentation**: See README.md
- **Language Fixes**: See LANGUAGE_FIXES.md for details on language support
- **Test Results**: See TEST_RESULTS.md for validation tests
- **Command Line**: See README.md for CLI usage

---

## ✅ What's Working

- ✅ All 40+ languages supported
- ✅ 12 emotions with visual icons
- ✅ 9 tone styles
- ✅ Speed control (0.5x - 2.0x)
- ✅ Custom instructions
- ✅ Speaker management (extract, add, delete)
- ✅ Dialogue processing with full tag support
- ✅ Real-time audio playback
- ✅ Visual status feedback
- ✅ File upload (audio, embeddings)
- ✅ Live configuration

**The Web UI provides complete access to all VoiceForge capabilities!** 🎉
