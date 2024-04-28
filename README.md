![image](https://github.com/GroveDG/LyingBard/assets/87248833/250f3b66-3bdc-4c01-b575-fda0265b341e)

## Download (Windows)
[Download](https://drive.google.com/file/d/16VTnAVjqCFLJ6_hMZsNr4BkAuhNvZnFP/view?usp=drive_link)

If you're on Linux, I'm sure you can figure it out.

## Setup
### Install ffmpeg
1. [Download](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)
2. Extract this somewhere. Doesn't matter where, but ideally somewhere you're ok keeping it forever.
3. Open the Windows start menu.
4. Search for "Edit the system environment variables".
5. Click "Environment Variables..."
6. In the upper box, double click "Path".
7. Click "New".
8. Paste in the path to specifically the ffmpeg "bin" folder.

If you're on Linux this is actually easier for you. You can figure it out.

### Start up
This will take a while.

If you see a command line pop up, it's loading. This command line will have info that the GUI doesn't have yet. Take a look here if something is wrong and send me what's in here if you need help.

Here are things you might see in the command line:
- `cpu`/`cuda`: This tells you what device LyingBard will be running on. CUDA is GPU and will run faster and CPU will run slower.</li>
- Warnings starting with `torch/`: These are usually `UserWarning`s for deprecation. These are totally fine. Since LyreBird is old, a lot of the contemporary interfaces I'm using have since been deprecated. That's what it's "warning" you about.</li>
- `pydub\utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work`: This means you need to install ffmpeg.</li>
- Any Error (not warning): This means something has gone wrong and you should probably report it to me. Send me the full error message.</li>

## GPU (Optional, Recommended)
[Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)

GPU goes much faster. Only available for NVIDIA GPUs.

## Pretrained Models
- [Grove](https://drive.google.com/file/d/1smJK-7fDIkMA10Pwgoj2KTE_ceKAHi2D/view?usp=sharing) (Me)
- [CoolVids](https://drive.google.com/file/d/1U7xdZ0bqbeOfPkHn9ffsOaXsWZ0xaeUK/view?usp=sharing) (The Gnome, does not sound like him)

## Use/Share Trained Models
1. Model > Open Folder
2. Open/Create the "TTS" folder
3. Share/Paste the model folder

## Training
1. Model > Train, then follow the wizard.
2. When you select the audio the wizard is sent behind main window for some reason.
3. Look at the console for training progress.
4. Close and Reopen the program to refresh the list of TTS's.

## Discord Bot
1. Go to https://discord.com/developers/applications
2. Create an application.
3. Change installation settings to give the "applications.commands" and "bot" scopes and the "Speak" permission.
4. Install the bot to your Discord server.
5. Go to the "Bot" tab and get your token.
6. In LyingBard, Bot > Discord > Setup, copy in your token.
7. Bot > Discord > Start

## Help
Send me an ask on [Tumblr](https://www.tumblr.com/lyingbard), send me an email at <grovedgdev@gmail.com>, or submit a Github issue.
