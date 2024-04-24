![image](https://github.com/GroveDG/LyingBard/assets/87248833/250f3b66-3bdc-4c01-b575-fda0265b341e)

## Download (Windows)
[Download](https://drive.google.com/file/d/16VTnAVjqCFLJ6_hMZsNr4BkAuhNvZnFP/view?usp=drive_link)

If you're on Linux, I'm sure you can figure it out.

## Setup
### Install ffmpeg
<ol>
  <li><a href=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip>Download</a></li>
  <li>Extract this somewhere. Doesn't matter where, but ideally somewhere you're ok keeping it forever.</li>
  <li>Open the Windows start menu.</li>
  <li>Search for "Edit the system environment variables".</li>
  <li>Click "Environment Variables..."</li>
  <li>In the upper box, double click "Path".</li>
  <li>Click "New".</li>
  <li>Paste in the path to specifically the ffmpeg "bin" folder.</li>
</ol>
If you're on Linux this is actually easier for you. You can figure it out.

## Pretrained Models
<ul>
  <li><a href="https://drive.google.com/file/d/1smJK-7fDIkMA10Pwgoj2KTE_ceKAHi2D/view?usp=sharing">Grove</a> (Me)</li>
  <li><a href="https://drive.google.com/file/d/1U7xdZ0bqbeOfPkHn9ffsOaXsWZ0xaeUK/view?usp=sharing">CoolVids</a> (The Gnome, does not sound like him)</li>
</ul>

## GPU
[Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)

GPU goes much faster. Only available for NVIDIA GPUs.

## Training
<ol>
  <li>Model > Train, then follow the wizard.</li>
  <li>When you select the audio the wizard is sent behind main window for some reason.</li>
  <li>Look at the console for training progress.</li>
  <li>Close and Reopen the program to refresh the list of TTS's.</li>
</ol>

## Use/Share Trained Models
<ol>
  <li>Model > Open Folder</li>
  <li>Open the "TTS" folder</li>
  <li>Share/paste the model folder</li>
</ol>

## Discord Bot
<ol>
  <li>Go to https://discord.com/developers/applications</li>
  <li>Create an application.</li>
  <li>Change installation settings to give the "applications.commands" and "bot" scopes and the "Speak" permission.</li>
  <li>Install the bot to your Discord server.</li>
  <li>Go to the "Bot" tab and get your token.</li>
  <li>In LyingBard, Bot > Discord > Setup, copy in your token.</li>
  <li>Bot > Discord > Start</li>
</ol>

## Help
Send me an ask on [Tumblr](https://www.tumblr.com/lyingbard), send me an email at <grovedgdev@gmail.com> or submit a Github issue.
