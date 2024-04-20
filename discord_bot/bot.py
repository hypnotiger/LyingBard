import discord
import os
import asyncio
import threading
from lyingbard.models import TTS
import atexit
from io import BytesIO
import platformdirs
from pathlib import Path
from tempfile import NamedTemporaryFile

DISCORD_DIR = Path(platformdirs.user_data_dir("LyingBard", "FUBAI"))/"Bot"/"Discord"

bot = discord.Bot()

tts = None

@bot.event
async def on_ready():
    global tts
    tts = TTS()
    print(f"{bot.user} is ready and online!")

@bot.slash_command(name = "list", description = "List TTS's")
async def list(ctx: discord.commands.context.ApplicationContext):
    return await ctx.respond(", ".join(TTS.list_usable()), ephemeral=True)

@bot.slash_command(name = "speak", description = "Make the Lying Bard speak")
async def speak(ctx: discord.commands.context.ApplicationContext, text: str, tts_name:str=None):
    global tts

    vc = ctx.voice_client 
    if not vc: # check if the bot is not in a voice channel
        if ctx.author.voice == None:
            return await ctx.respond("Join a voice channel LyingBard can connect to.", ephemeral=True)
        vc = await ctx.author.voice.channel.connect() # connect to the voice channel

    await ctx.defer()

    if tts_name != None:
        if not TTS.usable(tts_name):
            return await ctx.send_followup(f'TTS "{tts_name}" does not exist.', ephemeral=True)
        tts.change(tts_name)
    elif tts.name == None:
        return await ctx.send_followup(f"No TTS active. Specify a TTS name.", ephemeral=True)

    audio = tts.speak(text)

    audio.export(DISCORD_DIR/"LyingBard_Generation.mp3", format="mp3")

    await ctx.send_followup(file=discord.File(DISCORD_DIR/"LyingBard_Generation.mp3"))

    if vc.is_playing():
        vc.stop()
    vc.play(discord.FFmpegPCMAudio(DISCORD_DIR/"LyingBard_Generation.mp3"))

    # (DISCORD_DIR/"LyingBard_Generation.mp3").unlink()

@bot.slash_command(name = "stop", description = "Make the Lying Bard stop")
async def stop(ctx: discord.commands.context.ApplicationContext):
    vc = ctx.voice_client 
    if not vc: # check if the bot is not in a voice channel
        return await ctx.respond("The LyingBard is not in a voice channel.", ephemeral=True)
    if vc.is_playing():
        vc.stop()
        return await ctx.respond("Stopped.", ephemeral=True)
    else:
        return await ctx.respond("The LyingBard is not talking.", ephemeral=True)

def bot_thread_fn(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(bot.start(read_token()))

def start_bot():
    event_loop = asyncio.get_event_loop()
    bot_thread = threading.Thread(target=bot_thread_fn,args=(event_loop,), daemon=True)
    atexit.register(event_loop.stop)
    bot_thread.start()

def read_token() -> str:
    with open(DISCORD_DIR/"Token.txt") as file:
        return file.readline()