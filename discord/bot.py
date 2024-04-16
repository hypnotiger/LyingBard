import discord
import os
import asyncio
import threading
from lyingbard.tts import text_to_speech_and_save
from lyingbard.models import Speaker, Generation
import atexit

bot = discord.Bot()

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

@bot.slash_command(name = "speak", description = "Make the Lying Bard speak")
async def speak(ctx: discord.commands.context.ApplicationContext, text: str, speaker_name:str="Grove"):
    vc = ctx.voice_client 
    if not vc: # check if the bot is not in a voice channel
        if ctx.author.voice == None:
            return await ctx.respond("Join a voice channel LyingBard can connect to.")
        vc = await ctx.author.voice.channel.connect() # connect to the voice channel

    await ctx.defer()
    speaker = await Speaker.objects.aget(name=speaker_name)

    generation = text_to_speech_and_save(text=text, model=None, speaker=speaker, user=None)

    await ctx.send_followup(file=discord.File(generation.file.path))
    
    if vc.is_playing():
        vc.stop()
    vc.play(discord.FFmpegPCMAudio(generation.file.path))

def bot_thread_fn(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bot.start(os.getenv('LYINGBARD_DISCORD_TOKEN')))
    except:
        print("Discord not available.")

event_loop = asyncio.get_event_loop()
bot_thread = threading.Thread(target=bot_thread_fn,args=(event_loop,), daemon=True)
atexit.register(event_loop.stop)
bot_thread.start()