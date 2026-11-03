from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import shutil
import sys
import subprocess
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
SONGS_PER_PAGE = 25
PLAYLISTS_PER_PAGE = 23  # Leave room for previous/next options in the same menu.
EMPTY_CHANNEL_TIMEOUT = 5 * 60


def configured_path(variable_name: str, default: Path) -> Path:
    value = os.getenv(variable_name, "").strip()
    if not value:
        return default.resolve()

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


MUSIC_DIR = configured_path("MUSIC_DIR", BASE_DIR / "music")
FFMPEG_EXECUTABLE = os.getenv("FFMPEG_PATH", "").strip() or shutil.which("ffmpeg")


def get_test_guild_id() -> int | None:
    raw_value = os.getenv("DISCORD_GUILD_ID", "").strip()
    if not raw_value:
        return None

    try:
        return int(raw_value)
    except ValueError as error:
        raise SystemExit("DISCORD_GUILD_ID должен содержать только цифры.") from error


TEST_GUILD_ID = get_test_guild_id()


@dataclass
class PlaybackState:
    serial: int = 0
    track: Path | None = None
    volume_percent: int = 100
    playlist: str = "."
    song_page: int = 0
    playlist_page: int = 0
    repeat_enabled: bool = False


@dataclass
class ReplyBuffer:
    interaction: discord.Interaction
    messages: list[tuple[str, bool]] = field(default_factory=list)


@dataclass
class EmptyChannelCountdown:
    voice: discord.VoiceClient
    channel_id: int
    task: asyncio.Task[None] | None = None
    disconnecting: bool = False


playback: dict[int, PlaybackState] = {}
player_messages: dict[int, discord.Message | discord.PartialMessage] = {}
player_locks: dict[int, asyncio.Lock] = {}
panel_update_locks: dict[int, asyncio.Lock] = {}
repeat_tasks: set[asyncio.Task[None]] = set()
panel_refresh_tasks: dict[int, asyncio.Task[None]] = {}
panel_refresh_dirty: set[int] = set()
panel_refresh_events: dict[int, asyncio.Event] = {}
empty_channel_timers: dict[int, EmptyChannelCountdown] = {}
pending_replies: ContextVar[ReplyBuffer | None] = ContextVar("music_replies", default=None)


class VoiceConnectionError(RuntimeError):
    """The voice link was lost between connecting and starting the decoder."""


class ProgressAudio(discord.PCMVolumeTransformer[discord.FFmpegPCMAudio]):
    """Count decoded audio frames; pausing or changing volume never resets time."""

    def __init__(self, source: discord.FFmpegPCMAudio, volume: float, *, position: float = 0.0) -> None:
        super().__init__(source, volume=volume)
        self.frames_read = 0
        self.start_position = position
        self._startup_hold = False
        self._pause_check: Callable[[], bool] | None = None

    def hold_for_pause(self, voice: discord.VoiceClient) -> None:
        # VoiceClient.play starts its thread immediately. Send only silence
        # until pause has been applied, then leave the decoder untouched until resume.
        self._startup_hold = True
        self._pause_check = voice.is_paused

    def release_startup_hold(self) -> None:
        self._startup_hold = False

    def read(self) -> bytes:
        if self._startup_hold or (self._pause_check is not None and self._pause_check()):
            return b"\0" * 3840
        self._pause_check = None
        data = super().read()
        if not data:
            # discord.py inspects the outer source, so propagate decoder errors
            # from the wrapped FFmpeg source instead of reporting a normal EOF.
            error = getattr(self.original, "_current_error", None)
            process = getattr(self.original, "_process", None)
            if error is None and isinstance(process, subprocess.Popen):
                # stdout can reach EOF a moment before Windows exposes the exit
                # code. This runs in the audio thread, not the Discord event loop.
                try:
                    returncode = process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("FFmpeg не завершился после закрытия аудиопотока.") from None
                if returncode != 0 and not getattr(self.original, "_stopped", False):
                    error_type = getattr(discord, "FFmpegProcessError", RuntimeError)
                    error = error_type(f"FFmpeg завершился с ошибкой (код {returncode}).")
            if error is not None:
                raise error
        if data:
            self.frames_read += 1
        return data

    def cleanup(self) -> None:
        process = getattr(getattr(self, "original", None), "_process", None)
        try:
            super().cleanup()
        finally:
            if isinstance(process, subprocess.Popen):
                # discord.py may leave pipe handles open when FFmpeg has already
                # exited. Releasing them also permits deleting/replacing files.
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()

    @property
    def position(self) -> float:
        return self.start_position + self.frames_read * 0.02


@lru_cache(maxsize=128)
def track_duration(track: Path, modified_ns: int, size: int) -> float:
    # File metadata is part of the cache key so replacing a song invalidates it.
    del modified_ns, size
    executable = shutil.which(FFMPEG_EXECUTABLE or "ffmpeg")
    probe = Path(executable).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe") if executable else None
    if probe is None or not probe.is_file():
        raise FileNotFoundError("ffprobe не найден.")
    # lru_cache never caches exceptions. Only successful probes are retained.
    result = subprocess.run(
        [str(probe), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(track)],
        capture_output=True, text=True, timeout=5, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("ffprobe вернул некорректную длительность.")
    return duration


def format_time(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02}:{seconds:02}"


async def get_track_duration(track: Path) -> float | None:
    try:
        stat = track.stat()
        return await asyncio.to_thread(track_duration, track, stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def build_player_embed(guild: discord.Guild, duration: float | None) -> discord.Embed:
    state = playback.get(guild.id, PlaybackState())
    voice = guild.voice_client
    active = voice is not None and (voice.is_playing() or voice.is_paused())
    embed = discord.Embed(title="🎵 Музыкальный плеер", colour=0x5865F2)
    if active and state.track is not None:
        position = voice.source.position if isinstance(voice.source, ProgressAudio) else 0
        position = min(position, duration) if duration else position
        marker = min(17, int(position / duration * 18)) if duration else 0
        bar = "━" * marker + "●" + "─" * (17 - marker)
        status = "⏸ Пауза" if voice.is_paused() else "▶ Сейчас играет"
        track_name = discord.utils.escape_markdown(state.track.stem)[:256]
        embed.description = f"**{track_name}**\n{status}\n\n`{format_time(position)}  {bar}  {format_time(duration) if duration else '--:--'}`"
    else:
        embed.description = "Музыка остановлена.\nНажми **Пуск / Пауза** или выбери песню командой `/play`."
    embed.add_field(name="Плейлист", value=discord.utils.escape_markdown(playlist_label(state.playlist))[:1024], inline=True)
    embed.add_field(name="Громкость", value=f"{state.volume_percent}%", inline=True)
    embed.add_field(name="Повтор", value="Одна песня" if state.repeat_enabled else "Выключен", inline=True)
    if voice is not None:
        embed.add_field(name="Канал", value=voice.channel.mention, inline=True)
    embed.set_footer(text="Локальная музыка • обновление каждые 10 с • /player")
    return embed


async def player_embed(guild: discord.Guild) -> discord.Embed:
    state = playback.get(guild.id)
    track = state.track if state else None
    duration = await get_track_duration(track) if track is not None else None
    # Another command could have changed the song while metadata was loading.
    if state is not None and state.track != track:
        duration = None
    return build_player_embed(guild, duration)


async def _refresh_player_once(guild: discord.Guild) -> None:
    # Serialize timer and interaction edits without holding the playback lock.
    # Resolve the active message and build its payload only after taking the
    # update lock, so queued requests cannot resend an older page or playlist.
    async with panel_update_locks.setdefault(guild.id, asyncio.Lock()):
        message = player_messages.get(guild.id)
        if message is None:
            return
        try:
            embed = await player_embed(guild)
            if player_messages.get(guild.id) is not message:
                return  # Another panel became active while metadata was loading.
            await message.edit(embed=embed, view=PlayerView(guild.id))
        except (discord.NotFound, discord.Forbidden):
            if player_messages.get(guild.id) is message:
                player_messages.pop(guild.id, None)
        except discord.HTTPException:
            logging.warning("Не удалось обновить панель плеера", exc_info=True)


async def _refresh_player_worker(guild: discord.Guild) -> None:
    current_event = None
    try:
        while guild.id in panel_refresh_dirty and not bot._music_closing:
            panel_refresh_dirty.discard(guild.id)
            current_event = panel_refresh_events.pop(guild.id, None)
            await _refresh_player_once(guild)
            if current_event is not None:
                current_event.set()
                current_event = None
    except Exception:
        logging.exception("Не удалось обновить панель плеера")
    finally:
        if current_event is not None:
            current_event.set()
        if panel_refresh_tasks.get(guild.id) is asyncio.current_task():
            panel_refresh_tasks.pop(guild.id, None)
            panel_refresh_dirty.discard(guild.id)
            pending = panel_refresh_events.pop(guild.id, None)
            if pending is not None:
                pending.set()


def request_player_refresh(guild: discord.Guild) -> asyncio.Task[None] | None:
    """One worker per guild; any number of updates collapse into its next edit."""
    if bot._music_closing or guild.id not in player_messages:
        return None
    panel_refresh_events.setdefault(guild.id, asyncio.Event())
    panel_refresh_dirty.add(guild.id)
    task = panel_refresh_tasks.get(guild.id)
    if task is None or task.done():
        task = asyncio.create_task(_refresh_player_worker(guild))
        panel_refresh_tasks[guild.id] = task
    return task


async def refresh_player(guild: discord.Guild) -> None:
    task = request_player_refresh(guild)
    if task is not None:
        # Wait for this requested edit, not for the worker to become idle.
        # Continuous short repeats must not keep an interaction alive forever.
        await panel_refresh_events[guild.id].wait()


@tasks.loop(seconds=10)
async def refresh_players() -> None:
    for guild_id in list(player_messages):
        guild = bot.get_guild(guild_id)
        if guild is not None:
            request_player_refresh(guild)


@refresh_players.before_loop
async def before_refresh_players() -> None:
    await bot.wait_until_ready()


def available_playlists() -> dict[str, Path]:
    """Root files plus immediate child folders; never follow paths outside music."""
    playlists = {".": MUSIC_DIR}
    if MUSIC_DIR.is_dir():
        root = MUSIC_DIR.resolve()
        for directory in sorted(MUSIC_DIR.iterdir(), key=lambda path: path.name.casefold()):
            if directory.is_dir() and directory.resolve().parent == root:
                playlists[directory.name] = directory
    return playlists


def playlist_label(playlist: str) -> str:
    return "Без папки" if playlist == "." else playlist


def playlist_token(playlist: str) -> str:
    # Discord select values are limited to 100 characters. Folder names can be
    # longer, so use a stable token and resolve it only against discovered folders.
    return hashlib.sha256(playlist.encode("utf-8")).hexdigest()[:32]


def song_token(playlist: str, track: Path) -> str:
    # Include the playlist so a stale menu cannot select a same-named file
    # from a different folder after another member switches the playlist.
    return hashlib.sha256(f"{playlist}\0{track.name}".encode("utf-8")).hexdigest()[:32]


def song_page_bounds(requested: int, track_count: int) -> tuple[int, int]:
    pages = max(1, (track_count + SONGS_PER_PAGE - 1) // SONGS_PER_PAGE)
    return max(0, min(requested, pages - 1)), pages


def playlist_page_bounds(requested: int, playlist_count: int) -> tuple[int, int]:
    pages = max(1, (playlist_count + PLAYLISTS_PER_PAGE - 1) // PLAYLISTS_PER_PAGE)
    return max(0, min(requested, pages - 1)), pages


def build_song_list(playlist: str, tracks: list[Path], *, numbered: bool = True) -> str:
    header = f"**Плейлист: {discord.utils.escape_markdown(playlist_label(playlist))[:600]}**\n"
    footer = "\n\nВыбрать песню можно в списке под плейлистами в `/player` или через `/play`."
    lines: list[str] = []
    for index, track in enumerate(tracks, 1):
        prefix = f"{index}. " if numbered else "• "
        line = prefix + discord.utils.escape_markdown(track.name)
        remaining = len(tracks) - index
        tail = f"\n…и ещё {remaining}." if remaining else ""
        if len(header) + len("\n".join(lines + [line])) + len(tail) + len(footer) > 2000:
            break
        lines.append(line)
    remaining = len(tracks) - len(lines)
    body = "\n".join(lines) or "В плейлисте пока нет песен."
    if remaining:
        body += f"\n…и ещё {remaining}."
    return header + body + footer


def selected_playlist(guild_id: int | None) -> str:
    state = playback.get(guild_id)
    return state.playlist if state else "."


def available_tracks(playlist: str = ".") -> list[Path]:
    directory = available_playlists().get(playlist)
    if directory is None or not directory.is_dir():
        return []

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and path.resolve().parent == directory.resolve()
        ),
        key=lambda path: path.name.casefold(),
    )


def find_track(query: str, playlist: str = ".") -> tuple[Path | None, str | None]:
    """Select only a discovered file inside the chosen playlist."""
    cleaned_query = Path(query.strip()).name.casefold()
    if not cleaned_query:
        return None, "Укажи название песни. Посмотреть список можно командой `/songs`."

    if playlist not in available_playlists():
        return None, "Папка плейлиста переименована или удалена. Выбери плейлист заново в `/player`."
    tracks = available_tracks(playlist)
    exact = [
        path
        for path in tracks
        if path.name.casefold() == cleaned_query or path.stem.casefold() == cleaned_query
    ]
    if len(exact) == 1:
        return exact[0], None

    partial = [
        path
        for path in tracks
        if cleaned_query in path.name.casefold() or cleaned_query in path.stem.casefold()
    ]
    if len(partial) == 1:
        return partial[0], None
    if len(partial) > 1:
        examples = ", ".join(f"`{path.name}`" for path in partial[:5])
        return None, f"Нашлось несколько совпадений: {examples}. Укажи точнее."

    return None, "Такой песни нет в выбранном плейлисте. Посмотреть список: `/songs`; сменить плейлист: `/player`."


def voice_channel_empty(channel: discord.abc.GuildChannel) -> bool | None:
    members = getattr(channel, "members", None)
    if not isinstance(members, (list, tuple)):
        return None  # Do not disconnect when membership is not known yet.
    states = getattr(channel, "voice_states", None)
    if isinstance(states, dict):
        known = {member.id: member for member in members}
        # Voice states can be complete while the member cache is not. Treat
        # unknown occupants as people instead of disconnecting their music.
        return not any(member_id not in known or not known[member_id].bot for member_id in states)
    return not any(not member.bot for member in members)


def cancel_empty_channel_timer(guild_id: int) -> None:
    existing = empty_channel_timers.get(guild_id)
    if existing is not None and existing.disconnecting:
        return  # The deadline passed; let the committed disconnect finish.
    countdown = empty_channel_timers.pop(guild_id, None)
    if countdown is not None and countdown.task is not None:
        if countdown.task is not asyncio.current_task():
            countdown.task.cancel()


def update_empty_channel_timer(guild: discord.Guild) -> None:
    voice = guild.voice_client
    if voice is None or bot._music_closing:
        cancel_empty_channel_timer(guild.id)
        return
    empty = voice_channel_empty(voice.channel)
    existing = empty_channel_timers.get(guild.id)
    if empty is False:
        cancel_empty_channel_timer(guild.id)
        return
    if empty is None:
        if existing is not None and (existing.voice is not voice or existing.channel_id != voice.channel.id):
            cancel_empty_channel_timer(guild.id)
        return
    if (existing is not None and existing.voice is voice and existing.channel_id == voice.channel.id
            and existing.task is not None and not existing.task.done()):
        return  # Mutes, deafens and unrelated voice events do not reset the deadline.
    cancel_empty_channel_timer(guild.id)
    countdown = EmptyChannelCountdown(voice, voice.channel.id)
    empty_channel_timers[guild.id] = countdown
    countdown.task = asyncio.create_task(leave_empty_channel_after_timeout(guild, countdown))


async def leave_empty_channel_after_timeout(guild: discord.Guild, countdown: EmptyChannelCountdown) -> None:
    try:
        await asyncio.sleep(EMPTY_CHANNEL_TIMEOUT)
        async with player_locks.setdefault(guild.id, asyncio.Lock()):
            voice = guild.voice_client
            if (empty_channel_timers.get(guild.id) is not countdown or voice is not countdown.voice
                    or voice.channel.id != countdown.channel_id or voice_channel_empty(voice.channel) is not True):
                return
            state = playback.setdefault(guild.id, PlaybackState())
            state.serial += 1
            state.track = None
            state.repeat_enabled = False
            countdown.disconnecting = True
            await voice.disconnect(force=True)
            logging.info("Вышел из пустого голосового канала после 5 минут, сервер %s", guild.id)
        request_player_refresh(guild)
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Не удалось выйти из пустого голосового канала, сервер %s", guild.id)
    finally:
        if empty_channel_timers.get(guild.id) is countdown:
            empty_channel_timers.pop(guild.id, None)


class MusicBot(commands.Bot):
    def __init__(self) -> None:
        self._music_closing = False
        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=discord.Intents.default(),
            allowed_mentions=discord.AllowedMentions.none(),
        )

    async def setup_hook(self) -> None:
        self.add_view(PlayerView(register_all=True))
        if TEST_GUILD_ID is not None:
            guild = discord.Object(id=TEST_GUILD_ID)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            print(f"Команды синхронизированы с тестовым сервером: {len(synced)}")
        else:
            synced = await self.tree.sync()
            print(f"Глобальные команды синхронизированы: {len(synced)}")
        refresh_players.start()

    async def close(self) -> None:
        self._music_closing = True
        refresh_players.cancel()
        pending = set(repeat_tasks) | set(panel_refresh_tasks.values())
        pending.update(timer.task for timer in empty_channel_timers.values() if timer.task is not None)
        empty_channel_timers.clear()
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        panel_refresh_tasks.clear()
        panel_refresh_dirty.clear()
        panel_refresh_events.clear()
        repeat_tasks.clear()
        await super().close()


bot = MusicBot()


@bot.event
async def on_ready() -> None:
    if bot.user is not None:
        print(f"Бот запущен: {bot.user} (ID: {bot.user.id})")
    for guild in bot.guilds:
        update_empty_channel_timer(guild)


@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState) -> None:
    update_empty_channel_timer(member.guild)


async def send_deferred_reply(
    interaction: discord.Interaction, message: str | None = None, *, ephemeral: bool = False, **payload
) -> discord.WebhookMessage | None:
    """Keep private slash acknowledgements separate from public results."""
    extras = getattr(interaction, "extras", {})
    completed_key = "sveklobot:defer_completed"
    pending = (
        getattr(interaction.response, "type", None)
        == discord.InteractionResponseType.deferred_channel_message
        and not extras.get(completed_key, False)
    )
    args = (message,) if message is not None else ()
    if ephemeral or not pending:
        result = await interaction.followup.send(*args, ephemeral=ephemeral, **payload)
        extras[completed_key] = True
        return result

    # Discord's first followup would inherit the private defer. Complete it
    # explicitly first so the next webhook message can really be public.
    await interaction.edit_original_response(content="Готово.")
    extras[completed_key] = True
    try:
        return await interaction.followup.send(*args, ephemeral=False, **payload)
    finally:
        try:
            await interaction.delete_original_response()
        except discord.HTTPException as error:
            logging.warning("Не удалось убрать личное подтверждение команды: %s", error)


async def send_reply(
    interaction: discord.Interaction, message: str, *, ephemeral: bool = False
) -> None:
    buffer = pending_replies.get()
    if buffer is not None and buffer.interaction is interaction:
        buffer.messages.append((message, ephemeral))
        return
    if interaction.response.is_done():
        await send_deferred_reply(interaction, message, ephemeral=ephemeral)
    else:
        await interaction.response.send_message(message, ephemeral=ephemeral)


@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction, error: app_commands.AppCommandError
) -> None:
    original = getattr(error, "original", error)
    if isinstance(original, VoiceConnectionError):
        logging.warning("Голосовое соединение прервалось перед запуском музыки: %s", original)
        await send_reply(
            interaction,
            "Голосовое соединение с Discord прервалось. Повтори выбор песни или `/play` — "
            "бот попробует переподключиться.",
            ephemeral=True,
        )
        return
    if isinstance(original, discord.NotFound) and original.code == 10062:
        # An expired interaction cannot accept another response, even an error.
        age = (discord.utils.utcnow() - interaction.created_at).total_seconds()
        logging.warning(
            "Discord отклонил просроченную команду /%s: возраст %.2f с, "
            "задержка Gateway %.2f с. Повторите команду.",
            getattr(interaction.command, "name", "unknown"),
            age,
            bot.latency,
        )
        return
    logging.error(
        "Ошибка slash-команды %s",
        getattr(interaction.command, "name", "unknown"),
        exc_info=(type(original), original, original.__traceback__),
    )
    try:
        await send_reply(
            interaction,
            "Команда завершилась с ошибкой. Подробности записаны в `bot.log`.",
            ephemeral=True,
        )
    except discord.HTTPException:
        logging.exception("Не удалось отправить сообщение об ошибке команды")


async def connect_to_user(
    interaction: discord.Interaction,
) -> discord.VoiceClient | None:
    error_message = panel_access_error(interaction)
    if error_message:
        await send_reply(interaction, error_message, ephemeral=True)
        return None

    voice_client = interaction.guild.voice_client
    created_voice = None
    try:
        if voice_client is not None and not voice_client.is_connected():
            logging.info("Восстанавливаю голосовое соединение: сервер %s", interaction.guild.id)
            state = playback.get(interaction.guild.id)
            if state is not None:
                # disconnect() stops the old audio thread. Ignore its late callback.
                state.serial += 1
                state.track = None
            await voice_client.disconnect(force=True)
            voice_client = None
            # Disconnecting can take time; never join on behalf of someone who left.
            error_message = panel_access_error(interaction)
            if error_message:
                await send_reply(interaction, error_message, ephemeral=True)
                return None
        if voice_client is None:
            voice_client = await interaction.user.voice.channel.connect(timeout=20.0, reconnect=True)
            created_voice = voice_client
        # The member may have left or changed channels during the handshake.
        error_message = panel_access_error(interaction)
        if error_message:
            if created_voice is not None:
                await created_voice.disconnect(force=True)
            await send_reply(interaction, error_message, ephemeral=True)
            return None
        if not voice_client.is_connected():
            raise VoiceConnectionError("Голосовое соединение не восстановлено после подключения.")
        update_empty_channel_timer(interaction.guild)
        return voice_client
    except (discord.ClientException, discord.HTTPException, aiohttp.ClientError,
            OSError, TimeoutError, VoiceConnectionError) as error:
        if created_voice is not None:
            try:
                await created_voice.disconnect(force=True)
            except (discord.DiscordException, aiohttp.ClientError, OSError, TimeoutError):
                logging.warning("Не удалось закрыть незавершённое голосовое подключение", exc_info=True)
        logging.warning("Не удалось установить голосовое соединение", exc_info=True)
        message = (
            "Не удалось подключиться. Проверь права бота: «Подключаться» и «Говорить»."
            if isinstance(error, discord.Forbidden) else
            "Не удалось установить голосовое соединение с Discord. "
            "Проверь интернет и доступ к Discord, затем повтори выбор песни или `/play`."
        )
        await send_reply(
            interaction,
            message,
            ephemeral=True,
        )
        return None


def finish_track(
    guild_id: int, serial: int, error: Exception | None, frames_read: int | None = None
) -> None:
    state = playback.get(guild_id)
    if state is not None and state.serial == serial:
        if (error is None and state.repeat_enabled and state.track is not None
                and frames_read != 0 and not bot._music_closing and not bot.is_closed()):
            task = asyncio.create_task(repeat_finished_track(guild_id, serial, state.track))
            repeat_tasks.add(task)
            task.add_done_callback(repeat_tasks.discard)
        else:
            state.track = None
            if error is not None or frames_read == 0:
                if state.repeat_enabled and error is None:
                    logging.warning("Повтор выключен: песня завершилась без аудио, сервер %s", guild_id)
                state.repeat_enabled = False
    if error is not None:
        logging.error("Ошибка воспроизведения: %s", error)


async def repeat_finished_track(guild_id: int, serial: int, track: Path) -> None:
    # A stop, seek or song selection can arrive before this task takes the lock.
    # Restart only the completion that still belongs to the current playback.
    guild = bot.get_guild(guild_id)
    async with player_locks.setdefault(guild_id, asyncio.Lock()):
        state = playback.get(guild_id)
        if state is None or state.serial != serial or state.track != track:
            return
        voice = guild.voice_client if guild is not None else None
        if not state.repeat_enabled or bot._music_closing or bot.is_closed():
            state.track = None
        elif voice is None or not voice.is_connected():
            state.track = None
            state.repeat_enabled = False
        elif voice.is_playing() or voice.is_paused():
            return  # Never interrupt audio that is already active.
        else:
            try:
                if not track.is_file():
                    raise FileNotFoundError("Файл повторяемой песни удалён или переименован.")
                start_track(voice, guild_id, state, track)
            except Exception:
                state.track = None
                state.repeat_enabled = False
                logging.exception("Не удалось повторить песню, сервер %s", guild_id)
    if guild is not None:
        request_player_refresh(guild)


def create_audio_source(
    track: Path, volume_percent: int, *, position: float = 0.0
) -> ProgressAudio:
    """Adjust each PCM frame before native Opus encoding in the audio thread."""
    if not math.isfinite(position) or position < 0:
        raise ValueError("Позиция песни должна быть неотрицательным конечным числом.")
    source = discord.FFmpegPCMAudio(
        str(track),
        executable=FFMPEG_EXECUTABLE,
        before_options=f"-ss {position:.6f}" if position > 0 else None,
        options="-vn",
    )
    return ProgressAudio(source, volume=volume_percent / 100, position=position)


def start_track(
    voice_client: discord.VoiceClient,
    guild_id: int,
    state: PlaybackState,
    track: Path,
    *,
    position: float = 0.0,
    paused: bool = False,
) -> None:
    if not math.isfinite(position) or position < 0:
        raise ValueError("Некорректная позиция песни.")
    if not voice_client.is_connected():
        raise VoiceConnectionError("Соединение потеряно до создания аудиопотока.")
    state.serial += 1
    serial = state.serial
    state.track = track

    if voice_client.is_playing() or voice_client.is_paused():
        voice_client.stop()

    source = (create_audio_source(track, state.volume_percent, position=position)
              if position else create_audio_source(track, state.volume_percent))
    if paused:
        source.hold_for_pause(voice_client)
    try:
        voice_client.play(
            source,
            bitrate=128,
            signal_type="music",
            after=lambda error: bot.loop.call_soon_threadsafe(
                finish_track, guild_id, serial, error, source.frames_read
            ),
        )
        if paused:
            voice_client.pause()
            source.release_startup_hold()
    except Exception as error:
        if paused:
            voice_client.stop()
        source.cleanup()
        if isinstance(error, discord.ClientException) and not voice_client.is_connected():
            raise VoiceConnectionError("Соединение потеряно при запуске аудиопотока.") from error
        raise


async def song_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    playlist = selected_playlist(interaction.guild.id if interaction.guild else None)
    current_folded = current.casefold()
    matches = [
        path.name
        for path in available_tracks(playlist)
        if current_folded in path.name.casefold() and len(path.name) <= 100
    ]
    return [app_commands.Choice(name=name, value=name) for name in matches[:25]]


def panel_access_error(interaction: discord.Interaction) -> str | None:
    if interaction.guild is None:
        return "Плеер работает только на сервере Discord."
    user_voice = getattr(interaction.user, "voice", None)
    if user_voice is None or user_voice.channel is None:
        return "Сначала зайди в голосовой канал."
    voice = interaction.guild.voice_client
    if voice is not None and voice.channel.id != user_voice.channel.id:
        return "Для управления зайди в тот же голосовой канал, что и бот."
    return None


@asynccontextmanager
async def playback_control(
    interaction: discord.Interaction, *, acknowledged: bool = False, check_voice: bool = True
):
    """Acknowledge immediately; serialize all mutating slash and panel actions."""
    if not acknowledged:
        # For components this remains a deferred message update; only slash
        # commands create a private loading message.
        await interaction.response.defer(ephemeral=True)
    guild = interaction.guild
    if guild is None:
        await send_deferred_reply(interaction, "Эта команда работает только на сервере Discord.", ephemeral=True)
        yield False
        return
    buffer = ReplyBuffer(interaction)
    async with player_locks.setdefault(guild.id, asyncio.Lock()):
        token = pending_replies.set(buffer)
        try:
            error = panel_access_error(interaction) if check_voice else None
            if error:
                buffer.messages.append((error, True))
            yield error is None
        finally:
            pending_replies.reset(token)
    # Text delivery can be slow or rate limited. The playback lock is free now.
    for message, ephemeral in buffer.messages:
        await send_deferred_reply(interaction, message, ephemeral=ephemeral)


def adjacent_track(tracks: list[Path], current: Path | None, direction: int) -> Path:
    if current in tracks:
        return tracks[(tracks.index(current) + direction) % len(tracks)]
    return tracks[-1] if direction < 0 else tracks[0]


async def handle_player_action(interaction: discord.Interaction, action: str) -> None:
    await interaction.response.defer()
    guild = interaction.guild
    if guild is None:
        await interaction.followup.send("Плеер работает только на сервере Discord.", ephemeral=True)
        return
    if action == "songs":
        playlist = selected_playlist(guild.id)
        tracks = available_tracks(playlist)
        await interaction.followup.send(
            build_song_list(playlist, tracks, numbered=False), ephemeral=True,
        )
        return
    read_only = {"refresh", "songs_previous", "songs_next", "playlists_previous", "playlists_next"}
    async with playback_control(interaction, acknowledged=True, check_voice=action not in read_only) as allowed:
        if not allowed:
            return
        state = playback.setdefault(guild.id, PlaybackState())
        voice = guild.voice_client
        if action in {"playlists_previous", "playlists_next"}:
            page, pages = playlist_page_bounds(state.playlist_page, len(available_playlists()))
            state.playlist_page = max(0, min(pages - 1, page + (1 if action == "playlists_next" else -1)))
        elif action in {"songs_previous", "songs_next"}:
            tracks = available_tracks(state.playlist)
            page, pages = song_page_bounds(state.song_page, len(tracks))
            state.song_page = max(0, min(pages - 1, page + (1 if action == "songs_next" else -1)))
        elif action in {"restart", "rewind", "forward"}:
            track = state.track
            if track is None:
                await send_reply(interaction,
                    "Сейчас ничего не играет. Сначала выбери песню.", ephemeral=True
                )
                return
            if not track.is_file():
                await send_reply(interaction,
                    "Файл текущей песни удалён или переименован. Выбери песню заново.",
                    ephemeral=True,
                )
                return
            position, paused = 0.0, False
            if action != "restart":
                source = voice.source if voice is not None else None
                if not isinstance(source, ProgressAudio) or not (voice.is_playing() or voice.is_paused()):
                    await send_reply(interaction,
                        "Сейчас нет песни, которую можно перемотать. Выбери её заново.", ephemeral=True
                    )
                    return
                serial = state.serial
                duration = await get_track_duration(track)
                # Audio can end while ffprobe is running; never revive an ended track.
                if (state.serial != serial or state.track != track or guild.voice_client is not voice
                        or voice.source is not source or not (voice.is_playing() or voice.is_paused())):
                    await send_reply(interaction,
                        "Песня уже завершилась. Выбери её заново.", ephemeral=True
                    )
                    return
                if duration is None:
                    await send_reply(interaction,
                        "Не удалось определить длительность песни для перемотки. Повтори попытку.", ephemeral=True
                    )
                    return
                current = source.position
                position = max(0.0, min(max(0.0, duration - 0.02), current + (-10 if action == "rewind" else 10)))
                if abs(position - current) < 0.001 or (action == "forward" and position < current):
                    await send_reply(interaction,
                        "Достигнуто начало песни." if action == "rewind" else "Достигнут конец песни.",
                        ephemeral=True,
                    )
                    return
                paused = voice.is_paused()
            # Keep the song locally: reconnecting clears stale playback state.
            voice = await connect_to_user(interaction)
            if voice is None:
                return
            try:
                if action == "restart":
                    start_track(voice, guild.id, state, track)
                else:
                    start_track(voice, guild.id, state, track, position=position, paused=paused)
            except Exception:
                state.track = None
                raise
        elif action == "toggle" and voice is not None and voice.is_connected() and voice.is_paused():
            voice.resume()
        elif action == "toggle" and voice is not None and voice.is_connected() and voice.is_playing():
            voice.pause()
        elif action in {"toggle", "previous", "next"}:
            tracks = available_tracks(state.playlist)
            if not tracks:
                await send_reply(interaction,
                    "В выбранном плейлисте нет музыки или его папка переименована. "
                    "Добавь файлы в нужную папку и выбери плейлист в панели.", ephemeral=True
                )
                return
            if len(tracks) == 1 and action != "toggle" and voice is not None and voice.is_connected() and (
                voice.is_playing() or voice.is_paused()
            ):
                await send_reply(interaction,"В папке пока только одна песня.", ephemeral=True)
                return
            track = adjacent_track(tracks, state.track, -1 if action == "previous" else 1)
            voice = await connect_to_user(interaction)
            if voice is None:
                return
            try:
                start_track(voice, guild.id, state, track)
            except Exception:
                state.track = None
                raise
        elif action == "loop":
            state.repeat_enabled = not state.repeat_enabled
        elif action in {"quieter", "louder"}:
            state.volume_percent = max(0, min(200, state.volume_percent + (10 if action == "louder" else -10)))
            if voice is not None and isinstance(voice.source, discord.PCMVolumeTransformer):
                voice.source.volume = state.volume_percent / 100
        elif action == "stop":
            state.serial += 1
            state.track = None
            state.repeat_enabled = False
            if voice is not None:
                voice.stop()
        if interaction.message is not None:
            player_messages[guild.id] = interaction.message
    await refresh_player(guild)


async def handle_playlist_selection(interaction: discord.Interaction, value: str) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        guild = interaction.guild
        playlists = list(available_playlists())
        playlist = next((name for name in playlists if playlist_token(name) == value), None)
        if interaction.message is not None:
            player_messages[guild.id] = interaction.message
        tracks = available_tracks(playlist) if playlist is not None else []
        if playlist is None or not tracks:
            message = (
                "Эта папка переименована или удалена. Обнови панель и выбери её новое название."
                if playlist is None else
                f"Плейлист «{playlist_label(playlist)}» пока пуст. Добавь туда музыку и выбери его снова."
            )
            await send_reply(interaction, message, ephemeral=True)
        else:
            state = playback.setdefault(guild.id, PlaybackState())
            state.playlist_page = playlists.index(playlist) // PLAYLISTS_PER_PAGE
            voice = guild.voice_client
            active = voice is not None and voice.is_connected() and (voice.is_playing() or voice.is_paused())
            if state.playlist != playlist or not active:
                voice = await connect_to_user(interaction)
                if voice is None:
                    return
                state.playlist = playlist
                state.song_page = 0
                try:
                    start_track(voice, guild.id, state, tracks[0])
                except Exception:
                    state.track = None
                    raise
    await refresh_player(guild)


async def handle_song_selection(interaction: discord.Interaction, value: str) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        guild = interaction.guild
        state = playback.setdefault(guild.id, PlaybackState())
        tracks = available_tracks(state.playlist)
        index = next((index for index, track in enumerate(tracks)
                      if song_token(state.playlist, track) == value), None)
        if interaction.message is not None:
            player_messages[guild.id] = interaction.message
        if index is None:
            await send_reply(interaction,
                "Плейлист или список песен изменился. Выбери песню в обновлённом списке.",
                ephemeral=True,
            )
        else:
            track = tracks[index]
            voice = await connect_to_user(interaction)
            if voice is None:
                return
            active = voice.is_playing() or voice.is_paused()
            if state.track != track or not active:
                try:
                    start_track(voice, guild.id, state, track)
                except Exception:
                    state.track = None
                    raise
            state.song_page = index // SONGS_PER_PAGE
    await refresh_player(guild)


class PlaylistSelect(discord.ui.Select):
    def __init__(self, selected: str = ".", *, page: int | None = None) -> None:
        names = list(available_playlists())
        requested = names.index(selected) // PLAYLISTS_PER_PAGE if page is None and selected in names else (page or 0)
        self.page, self.pages = playlist_page_bounds(requested, len(names))
        start = self.page * PLAYLISTS_PER_PAGE
        visible = names[start:start + PLAYLISTS_PER_PAGE]
        options = [discord.SelectOption(
            label=playlist_label(name)[:100], value=playlist_token(name),
            default=name == selected,
        ) for name in visible]
        if self.page > 0:
            options.insert(0, discord.SelectOption(label="◀ Предыдущая страница плейлистов", value="playlists_previous"))
        if self.page < self.pages - 1:
            options.append(discord.SelectOption(label="Следующая страница плейлистов ▶", value="playlists_next"))
        placeholder = "Выбери плейлист"
        if self.pages > 1:
            placeholder += f" • страница {self.page + 1}/{self.pages}"
        super().__init__(
            placeholder=placeholder, options=options, min_values=1, max_values=1,
            custom_id="music:playlist", row=2,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        value = self.values[0]
        if value in {"playlists_previous", "playlists_next"}:
            await handle_player_action(interaction, value)
        else:
            await handle_playlist_selection(interaction, value)


class SongSelect(discord.ui.Select):
    def __init__(self, state: PlaybackState) -> None:
        tracks = available_tracks(state.playlist)
        self.page, self.pages = song_page_bounds(state.song_page, len(tracks))
        start = self.page * SONGS_PER_PAGE
        options = [discord.SelectOption(
            label=f"{index}. {track.name}"[:100],
            value=song_token(state.playlist, track),
            default=track == state.track,
        ) for index, track in enumerate(tracks[start:start + SONGS_PER_PAGE], start=start + 1)]
        if not options:
            options = [discord.SelectOption(label="В плейлисте пока нет песен", value="empty")]
        placeholder = "Выбрать песню"
        if self.pages > 1:
            placeholder += f" • страница {self.page + 1}/{self.pages}"
        super().__init__(
            placeholder=placeholder, options=options, min_values=1, max_values=1,
            custom_id="music:song", row=3, disabled=not tracks,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        await handle_song_selection(interaction, self.values[0])


class PlayerView(discord.ui.View):
    def __init__(self, guild_id: int | None = None, *, register_all: bool = False) -> None:
        super().__init__(timeout=None)
        state = playback.get(guild_id, PlaybackState())
        self.loop_button.label = "Повтор: вкл" if state.repeat_enabled else "Повтор: выкл"
        self.loop_button.style = discord.ButtonStyle.success if state.repeat_enabled else discord.ButtonStyle.secondary
        self.add_item(PlaylistSelect(state.playlist, page=state.playlist_page))
        songs = SongSelect(state)
        self.add_item(songs)
        self.songs_previous.disabled = songs.page == 0
        self.songs_next.disabled = songs.page == songs.pages - 1
        # Register pagination callbacks even when the default playlist is empty,
        # so old panels with many songs still work after a bot restart.
        if songs.pages == 1 and not register_all:
            self.remove_item(self.songs_previous)
            self.remove_item(self.songs_next)

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item) -> None:
        await on_app_command_error(interaction, error)

    @discord.ui.button(label="Назад", emoji="⏮️", custom_id="music:previous", row=0)
    async def previous(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "previous")

    @discord.ui.button(label="−10 с", emoji="⏪", custom_id="music:rewind", row=0)
    async def rewind(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "rewind")

    @discord.ui.button(label="Пуск / Пауза", emoji="⏯️", style=discord.ButtonStyle.primary, custom_id="music:toggle", row=0)
    async def toggle(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "toggle")

    @discord.ui.button(label="+10 с", emoji="⏩", custom_id="music:forward", row=0)
    async def forward(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "forward")

    @discord.ui.button(label="Вперёд", emoji="⏭️", custom_id="music:next", row=0)
    async def next_track(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "next")

    @discord.ui.button(label="С начала", emoji="⏮️", custom_id="music:restart", row=1)
    async def restart(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "restart")

    @discord.ui.button(label="Песни", emoji="🎵", custom_id="music:songs", row=4)
    async def songs_list(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "songs")

    @discord.ui.button(label="Повтор: выкл", emoji="🔂", custom_id="music:loop", row=4)
    async def loop_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "loop")

    @discord.ui.button(label="−10%", emoji="🔉", custom_id="music:quieter", row=1)
    async def quieter(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "quieter")

    @discord.ui.button(label="+10%", emoji="🔊", custom_id="music:louder", row=1)
    async def louder(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "louder")

    @discord.ui.button(label="Стоп", emoji="⏹️", style=discord.ButtonStyle.danger, custom_id="music:stop", row=1)
    async def stop_playing(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "stop")

    @discord.ui.button(label="Обновить", emoji="🔄", custom_id="music:refresh", row=1)
    async def refresh(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "refresh")


    @discord.ui.button(label="Песни: назад", emoji="◀️", custom_id="music:songs_previous", row=4)
    async def songs_previous(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "songs_previous")

    @discord.ui.button(label="Песни: дальше", emoji="▶️", custom_id="music:songs_next", row=4)
    async def songs_next(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await handle_player_action(interaction, "songs_next")


def build_help_message() -> str:
    order = {name: index for index, name in enumerate((
        "help", "player", "join", "play", "songs", "now", "pause", "resume", "loop", "volume", "stop", "leave"
    ))}
    usage = {"play": "play <название>", "volume": "volume <0–200>"}
    commands = sorted(bot.tree.get_commands(), key=lambda command: (order.get(command.name, len(order)), command.name))
    lines = [
        f"`/{usage.get(command.name, command.name)}` — {command.description}"
        for command in commands
    ]
    return (
        "**Команды SvekloBot**\n\n" + "\n".join(lines)
        + "\n\n**Как начать**\n"
        "Зайди в голосовой канал, открой `/player` и выбери плейлист. "
        "Для конкретной песни используй список «Выбрать песню» под плейлистами "
        "или `/play` и подсказки названий.\n\n"
        "`/songs`, `/play` и переключение треков работают с выбранным плейлистом. "
        "Громкость меняется без перезапуска песни. "
        "Кнопка «С начала» запускает текущую песню заново, в том числе с паузы. "
        "Кнопки «−10 с» и «+10 с» перематывают песню, сохраняя паузу и громкость. "
        "«Повтор» или `/loop` включает повтор одной песни; «Стоп» отключает его.\n\n"
        "Управлять музыкой могут участники голосового канала бота; "
        "права администратора не нужны. Справка доступна без входа в голосовой канал. "
        "Если в голосовом канале нет людей 5 минут, бот выходит. "
        "Возвращение человека отменяет таймер."
    )


@bot.tree.command(name="help", description="Показать все команды бота и краткую инструкцию")
async def help_command(interaction: discord.Interaction) -> None:
    await interaction.response.send_message(build_help_message(), ephemeral=True)


@bot.tree.command(name="player", description="Открыть музыкальную панель с кнопками")
@app_commands.guild_only()
async def player(interaction: discord.Interaction) -> None:
    await interaction.response.defer(ephemeral=True)
    if interaction.guild is None:
        await send_deferred_reply(interaction, "Плеер работает только на сервере Discord.", ephemeral=True)
        return
    if not interaction.app_permissions.embed_links:
        await send_deferred_reply(interaction,
            "Для панели дай боту право «Встраивать ссылки» в этом текстовом канале.", ephemeral=True
        )
        return
    message = await send_deferred_reply(interaction,
        embed=await player_embed(interaction.guild), view=PlayerView(interaction.guild.id), wait=True,
    )
    # Use a normal bot-authenticated message for edits after the interaction
    # token expires (15 minutes). Only the newest panel is refreshed per guild.
    player_messages[interaction.guild.id] = interaction.channel.get_partial_message(message.id)


@bot.tree.command(name="join", description="Подключить бота к твоему голосовому каналу")
async def join(interaction: discord.Interaction) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        voice_client = await connect_to_user(interaction)
        if voice_client is not None:
            await send_reply(interaction,
                f"Подключился к каналу **{voice_client.channel}**."
            )


@bot.tree.command(name="songs", description="Показать музыку из локальной папки")
async def songs(interaction: discord.Interaction) -> None:
    playlist = selected_playlist(interaction.guild.id if interaction.guild else None)
    tracks = available_tracks(playlist)
    if not tracks:
        await interaction.response.send_message(
            "В выбранном плейлисте нет музыки или его папка переименована. "
            "Положи файлы в нужную папку и выбери плейлист в `/player`.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(build_song_list(playlist, tracks))


@bot.tree.command(name="play", description="Включить песню из локальной папки")
@app_commands.describe(song="Имя файла или часть названия")
@app_commands.autocomplete(song=song_autocomplete)
async def play(interaction: discord.Interaction, song: str) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        playlist = selected_playlist(interaction.guild.id)
        track, error_message = find_track(song, playlist)
        if track is None:
            await send_reply(interaction,error_message, ephemeral=True)
            return
        voice_client = await connect_to_user(interaction)
        if voice_client is None:
            return

        state = playback.setdefault(interaction.guild.id, PlaybackState())
        try:
            start_track(voice_client, interaction.guild.id, state, track)
        except (discord.ClientException, discord.opus.OpusError, discord.opus.OpusNotLoaded, OSError) as error:
            state.track = None
            logging.exception("Не удалось запустить файл", exc_info=error)
            await send_reply(interaction,
                "Не удалось запустить файл. Проверь FFmpeg; подробности в `bot.log`.", ephemeral=True
            )
            return

        await send_reply(interaction,f"▶️ Сейчас играет: **{track.name}**")


@bot.tree.command(name="now", description="Показать текущую песню")
async def now(interaction: discord.Interaction) -> None:
    if interaction.guild is None:
        await interaction.response.send_message(
            "Эта команда работает только на сервере Discord.", ephemeral=True
        )
        return

    state = playback.get(interaction.guild.id)
    if state is None or state.track is None:
        await interaction.response.send_message("Сейчас ничего не играет.")
        return
    await interaction.response.send_message(f"🎵 Сейчас играет: **{state.track.name}**")


@bot.tree.command(name="pause", description="Поставить музыку на паузу")
async def pause(interaction: discord.Interaction) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        voice_client = interaction.guild.voice_client
        if voice_client is None or not voice_client.is_playing():
            await send_reply(interaction,"Сейчас ничего не играет.", ephemeral=True)
            return
        voice_client.pause()
        await send_reply(interaction,"⏸️ Пауза.")


@bot.tree.command(name="resume", description="Продолжить воспроизведение")
async def resume(interaction: discord.Interaction) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        voice_client = interaction.guild.voice_client
        if voice_client is None or not voice_client.is_paused():
            await send_reply(interaction,"Музыка не стоит на паузе.", ephemeral=True)
            return
        voice_client.resume()
        await send_reply(interaction,"▶️ Продолжаю.")


@bot.tree.command(name="loop", description="Включить или выключить повтор одной песни")
async def loop_command(interaction: discord.Interaction) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        state = playback.setdefault(interaction.guild.id, PlaybackState())
        state.repeat_enabled = not state.repeat_enabled
        await send_reply(interaction,
            "🔂 Повтор одной песни включён." if state.repeat_enabled else "Повтор выключен."
        )
    await refresh_player(interaction.guild)


@bot.tree.command(name="volume", description="Изменить громкость от 0 до 200 процентов")
@app_commands.describe(percent="Громкость от 0 до 200")
async def volume(interaction: discord.Interaction, percent: app_commands.Range[int, 0, 200]) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        state = playback.setdefault(interaction.guild.id, PlaybackState())
        state.volume_percent = percent
        voice_client = interaction.guild.voice_client
        if voice_client is None or state.track is None or not (
            voice_client.is_playing() or voice_client.is_paused()
        ):
            await send_reply(interaction,
                f"🔊 Громкость **{percent}%** применится к следующей песне."
            )
            return

        source = voice_client.source
        if not isinstance(source, discord.PCMVolumeTransformer):
            await send_reply(interaction,
                "У текущего потока нет регулировки громкости. "
                "Новое значение применится к следующей песне.", ephemeral=True
            )
            return

        source.volume = percent / 100
        await send_reply(interaction,f"🔊 Громкость: **{percent}%**.")


@bot.tree.command(name="stop", description="Остановить музыку")
async def stop(interaction: discord.Interaction) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        state = playback.setdefault(interaction.guild.id, PlaybackState())
        state.serial += 1
        state.track = None
        state.repeat_enabled = False
        voice_client = interaction.guild.voice_client
        if voice_client is not None and (voice_client.is_playing() or voice_client.is_paused()):
            voice_client.stop()
        await send_reply(interaction,"⏹️ Воспроизведение остановлено.")


@bot.tree.command(name="leave", description="Отключить бота от голосового канала")
async def leave(interaction: discord.Interaction) -> None:
    async with playback_control(interaction) as allowed:
        if not allowed:
            return
        state = playback.setdefault(interaction.guild.id, PlaybackState())
        state.serial += 1
        state.track = None
        state.repeat_enabled = False
        cancel_empty_channel_timer(interaction.guild.id)
        if interaction.guild.voice_client is None:
            await send_reply(interaction,
                "Бот не подключён к голосовому каналу.", ephemeral=True
            )
            return
        await interaction.guild.voice_client.disconnect(force=True)
        await send_reply(interaction,"👋 Отключился от голосового канала.")


def main() -> None:
    token = os.getenv("DISCORD_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "Не найден DISCORD_TOKEN. Скопируй .env.example в .env и вставь токен."
        )
    if FFMPEG_EXECUTABLE is None:
        raise SystemExit(
            "FFmpeg не найден. Установи его или укажи путь к ffmpeg.exe в FFMPEG_PATH."
        )

    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(BASE_DIR / "bot.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info(
        "Запуск бота: PID=%s, Python=%s, discord.py=%s, аудио=PCM + native Opus, "
        "подтверждение /pause до обработки",
        os.getpid(),
        sys.version.split()[0],
        discord.__version__,
    )
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
