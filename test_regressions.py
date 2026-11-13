"""Regression checks for voice access, concurrent actions, and decoder errors."""
import asyncio
import io
import selectors
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import discord
from discord.player import AudioPlayer
import bot


def make_interaction(guild=None):
    if guild is None:
        voice = Mock()
        voice.channel = SimpleNamespace(id=1)
        voice.is_playing.return_value = True
        voice.is_paused.return_value = False
        voice.disconnect = AsyncMock()
        guild = SimpleNamespace(id=123, voice_client=voice)
    channel = SimpleNamespace(id=1)
    return SimpleNamespace(
        guild=guild, user=SimpleNamespace(voice=SimpleNamespace(channel=channel)),
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock(), is_done=Mock(return_value=False)),
        followup=SimpleNamespace(send=AsyncMock()), message=None,
    )


class OfflineSelector(selectors.BaseSelector):
    """No sockets or real I/O: only asyncio tasks, events and mocks are used."""
    def register(self, *args, **kwargs):
        raise AssertionError("Unexpected real I/O in a regression test")

    def unregister(self, *args, **kwargs):
        raise AssertionError("Unexpected real I/O in a regression test")

    def select(self, timeout=None):
        return []

    def get_map(self):
        return {}


class OfflineLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(selector=OfflineSelector())

    def _make_self_pipe(self):
        pass

    def _close_self_pipe(self):
        pass

    def _write_to_self(self):
        pass


class RegressionTests(unittest.TestCase):
    def tearDown(self):
        bot.playback.clear()
        bot.player_messages.clear()
        bot.player_locks.clear()

    def run_local(self, coroutine):
        try:
            coroutine.send(None)
        except StopIteration as result:
            return result.value
        finally:
            coroutine.close()
        self.fail("Unexpected real I/O")

    def test_all_mutating_slash_commands_reject_users_outside_voice(self):
        for name in ("join", "play", "pause", "resume", "volume", "stop", "leave"):
            for user_voice in (None, SimpleNamespace(channel=SimpleNamespace(id=2))):
                with self.subTest(command=name, user_voice=user_voice):
                    item = make_interaction()
                    item.user.voice = user_voice
                    state = bot.PlaybackState(serial=3, track=Path("playing.mp3"), volume_percent=80)
                    bot.playback[123] = state
                    args = ("song",) if name == "play" else (50,) if name == "volume" else ()
                    with patch.object(bot, "connect_to_user", new_callable=AsyncMock) as connect, \
                         patch.object(bot, "start_track") as start:
                        self.run_local(getattr(bot, name).callback(item, *args))
                        connect.assert_not_awaited()
                        start.assert_not_called()
                    voice = item.guild.voice_client
                    for method in (voice.stop, voice.pause, voice.resume, voice.disconnect):
                        method.assert_not_called()
                    self.assertEqual((state.serial, state.track, state.volume_percent), (3, Path("playing.mp3"), 80))
                    item.response.defer.assert_awaited_once()
                    item.followup.send.assert_awaited_once()

    def test_connect_does_not_move_bot_from_another_channel(self):
        item = make_interaction()
        item.user.voice.channel = SimpleNamespace(id=2)
        result = self.run_local(bot.connect_to_user(item))
        self.assertIsNone(result)
        item.guild.voice_client.move_to.assert_not_called()

    def test_decoder_failure_reaches_audio_player_callback_and_log(self):
        failure = RuntimeError("FFmpeg decoder failed: regression fixture")
        class FailingAudio(discord.AudioSource):
            _current_error = failure
            def read(self):
                return b""
        source = bot.ProgressAudio(FailingAudio(), volume=1)
        bot.playback[123] = bot.PlaybackState(serial=7, track=Path("broken.mp3"))
        client = Mock()
        client.is_connected.return_value = False
        player = AudioPlayer(source, client, after=lambda error: bot.finish_track(123, 7, error))
        player._speak = Mock()
        with self.assertLogs(level="ERROR") as captured:
            player.run()
        self.assertIs(player._current_error, failure)
        self.assertIsNone(bot.playback[123].track)
        self.assertTrue(any(str(failure) in line for line in captured.output))

    def test_normal_end_of_audio_is_not_an_error(self):
        source = bot.ProgressAudio(discord.PCMAudio(io.BytesIO()), volume=1)
        try:
            self.assertEqual(source.read(), b"")
        finally:
            source.cleanup()

    def test_real_invalid_audio_raises_decoder_error(self):
        if not bot.FFMPEG_EXECUTABLE:
            self.skipTest("Needs FFmpeg")
        with tempfile.TemporaryDirectory(prefix=".decoder-test-", dir=Path(__file__).resolve().parent) as directory:
            track = Path(directory) / "broken.mp3"
            track.write_bytes(b"deliberately invalid audio fixture")
            source = bot.create_audio_source(track, 100)
            try:
                with self.assertRaises(discord.FFmpegProcessError):
                    source.read()
            finally:
                source.cleanup()


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    loop_factory = staticmethod(OfflineLoop)

    async def asyncSetUp(self):
        self.item = make_interaction()
        self.voice = self.item.guild.voice_client
        self.item.guild.voice_client = None
        self.state = bot.PlaybackState(playlist="1")
        bot.playback[123] = self.state
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.starts = []
        self.patches = [
            patch.object(bot, "connect_to_user", new=self.delayed_connect),
            patch.object(bot, "find_track", side_effect=lambda song, playlist: (Path(playlist) / "song.mp3", None)),
            patch.object(bot, "available_playlists", return_value={".": Path("music"), "1": Path("1"), "2": Path("2")}),
            patch.object(bot, "available_tracks", side_effect=lambda playlist=".": [Path(playlist) / "song.mp3"]),
            patch.object(bot, "start_track", side_effect=self.record_start),
            patch.object(bot, "refresh_player", new_callable=AsyncMock),
        ]
        for patcher in self.patches:
            patcher.start()

    async def asyncTearDown(self):
        self.release.set()
        for patcher in reversed(self.patches):
            patcher.stop()
        bot.playback.clear()
        bot.player_messages.clear()
        bot.player_locks.clear()

    async def delayed_connect(self, item):
        item.guild.voice_client = self.voice
        self.entered.set()
        await self.release.wait()
        return self.voice

    def record_start(self, voice, guild_id, state, track):
        state.serial += 1
        state.track = track
        self.starts.append((state.playlist, str(track)))

    async def test_playlist_change_waits_for_pending_play_then_wins(self):
        first = asyncio.create_task(bot.play.callback(self.item, "song"))
        await asyncio.wait_for(self.entered.wait(), timeout=1)
        second_item = make_interaction(self.item.guild)
        second = asyncio.create_task(bot.handle_playlist_selection(second_item, bot.playlist_token("2")))
        await asyncio.sleep(0)
        second_item.response.defer.assert_awaited_once()
        self.assertFalse(second.done())
        self.assertEqual(self.state.playlist, "1")
        self.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.assertEqual(self.starts, [("1", str(Path("1/song.mp3"))), ("2", str(Path("2/song.mp3")))])
        self.assertEqual((self.state.playlist, self.state.track), ("2", Path("2/song.mp3")))

    async def test_play_resolves_current_playlist_after_waiting_for_selector(self):
        first = asyncio.create_task(bot.handle_playlist_selection(self.item, bot.playlist_token("2")))
        await asyncio.wait_for(self.entered.wait(), timeout=1)
        second = asyncio.create_task(bot.play.callback(make_interaction(self.item.guild), "song"))
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        self.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.assertEqual(self.starts, [("2", str(Path("2/song.mp3"))), ("2", str(Path("2/song.mp3")))])
        self.assertEqual(self.state.track, Path("2/song.mp3"))

    async def test_stop_queued_during_connection_cannot_be_overwritten_by_play(self):
        first = asyncio.create_task(bot.play.callback(self.item, "song"))
        await asyncio.wait_for(self.entered.wait(), timeout=1)
        second = asyncio.create_task(bot.stop.callback(make_interaction(self.item.guild)))
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        self.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.voice.stop.assert_called_once()
        self.assertIsNone(self.state.track)

    async def test_access_is_checked_again_after_waiting_for_lock(self):
        self.item.guild.voice_client = self.voice
        lock = bot.player_locks.setdefault(123, asyncio.Lock())
        await lock.acquire()
        pending = asyncio.create_task(bot.pause.callback(self.item))
        await asyncio.sleep(0)
        self.item.user.voice = None
        lock.release()
        await asyncio.wait_for(pending, timeout=1)
        self.voice.pause.assert_not_called()
        self.item.followup.send.assert_awaited_once()

    async def test_song_selection_waits_for_pending_play(self):
        target = Path("1/other.mp3")
        with patch.object(bot, "available_tracks", return_value=[target]):
            first = asyncio.create_task(bot.play.callback(self.item, "song"))
            await asyncio.wait_for(self.entered.wait(), timeout=1)
            second_item = make_interaction(self.item.guild)
            second = asyncio.create_task(bot.handle_song_selection(second_item, bot.song_token("1", target)))
            await asyncio.sleep(0)
            second_item.response.defer.assert_awaited_once()
            self.assertFalse(second.done())
            self.release.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.assertEqual(self.starts, [("1", str(Path("1/song.mp3"))), ("1", str(target))])
        self.assertEqual(self.state.track, target)

    async def test_stale_song_selection_is_rejected_after_pending_playlist_change(self):
        first = asyncio.create_task(bot.handle_playlist_selection(self.item, bot.playlist_token("2")))
        await asyncio.wait_for(self.entered.wait(), timeout=1)
        second_item = make_interaction(self.item.guild)
        second = asyncio.create_task(bot.handle_song_selection(
            second_item, bot.song_token("1", Path("1/song.mp3"))
        ))
        await asyncio.sleep(0)
        self.assertFalse(second.done())
        self.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.assertEqual(self.starts, [("2", str(Path("2/song.mp3")))])
        self.assertEqual((self.state.playlist, self.state.track), ("2", Path("2/song.mp3")))
        self.assertIn("изменился", second_item.followup.send.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
