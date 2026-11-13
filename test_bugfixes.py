"""Regression checks for the 2026-11-13 fixes: paths, voice loss, panel refresh, logging."""
import io
import logging
import os
import struct
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import discord

import bot
import test_regressions as helpers


class ExecutableResolutionTests(unittest.TestCase):
    def test_relative_ffmpeg_path_is_resolved_from_bot_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "ffmpeg" / "bin").mkdir(parents=True)
            exe = base / "ffmpeg" / "bin" / "ffmpeg.exe"
            exe.write_bytes(b"stub")
            with patch.dict(bot.os.environ, {"DSH_TEST_FFMPEG": "ffmpeg/bin/ffmpeg.exe"}, clear=False):
                self.assertEqual(
                    bot.configured_executable("DSH_TEST_FFMPEG", base_dir=base),
                    str(exe.resolve()),
                )

    def test_bare_file_name_in_bot_folder_is_preferred_over_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            exe = base / "ffmpeg"
            exe.write_bytes(b"stub")
            with patch.dict(bot.os.environ, {"DSH_TEST_FFMPEG": "ffmpeg"}, clear=False), \
                 patch.object(bot.shutil, "which", side_effect=AssertionError("PATH lookup is not expected")):
                self.assertEqual(
                    bot.configured_executable("DSH_TEST_FFMPEG", base_dir=base),
                    str(exe.resolve()),
                )

    def test_bare_name_without_local_file_uses_path_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.dict(bot.os.environ, {"DSH_TEST_FFMPEG": "custom-ffmpeg"}, clear=False), \
                 patch.object(bot.shutil, "which", return_value="/opt/bin/custom-ffmpeg") as which:
                self.assertEqual(
                    bot.configured_executable("DSH_TEST_FFMPEG", base_dir=base),
                    "/opt/bin/custom-ffmpeg",
                )
                which.assert_called_once_with("custom-ffmpeg")

    def test_empty_variable_falls_back_to_default_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.dict(bot.os.environ, {"DSH_TEST_FFMPEG": ""}, clear=False), \
                 patch.object(bot.shutil, "which", return_value="ffmpeg") as which:
                self.assertEqual(
                    bot.configured_executable("DSH_TEST_FFMPEG", "ffmpeg", base_dir=base),
                    "ffmpeg",
                )
                which.assert_called_once_with("ffmpeg")

    def test_absolute_path_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "ffmpeg.exe"
            exe.write_bytes(b"stub")
            with patch.dict(bot.os.environ, {"DSH_TEST_FFMPEG": str(exe)}, clear=False):
                self.assertEqual(
                    bot.configured_executable("DSH_TEST_FFMPEG", base_dir=Path(directory) / "ignored"),
                    str(exe.resolve()),
                )


class StartupValidationTests(unittest.TestCase):
    def test_missing_ffmpeg_explains_the_fix(self):
        message = bot.ffmpeg_unavailable_reason(None)
        self.assertIn("FFmpeg", message)
        self.assertIn("FFMPEG_PATH", message)

    def test_missing_file_reports_the_bad_path(self):
        missing = Path(tempfile.gettempdir()) / "dsh-not-here" / "ffmpeg.exe"
        message = bot.ffmpeg_unavailable_reason(str(missing))
        self.assertIn(str(missing), message)
        self.assertIn("FFMPEG_PATH", message)

    def test_existing_executable_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "ffmpeg.exe"
            exe.write_bytes(b"stub")
            if os.name != "nt":
                exe.chmod(0o755)
            self.assertIsNone(bot.ffmpeg_unavailable_reason(str(exe)))


class FfprobeResolutionTests(unittest.TestCase):
    probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"

    def test_probe_next_to_ffmpeg_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            ffmpeg = base / "ffmpeg.exe"
            ffmpeg.write_bytes(b"stub")
            probe = base / self.probe_name
            probe.write_bytes(b"stub")
            self.assertEqual(bot.resolve_ffprobe(str(ffmpeg)), probe)

    def test_path_lookup_is_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "ffmpeg-only").mkdir()
            ffmpeg = base / "ffmpeg-only" / "ffmpeg.exe"
            ffmpeg.write_bytes(b"stub")
            probe = base / self.probe_name
            probe.write_bytes(b"stub")
            with patch.object(bot.shutil, "which", return_value=str(probe)):
                self.assertEqual(bot.resolve_ffprobe(str(ffmpeg)), probe)

    def test_missing_probe_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            ffmpeg = Path(directory) / "ffmpeg.exe"
            ffmpeg.write_bytes(b"stub")
            with patch.object(bot.shutil, "which", return_value=None):
                self.assertIsNone(bot.resolve_ffprobe(str(ffmpeg)))


class DurationMessageTests(unittest.TestCase):
    def test_missing_ffprobe_is_explained_for_seeking(self):
        with patch.object(bot, "FFPROBE_EXECUTABLE", None):
            message = bot.seek_duration_error()
        self.assertIn("ffprobe", message.lower())
        self.assertIn("перемот", message.lower())

    def test_other_probe_failure_keeps_the_retry_hint(self):
        with patch.object(bot, "FFPROBE_EXECUTABLE", Path("ffprobe")):
            self.assertIn("Повтори", bot.seek_duration_error())

    def test_panel_footer_mentions_missing_ffprobe(self):
        item = helpers.make_interaction()
        item.guild.voice_client.channel.mention = "<#1>"
        with patch.object(bot, "FFPROBE_EXECUTABLE", None):
            embed = bot.build_player_embed(item.guild, None)
        self.assertIn("ffprobe", (embed.footer.text or "").lower())


class SeekWithoutFfprobeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.track = Path(self.directory.name) / "song.mp3"
        self.track.write_bytes(b"fixture")
        self.item = helpers.make_interaction()
        self.voice = self.item.guild.voice_client
        self.voice.channel.mention = "<#1>"
        self.voice.is_connected.return_value = True
        self.voice.is_playing.return_value = True
        self.voice.is_paused.return_value = False
        self.source = bot.ProgressAudio(
            discord.PCMAudio(io.BytesIO(struct.pack("<h", 1) * 3840)), volume=1, position=10
        )
        self.voice.source = self.source
        bot.playback[123] = bot.PlaybackState(serial=3, track=self.track, playlist=".")

    def tearDown(self):
        self.source.cleanup()
        self.directory.cleanup()
        bot.playback.clear()
        bot.player_messages.clear()
        bot.player_locks.clear()
        bot.panel_update_locks.clear()

    def run_local(self, coroutine):
        try:
            coroutine.send(None)
        except StopIteration:
            return
        finally:
            coroutine.close()
        self.fail("Unexpected real I/O")

    def test_seek_reports_missing_probe_instead_of_generic_retry(self):
        async def no_duration(track):
            return None

        with patch.object(bot, "get_track_duration", side_effect=no_duration), \
             patch.object(bot, "FFPROBE_EXECUTABLE", None), \
             patch.object(bot, "start_track") as start:
            self.run_local(bot.handle_player_action(self.item, "forward"))
        start.assert_not_called()
        message = self.item.followup.send.call_args.args[0]
        self.assertIn("ffprobe", message.lower())


class VoiceLossTests(unittest.IsolatedAsyncioTestCase):
    loop_factory = staticmethod(helpers.OfflineLoop)

    async def asyncSetUp(self):
        self.item = helpers.make_interaction()
        self.item.guild.voice_client.channel.mention = "<#1>"
        self.voice = self.item.guild.voice_client
        self.state = bot.PlaybackState(serial=5, track=Path("1/b.mp3"), playlist="1")
        bot.playback[123] = self.state
        self.closing_patch = patch.object(bot.bot, "_music_closing", False)
        self.closing_patch.start()

    async def asyncTearDown(self):
        self.closing_patch.stop()
        bot.playback.clear()
        bot.player_messages.clear()
        bot.player_locks.clear()
        bot.panel_update_locks.clear()
        for countdown in bot.empty_channel_timers.values():
            if countdown.task is not None:
                countdown.task.cancel()
        bot.empty_channel_timers.clear()

    async def test_now_does_not_claim_playback_without_voice_client(self):
        self.item.guild.voice_client = None
        await bot.now.callback(self.item)
        message = self.item.response.send_message.call_args.args[0]
        self.assertNotIn("Сейчас играет", message)
        self.assertIn("потеряно", message)

    async def test_now_reports_reconnecting_voice(self):
        self.voice.is_connected.return_value = False
        await bot.now.callback(self.item)
        message = self.item.response.send_message.call_args.args[0]
        self.assertIn("восстанавливается", message)
        self.assertIn("b.mp3", message)

    async def test_now_reports_current_song_when_connected(self):
        self.voice.is_connected.return_value = True
        await bot.now.callback(self.item)
        self.assertIn("Сейчас играет", self.item.response.send_message.call_args.args[0])

    async def test_now_reports_nothing_when_selected_song_is_not_playing(self):
        # E.g. /join succeeded after a failed reconnect: the song is remembered
        # for the next "Пуск / Пауза", but nothing is playing yet.
        self.voice.is_connected.return_value = True
        self.voice.is_playing.return_value = False
        self.voice.is_paused.return_value = False
        await bot.now.callback(self.item)
        self.assertEqual(
            self.item.response.send_message.call_args.args[0], "Сейчас ничего не играет."
        )

    async def test_toggle_after_lost_connection_starts_current_not_next(self):
        self.voice.is_connected.return_value = False
        self.voice.is_playing.return_value = False
        self.voice.is_paused.return_value = False
        tracks = [Path("1/a.mp3"), Path("1/b.mp3"), Path("1/c.mp3")]
        with patch.object(bot, "available_tracks", return_value=tracks), \
             patch.object(bot, "connect_to_user", new=AsyncMock(return_value=self.voice)), \
             patch.object(bot, "start_track") as start, \
             patch.object(bot, "refresh_player", new_callable=AsyncMock):
            await bot.handle_player_action(self.item, "toggle")
        self.assertEqual(start.call_args.args[-1], Path("1/b.mp3"))

    async def test_toggle_falls_back_to_first_when_track_is_from_another_playlist(self):
        self.voice.is_connected.return_value = False
        self.voice.is_playing.return_value = False
        self.voice.is_paused.return_value = False
        self.state.track = Path("2/other.mp3")
        tracks = [Path("1/a.mp3"), Path("1/b.mp3")]
        with patch.object(bot, "available_tracks", return_value=tracks), \
             patch.object(bot, "connect_to_user", new=AsyncMock(return_value=self.voice)), \
             patch.object(bot, "start_track") as start, \
             patch.object(bot, "refresh_player", new_callable=AsyncMock):
            await bot.handle_player_action(self.item, "toggle")
        self.assertEqual(start.call_args.args[-1], Path("1/a.mp3"))

    async def test_panel_embed_without_channel_has_no_channel_field(self):
        self.voice.is_connected.return_value = True
        self.voice.is_playing.return_value = True
        self.voice.channel = None
        embed = bot.build_player_embed(self.item.guild, 100)
        self.assertNotIn("Канал", [field.name for field in embed.fields])

    async def test_empty_channel_timer_ignores_unknown_channel(self):
        self.voice.channel = None
        bot.update_empty_channel_timer(self.item.guild)
        self.assertNotIn(123, bot.empty_channel_timers)

    async def test_empty_channel_expiry_ignores_unknown_channel(self):
        self.voice.channel = None
        countdown = bot.EmptyChannelCountdown(self.voice, 1)
        bot.empty_channel_timers[123] = countdown
        with patch.object(bot, "EMPTY_CHANNEL_TIMEOUT", 0):
            await bot.leave_empty_channel_after_timeout(self.item.guild, countdown)
        self.voice.disconnect.assert_not_awaited()
        self.assertNotIn(123, bot.empty_channel_timers)


class ReconnectRetryTests(unittest.IsolatedAsyncioTestCase):
    """A failed reconnect must keep the current song for the next attempt."""

    loop_factory = staticmethod(helpers.OfflineLoop)

    async def asyncSetUp(self):
        self.item = helpers.make_interaction()
        self.stale = self.item.guild.voice_client
        self.stale.channel = SimpleNamespace(id=1)
        self.stale.is_connected.return_value = False
        self.stale.is_playing.return_value = False
        self.stale.is_paused.return_value = False

        async def disconnect_stale(**kwargs):
            self.item.guild.voice_client = None

        self.stale.disconnect = AsyncMock(side_effect=disconnect_stale)
        self.user_channel = self.item.user.voice.channel
        self.fresh = Mock()
        self.fresh.channel = self.user_channel
        self.fresh.is_connected.return_value = True
        self.fresh.is_playing.return_value = False
        self.fresh.is_paused.return_value = False
        self.fresh.disconnect = AsyncMock()
        self.attempts = 0

        async def connect(**kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("the first handshake fails")
            self.item.guild.voice_client = self.fresh
            return self.fresh

        self.user_channel.connect = AsyncMock(side_effect=connect)
        self.state = bot.PlaybackState(serial=5, track=Path("1/b.mp3"), playlist="1")
        bot.playback[123] = self.state
        self.tracks = [Path("1/a.mp3"), Path("1/b.mp3"), Path("1/c.mp3")]
        self.started = []
        self.patches = [
            patch.object(bot, "available_tracks", return_value=self.tracks),
            patch.object(bot, "start_track", side_effect=self.record_start),
            patch.object(bot, "refresh_player", new_callable=AsyncMock),
            patch.object(bot.bot, "_music_closing", False),
        ]
        for patcher in self.patches:
            patcher.start()

    def record_start(self, voice, guild_id, state, track, **kwargs):
        state.track = track
        self.started.append(track)

    async def asyncTearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        bot.playback.clear()
        bot.player_messages.clear()
        bot.player_locks.clear()
        bot.panel_update_locks.clear()
        for countdown in bot.empty_channel_timers.values():
            if countdown.task is not None:
                countdown.task.cancel()
        bot.empty_channel_timers.clear()

    async def fail_first_toggle(self):
        with self.assertLogs(level="WARNING"):
            await bot.handle_player_action(self.item, "toggle")
        self.assertEqual(self.attempts, 1)
        self.assertEqual(self.state.track, Path("1/b.mp3"))
        self.assertGreater(self.state.serial, 5)
        self.assertEqual(self.started, [])

    async def test_failed_reconnect_keeps_current_song_for_the_next_try(self):
        await self.fail_first_toggle()
        # An old callback from the invalidated playback must not erase it.
        bot.finish_track(123, 5, None, 50)
        self.assertEqual(self.state.track, Path("1/b.mp3"))
        await bot.handle_player_action(self.item, "toggle")
        self.assertEqual(self.attempts, 2)
        self.assertEqual(self.started, [Path("1/b.mp3")])
        self.assertEqual(self.state.track, Path("1/b.mp3"))

    async def test_stop_clears_song_preserved_after_failed_reconnect(self):
        await self.fail_first_toggle()
        await bot.stop.callback(self.item)
        self.assertIsNone(self.state.track)

    async def test_leave_clears_song_preserved_after_failed_reconnect(self):
        await self.fail_first_toggle()
        await bot.leave.callback(self.item)
        self.assertIsNone(self.state.track)


class ImmediatePanelRefreshTests(unittest.IsolatedAsyncioTestCase):
    loop_factory = staticmethod(helpers.OfflineLoop)

    async def asyncSetUp(self):
        bot.playback[123] = bot.PlaybackState(track=Path("track.mp3"))

    async def asyncTearDown(self):
        bot.playback.clear()
        bot.player_messages.clear()
        bot.player_locks.clear()
        bot.panel_update_locks.clear()

    async def run_with_refresh(self, coroutine):
        observed = []

        def record_refresh(guild):
            lock = bot.player_locks.get(123)
            observed.append(lock is None or not lock.locked())
            observed.append(guild.id)

        with patch.object(bot, "request_player_refresh", side_effect=record_refresh):
            await coroutine
        return observed

    async def test_pause_refreshes_panel_after_lock_is_released(self):
        item = helpers.make_interaction()
        item.guild.voice_client.is_playing.return_value = True
        self.assertEqual(await self.run_with_refresh(bot.pause.callback(item)), [True, 123])
        item.guild.voice_client.pause.assert_called_once()

    async def test_resume_refreshes_panel_after_lock_is_released(self):
        item = helpers.make_interaction()
        item.guild.voice_client.is_paused.return_value = True
        self.assertEqual(await self.run_with_refresh(bot.resume.callback(item)), [True, 123])
        item.guild.voice_client.resume.assert_called_once()

    async def test_volume_refreshes_panel_after_lock_is_released(self):
        item = helpers.make_interaction()
        item.guild.voice_client.is_playing.return_value = True
        self.assertEqual(await self.run_with_refresh(bot.volume.callback(item, 35)), [True, 123])
        self.assertEqual(bot.playback[123].volume_percent, 35)

    async def test_stop_refreshes_panel_after_lock_is_released(self):
        item = helpers.make_interaction()
        item.guild.voice_client.is_playing.return_value = True
        self.assertEqual(await self.run_with_refresh(bot.stop.callback(item)), [True, 123])
        self.assertIsNone(bot.playback[123].track)

    async def test_rejected_command_does_not_refresh_panel(self):
        item = helpers.make_interaction()
        item.user.voice = None
        with patch.object(bot, "request_player_refresh") as refresh:
            await bot.pause.callback(item)
        refresh.assert_not_called()

    async def test_play_refreshes_panel_after_starting(self):
        item = helpers.make_interaction()
        with patch.object(bot, "find_track", return_value=(Path("track.mp3"), None)), \
             patch.object(bot, "connect_to_user", new=AsyncMock(return_value=item.guild.voice_client)), \
             patch.object(bot, "start_track") as start:
            observed = await self.run_with_refresh(bot.play.callback(item, "track"))
        self.assertEqual(observed, [True, 123])
        start.assert_called_once()


class LogRotationTests(unittest.TestCase):
    def test_bot_log_rotates_and_keeps_bounded_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bot.log"
            with patch.object(bot, "LOG_MAX_BYTES", 200), patch.object(bot, "LOG_BACKUP_COUNT", 2):
                handlers = bot.build_log_handlers(path)
                try:
                    file_handlers = [handler for handler in handlers if isinstance(handler, RotatingFileHandler)]
                    self.assertEqual(len(file_handlers), 1)
                    file_handler = file_handlers[0]
                    self.assertEqual((file_handler.maxBytes, file_handler.backupCount), (200, 2))
                    self.assertTrue(any(type(handler) is logging.StreamHandler for handler in handlers))
                    record = logging.LogRecord("test", logging.INFO, __file__, 1, "x" * 500, None, None)
                    # Python never rotates an empty file, so the first record just
                    # fills it and the second one triggers the rollover.
                    file_handler.emit(record)
                    file_handler.emit(record)
                    self.assertTrue(path.with_name("bot.log.1").is_file())
                    self.assertFalse(path.with_name("bot.log.2").is_file())
                finally:
                    for handler in handlers:
                        handler.close()


if __name__ == "__main__":
    unittest.main()
