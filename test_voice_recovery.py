"""Recover stale voice connections without using Discord or decoding audio."""
import asyncio
import socket
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import discord

import bot
import test_regressions as helpers


class VoiceRecoveryTests(unittest.IsolatedAsyncioTestCase):
    loop_factory = staticmethod(helpers.OfflineLoop)

    async def asyncSetUp(self):
        self.item = helpers.make_interaction()
        self.item.response.is_done.return_value = True
        self.stale = self.item.guild.voice_client
        self.stale.is_connected.return_value = False
        self.stale.is_playing.return_value = False
        self.stale.is_paused.return_value = False
        self.fresh = Mock()
        self.fresh.channel = self.stale.channel
        self.fresh.is_connected.return_value = True
        self.fresh.is_playing.return_value = False
        self.fresh.is_paused.return_value = False
        self.fresh.disconnect = AsyncMock()
        self.channel = self.item.user.voice.channel
        self.channel.connect = AsyncMock(side_effect=self.connect_fresh)
        self.stale.disconnect.side_effect = self.disconnect_stale
        self.state = bot.PlaybackState(
            serial=4, track=Path("1/old.mp3"), playlist="1", volume_percent=35
        )
        bot.playback[123] = self.state
        self.target = Path("1/selected.mp3")
        self.patches = [
            patch.object(bot, "refresh_player", new_callable=AsyncMock),
            patch.object(bot, "available_tracks", return_value=[self.target, Path("1/next.mp3")]),
            patch.object(bot, "available_playlists", return_value={"1": Path("1"), "2": Path("2")}),
            patch.object(bot, "find_track", return_value=(self.target, None)),
        ]
        for patcher in self.patches:
            patcher.start()

    async def asyncTearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        bot.playback.clear()
        bot.player_messages.clear()
        bot.player_locks.clear()
        bot.panel_update_locks.clear()

    async def disconnect_stale(self, **kwargs):
        self.item.guild.voice_client = None
        # A delayed old audio callback must not clear a newer song later.
        self.old_serial = 4

    async def connect_fresh(self, **kwargs):
        self.item.guild.voice_client = self.fresh
        return self.fresh

    async def test_existing_healthy_connection_preserves_playback(self):
        self.stale.is_connected.return_value = True
        self.stale.is_paused.return_value = True
        result = await bot.connect_to_user(self.item)
        self.assertIs(result, self.stale)
        self.stale.disconnect.assert_not_awaited()
        self.channel.connect.assert_not_awaited()
        self.assertEqual((self.state.serial, self.state.track, self.state.volume_percent),
                         (4, Path("1/old.mp3"), 35))

    async def test_stale_connection_is_replaced_and_old_callback_invalidated(self):
        result = await bot.connect_to_user(self.item)
        self.assertIs(result, self.fresh)
        self.stale.disconnect.assert_awaited_once_with(force=True)
        self.channel.connect.assert_awaited_once()
        self.assertGreater(self.state.serial, 4)
        self.state.track = self.target
        bot.finish_track(123, 4, None)
        self.assertEqual((self.state.playlist, self.state.volume_percent, self.state.track),
                         ("1", 35, self.target))

    async def test_all_music_entry_points_recover_before_starting_audio(self):
        actions = ("play", "toggle", "previous", "next", "restart", "song", "playlist")
        for action in actions:
            with self.subTest(action=action):
                self.item.guild.voice_client = self.stale
                self.state.track = Path("1/old.mp3")
                self.state.playlist = "1"
                self.channel.connect.reset_mock()
                with patch.object(bot, "start_track") as start, \
                     patch.object(Path, "is_file", return_value=True):
                    if action == "play":
                        await bot.play.callback(self.item, "selected")
                    elif action == "song":
                        await bot.handle_song_selection(self.item, bot.song_token("1", self.target))
                    elif action == "playlist":
                        await bot.handle_playlist_selection(self.item, bot.playlist_token("2"))
                    else:
                        await bot.handle_player_action(self.item, action)
                    self.channel.connect.assert_awaited_once()
                    start.assert_called_once()
                    self.assertIs(start.call_args.args[0], self.fresh)
                    if action == "restart":
                        self.assertEqual(start.call_args.args[-1], Path("1/old.mp3"))
                    self.stale.play.assert_not_called()

    async def test_disconnected_paused_song_is_reconnected_instead_of_resumed(self):
        self.stale.is_paused.return_value = True
        self.state.track = self.target
        with patch.object(bot, "start_track") as start:
            await bot.handle_song_selection(self.item, bot.song_token("1", self.target))
            start.assert_called_once_with(self.fresh, 123, self.state, self.target)
        self.stale.resume.assert_not_called()

    async def test_network_failure_does_not_start_audio_and_next_click_can_retry(self):
        errors = (TimeoutError(), socket.gaierror(11001, "getaddrinfo failed"),
                  aiohttp.ClientConnectionError("voice websocket unavailable"))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.item.guild.voice_client = self.stale
                self.channel.connect.side_effect = error
                self.item.followup.send.reset_mock()
                with patch.object(bot, "start_track") as start, self.assertLogs(level="WARNING"):
                    await bot.handle_song_selection(self.item, bot.song_token("1", self.target))
                    start.assert_not_called()
                message = self.item.followup.send.call_args.args[0]
                self.assertIn("соединение", message)
                self.assertNotIn("FFmpeg", message)
                # A failed reconnect keeps the selected song for a retry.
                self.assertEqual(self.state.track, Path("1/old.mp3"))
                self.channel.connect.side_effect = self.connect_fresh
                with patch.object(bot, "start_track") as start:
                    await bot.handle_song_selection(self.item, bot.song_token("1", self.target))
                    start.assert_called_once_with(self.fresh, 123, self.state, self.target)

    async def test_failed_handshake_never_returns_a_disconnected_voice(self):
        self.fresh.is_connected.return_value = False
        with self.assertLogs(level="WARNING"):
            result = await bot.connect_to_user(self.item)
        self.assertIsNone(result)
        self.item.followup.send.assert_awaited_once()

    async def test_access_is_rechecked_after_disconnecting_stale_voice(self):
        async def leave_during_disconnect(**kwargs):
            self.item.guild.voice_client = None
            self.item.user.voice = None
        self.stale.disconnect.side_effect = leave_during_disconnect
        result = await bot.connect_to_user(self.item)
        self.assertIsNone(result)
        self.channel.connect.assert_not_awaited()

    async def test_disconnected_voice_in_another_channel_is_not_taken_over(self):
        self.item.user.voice.channel = SimpleNamespace(id=2, connect=AsyncMock())
        result = await bot.connect_to_user(self.item)
        self.assertIsNone(result)
        self.stale.disconnect.assert_not_awaited()
        self.item.user.voice.channel.connect.assert_not_awaited()

    async def test_concurrent_clicks_share_one_reconnection(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed_connect(**kwargs):
            entered.set()
            await release.wait()
            return await self.connect_fresh(**kwargs)
        self.channel.connect.side_effect = delayed_connect
        with patch.object(bot, "start_track") as start:
            first = asyncio.create_task(bot.play.callback(self.item, "selected"))
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                second_item = helpers.make_interaction(self.item.guild)
                second = asyncio.create_task(bot.play.callback(second_item, "selected"))
                await asyncio.sleep(0)
                second_item.response.defer.assert_awaited_once()
                self.assertFalse(second.done())
            finally:
                release.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
            self.channel.connect.assert_awaited_once()
            self.assertEqual(start.call_count, 2)
            self.assertTrue(all(call.args[0] is self.fresh for call in start.call_args_list))

    async def test_loss_just_before_play_is_reported_without_creating_decoder(self):
        with patch.object(bot, "create_audio_source") as create:
            try:
                bot.start_track(self.stale, 123, self.state, self.target)
            except Exception as error:
                with self.assertLogs(level="WARNING"):
                    await bot.on_app_command_error(self.item, error)
            else:
                self.fail("Disconnected voice must not start audio")
            create.assert_not_called()
        self.assertIn("соединение", self.item.followup.send.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
