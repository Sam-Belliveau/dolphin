// Copyright 2023 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef USE_RETRO_ACHIEVEMENTS

#include "Core/AchievementManager.h"

#include <cctype>
#include <memory>

#include <fmt/format.h>

#include <rcheevos/include/rc_api_info.h>
#include <rcheevos/include/rc_hash.h>

#include "Common/Image.h"
#include "Common/Logging/Log.h"
#include "Common/WorkQueueThread.h"
#include "Core/Config/AchievementSettings.h"
#include "Core/Core.h"
#include "Core/PowerPC/MMU.h"
#include "Core/System.h"
#include "DiscIO/Blob.h"
#include "VideoCommon/OnScreenDisplay.h"
#include "VideoCommon/VideoEvents.h"

static std::unique_ptr<OSD::Icon> DecodeBadgeToOSDIcon(const AchievementManager::Badge& badge);

AchievementManager& AchievementManager::GetInstance()
{
  static AchievementManager s_instance;
  return s_instance;
}

void AchievementManager::Init()
{
  if (!m_client && Config::Get(Config::RA_ENABLED))
  {
    m_client = rc_client_create(MemoryPeeker, Request);
    std::string host_url = Config::Get(Config::RA_HOST_URL);
    if (!host_url.empty())
      rc_client_set_host(m_client, host_url.c_str());
    rc_client_set_event_handler(m_client, EventHandler);
    rc_client_enable_logging(m_client, RC_CLIENT_LOG_LEVEL_VERBOSE,
                             [](const char* message, const rc_client_t* client) {
                               INFO_LOG_FMT(ACHIEVEMENTS, "{}", message);
                             });
    rc_client_set_hardcore_enabled(m_client, 0);
    rc_client_set_unofficial_enabled(m_client, 1);
    m_queue.Reset("AchievementManagerQueue", [](const std::function<void()>& func) { func(); });
    m_image_queue.Reset("AchievementManagerImageQueue",
                        [](const std::function<void()>& func) { func(); });
    if (HasAPIToken())
      Login("");
    INFO_LOG_FMT(ACHIEVEMENTS, "Achievement Manager Initialized");
  }
}

void AchievementManager::SetUpdateCallback(UpdateCallback callback)
{
  m_update_callback = std::move(callback);

  if (!m_update_callback)
    m_update_callback = [](UpdatedItems) {};

  m_update_callback(UpdatedItems{.all = true});
}

void AchievementManager::Login(const std::string& password)
{
  if (!m_client)
  {
    ERROR_LOG_FMT(
        ACHIEVEMENTS,
        "Attempted login to RetroAchievements server without achievement client initialized.");
    return;
  }
  if (password.empty())
  {
    rc_client_begin_login_with_token(m_client, Config::Get(Config::RA_USERNAME).c_str(),
                                     Config::Get(Config::RA_API_TOKEN).c_str(), LoginCallback,
                                     nullptr);
  }
  else
  {
    rc_client_begin_login_with_password(m_client, Config::Get(Config::RA_USERNAME).c_str(),
                                        password.c_str(), LoginCallback, nullptr);
  }
}

bool AchievementManager::HasAPIToken() const
{
  return !Config::Get(Config::RA_API_TOKEN).empty();
}

void AchievementManager::LoadGame(const std::string& file_path, const DiscIO::Volume* volume)
{
  if (!Config::Get(Config::RA_ENABLED) || !HasAPIToken())
  {
    return;
  }
  if (file_path.empty() && volume == nullptr)
  {
    WARN_LOG_FMT(ACHIEVEMENTS, "Called Load Game without a game.");
    return;
  }
  if (!m_client)
  {
    ERROR_LOG_FMT(ACHIEVEMENTS,
                  "Attempted to load game achievements without achievement client initialized.");
    return;
  }
  if (m_disabled)
  {
    INFO_LOG_FMT(ACHIEVEMENTS, "Achievement Manager is disabled until core is rebooted.");
    OSD::AddMessage("Achievements are disabled until you restart emulation.",
                    OSD::Duration::VERY_LONG, OSD::Color::RED);
    return;
  }
  if (volume)
  {
    std::lock_guard lg{m_lock};
    if (!m_loading_volume)
    {
      m_loading_volume = DiscIO::CreateVolume(volume->GetBlobReader().CopyReader());
    }
  }
  std::lock_guard lg{m_filereader_lock};
  rc_hash_filereader volume_reader{
      .open = (volume) ? &AchievementManager::FilereaderOpenByVolume :
                         &AchievementManager::FilereaderOpenByFilepath,
      .seek = &AchievementManager::FilereaderSeek,
      .tell = &AchievementManager::FilereaderTell,
      .read = &AchievementManager::FilereaderRead,
      .close = &AchievementManager::FilereaderClose,
  };
  rc_hash_init_custom_filereader(&volume_reader);
  rc_client_begin_identify_and_load_game(m_client, RC_CONSOLE_GAMECUBE, file_path.c_str(), NULL, 0,
                                         LoadGameCallback, NULL);
}

bool AchievementManager::IsGameLoaded() const
{
  auto* game_info = rc_client_get_game_info(m_client);
  return game_info && game_info->id != 0;
}

void AchievementManager::FetchPlayerBadge()
{
  FetchBadge(&m_player_badge, RC_IMAGE_TYPE_USER,
             [](const AchievementManager& manager) {
               auto* user_info = rc_client_get_user_info(manager.m_client);
               if (!user_info)
                 return std::string("");
               return std::string(user_info->display_name);
             },
             {.player_icon = true});
}

void AchievementManager::FetchGameBadges()
{
  FetchBadge(&m_game_badge, RC_IMAGE_TYPE_GAME,
             [](const AchievementManager& manager) {
               auto* game_info = rc_client_get_game_info(manager.m_client);
               if (!game_info)
                 return std::string("");
               return std::string(game_info->badge_name);
             },
             {.game_icon = true});

  if (!rc_client_has_achievements(m_client))
    return;

  rc_client_achievement_list_t* achievement_list;
  {
    std::lock_guard lg{m_lock};
    achievement_list = rc_client_create_achievement_list(
        m_client, RC_CLIENT_ACHIEVEMENT_CATEGORY_CORE_AND_UNOFFICIAL,
        RC_CLIENT_ACHIEVEMENT_LIST_GROUPING_PROGRESS);
  }
  for (u32 bx = 0; bx < achievement_list->num_buckets; bx++)
  {
    auto& bucket = achievement_list->buckets[bx];
    for (u32 achievement = 0; achievement < bucket.num_achievements; achievement++)
    {
      u32 achievement_id = bucket.achievements[achievement]->id;

      FetchBadge(
          &m_unlocked_badges[achievement_id], RC_IMAGE_TYPE_ACHIEVEMENT,
          [achievement_id](const AchievementManager& manager) {
            if (!rc_client_get_achievement_info(manager.m_client, achievement_id))
              return std::string("");
            return std::string(
                rc_client_get_achievement_info(manager.m_client, achievement_id)->badge_name);
          },
          {.achievements = {achievement_id}});
      FetchBadge(
          &m_locked_badges[achievement_id], RC_IMAGE_TYPE_ACHIEVEMENT_LOCKED,
          [achievement_id](const AchievementManager& manager) {
            if (!rc_client_get_achievement_info(manager.m_client, achievement_id))
              return std::string("");
            return std::string(
                rc_client_get_achievement_info(manager.m_client, achievement_id)->badge_name);
          },
          {.achievements = {achievement_id}});
    }
  }
  rc_client_destroy_achievement_list(achievement_list);
}

void AchievementManager::DoFrame()
{
  if (!IsGameLoaded() || !Core::IsCPUThread())
    return;
  if (m_framecount == 0x200)
  {
    DisplayWelcomeMessage();
  }
  if (m_framecount <= 0x200)
  {
    m_framecount++;
  }
  {
    std::lock_guard lg{m_lock};
    rc_client_do_frame(m_client);
  }
  if (!m_system)
    return;
  auto current_time = std::chrono::steady_clock::now();
  if (current_time - m_last_rp_time > std::chrono::seconds{10})
  {
    m_last_rp_time = current_time;
    rc_client_get_rich_presence_message(m_client, m_rich_presence.data(), RP_SIZE);
    m_update_callback(UpdatedItems{.rich_presence = true});
  }
}

std::recursive_mutex& AchievementManager::GetLock()
{
  return m_lock;
}

bool AchievementManager::IsHardcoreModeActive() const
{
  std::lock_guard lg{m_lock};
  if (!Config::Get(Config::RA_HARDCORE_ENABLED))
    return false;
  if (!Core::IsRunning())
    return true;
  if (!IsGameLoaded())
    return false;
  return (m_runtime.trigger_count + m_runtime.lboard_count > 0);
}

std::string_view AchievementManager::GetPlayerDisplayName() const
{
  if (!HasAPIToken())
    return "";
  auto* user = rc_client_get_user_info(m_client);
  if (!user)
    return "";
  return std::string_view(user->display_name);
}

u32 AchievementManager::GetPlayerScore() const
{
  if (!HasAPIToken())
    return 0;
  auto* user = rc_client_get_user_info(m_client);
  if (!user)
    return 0;
  return user->score;
}

const AchievementManager::BadgeStatus& AchievementManager::GetPlayerBadge() const
{
  return m_player_badge;
}

std::string_view AchievementManager::GetGameDisplayName() const
{
  return IsGameLoaded() ? std::string_view(rc_client_get_game_info(m_client)->title) : "";
}

rc_client_t* AchievementManager::GetClient()
{
  return m_client;
}

rc_api_fetch_game_data_response_t* AchievementManager::GetGameData()
{
  return &m_game_data;
}

const AchievementManager::BadgeStatus& AchievementManager::GetGameBadge() const
{
  return m_game_badge;
}

const AchievementManager::BadgeStatus& AchievementManager::GetAchievementBadge(AchievementId id,
                                                                               bool locked) const
{
  auto& badge_list = locked ? m_locked_badges : m_locked_badges;
  auto itr = badge_list.find(id);
  return (itr == badge_list.end()) ? m_default_badge : itr->second;
}

const AchievementManager::LeaderboardStatus*
AchievementManager::GetLeaderboardInfo(AchievementManager::AchievementId leaderboard_id)
{
  if (const auto leaderboard_iter = m_leaderboard_map.find(leaderboard_id);
      leaderboard_iter != m_leaderboard_map.end())
  {
    if (leaderboard_iter->second.entries.size() == 0)
      FetchBoardInfo(leaderboard_id);
    return &leaderboard_iter->second;
  }

  return nullptr;
}

AchievementManager::RichPresence AchievementManager::GetRichPresence() const
{
  return m_rich_presence;
}

void AchievementManager::SetDisabled(bool disable)
{
  bool previously_disabled;
  {
    std::lock_guard lg{m_lock};
    previously_disabled = m_disabled;
    m_disabled = disable;
    if (disable && m_is_game_loaded)
      CloseGame();
  }

  if (!previously_disabled && disable && Config::Get(Config::RA_ENABLED))
  {
    INFO_LOG_FMT(ACHIEVEMENTS, "Achievement Manager has been disabled.");
    OSD::AddMessage("Please close all games to re-enable achievements.", OSD::Duration::VERY_LONG,
                    OSD::Color::RED);
    m_update_callback(UpdatedItems{.all = true});
  }

  if (previously_disabled && !disable)
  {
    INFO_LOG_FMT(ACHIEVEMENTS, "Achievement Manager has been re-enabled.");
    m_update_callback(UpdatedItems{.all = true});
  }
};

const AchievementManager::NamedIconMap& AchievementManager::GetChallengeIcons() const
{
  return m_active_challenges;
}

std::vector<std::string> AchievementManager::GetActiveLeaderboards() const
{
  std::vector<std::string> display_values;
  for (u32 ix = 0; ix < MAX_DISPLAYED_LBOARDS && ix < m_active_leaderboards.size(); ix++)
  {
    display_values.push_back(std::string(m_active_leaderboards[ix].display));
  }
  return display_values;
}

void AchievementManager::DoState(PointerWrap& p)
{
  if (!m_client || !Config::Get(Config::RA_ENABLED))
    return;
  size_t size = 0;
  if (!p.IsReadMode())
    size = rc_client_progress_size(m_client);
  p.Do(size);
  auto buffer = std::make_unique<u8[]>(size);
  if (!p.IsReadMode())
  {
    int result = rc_client_serialize_progress_sized(m_client, buffer.get(), size);
    if (result != RC_OK)
    {
      ERROR_LOG_FMT(ACHIEVEMENTS, "Failed serializing achievement client with error code {}",
                    result);
      return;
    }
  }
  p.DoArray(buffer.get(), (u32)size);
  if (p.IsReadMode())
  {
    int result = rc_client_deserialize_progress_sized(m_client, buffer.get(), size);
    if (result != RC_OK)
    {
      ERROR_LOG_FMT(ACHIEVEMENTS, "Failed deserializing achievement client with error code {}",
                    result);
      return;
    }
    size_t new_size = rc_client_progress_size(m_client);
    if (size != new_size)
    {
      ERROR_LOG_FMT(ACHIEVEMENTS, "Loaded client size {} does not match size in state {}", new_size,
                    size);
      return;
    }
  }
  p.DoMarker("AchievementManager");
}

void AchievementManager::CloseGame()
{
  {
    std::lock_guard lg{m_lock};
    if (rc_client_get_game_info(m_client))
    {
      m_active_challenges.clear();
      m_active_leaderboards.clear();
      m_game_badge.name.clear();
      m_unlocked_badges.clear();
      m_locked_badges.clear();
      m_leaderboard_map.clear();
      rc_api_destroy_fetch_game_data_response(&m_game_data);
      m_game_data = {};
      m_queue.Cancel();
      m_image_queue.Cancel();
      rc_client_unload_game(m_client);
      m_system = nullptr;
    }
  }

  m_update_callback(UpdatedItems{.all = true});
  INFO_LOG_FMT(ACHIEVEMENTS, "Game closed.");
}

void AchievementManager::Logout()
{
  {
    std::lock_guard lg{m_lock};
    CloseGame();
    SetDisabled(false);
    m_player_badge.name.clear();
    Config::SetBaseOrCurrent(Config::RA_API_TOKEN, "");
  }

  m_update_callback(UpdatedItems{.all = true});
  INFO_LOG_FMT(ACHIEVEMENTS, "Logged out from server.");
}

void AchievementManager::Shutdown()
{
  if (m_client)
  {
    CloseGame();
    SetDisabled(false);
    m_queue.Shutdown();
    // DON'T log out - keep those credentials for next run.
    rc_client_destroy(m_client);
    m_client = nullptr;
    INFO_LOG_FMT(ACHIEVEMENTS, "Achievement Manager shut down.");
  }
}

void* AchievementManager::FilereaderOpenByFilepath(const char* path_utf8)
{
  auto state = std::make_unique<FilereaderState>();
  state->volume = DiscIO::CreateVolume(path_utf8);
  if (!state->volume)
    return nullptr;
  return state.release();
}

void* AchievementManager::FilereaderOpenByVolume(const char* path_utf8)
{
  auto state = std::make_unique<FilereaderState>();
  {
    auto& instance = GetInstance();
    std::lock_guard lg{instance.GetLock()};
    state->volume = std::move(instance.GetLoadingVolume());
  }
  if (!state->volume)
    return nullptr;
  return state.release();
}

void AchievementManager::FilereaderSeek(void* file_handle, int64_t offset, int origin)
{
  switch (origin)
  {
  case SEEK_SET:
    static_cast<FilereaderState*>(file_handle)->position = offset;
    break;
  case SEEK_CUR:
    static_cast<FilereaderState*>(file_handle)->position += offset;
    break;
  case SEEK_END:
    // Unused
    break;
  }
}

int64_t AchievementManager::FilereaderTell(void* file_handle)
{
  return static_cast<FilereaderState*>(file_handle)->position;
}

size_t AchievementManager::FilereaderRead(void* file_handle, void* buffer, size_t requested_bytes)
{
  FilereaderState* filereader_state = static_cast<FilereaderState*>(file_handle);
  bool success = (filereader_state->volume->Read(filereader_state->position, requested_bytes,
                                                 static_cast<u8*>(buffer), DiscIO::PARTITION_NONE));
  if (success)
  {
    filereader_state->position += requested_bytes;
    return requested_bytes;
  }
  else
  {
    return 0;
  }
}

void AchievementManager::FilereaderClose(void* file_handle)
{
  delete static_cast<FilereaderState*>(file_handle);
}

void AchievementManager::LoginCallback(int result, const char* error_message, rc_client_t* client,
                                       void* userdata)
{
  if (result != RC_OK)
  {
    WARN_LOG_FMT(ACHIEVEMENTS, "Failed to login {} to RetroAchievements server.",
                 Config::Get(Config::RA_USERNAME));
    return;
  }

  const rc_client_user_t* user;
  {
    std::lock_guard lg{AchievementManager::GetInstance().GetLock()};
    user = rc_client_get_user_info(client);
  }
  if (!user)
  {
    WARN_LOG_FMT(ACHIEVEMENTS, "Failed to retrieve user information from client.");
    return;
  }

  std::string config_username = Config::Get(Config::RA_USERNAME);
  if (config_username != user->username)
  {
    if (Common::CaseInsensitiveEquals(config_username, user->username))
    {
      INFO_LOG_FMT(ACHIEVEMENTS,
                   "Case mismatch between site {} and local {}; updating local config.",
                   user->username, Config::Get(Config::RA_USERNAME));
      Config::SetBaseOrCurrent(Config::RA_USERNAME, user->username);
    }
    else
    {
      INFO_LOG_FMT(ACHIEVEMENTS, "Attempted to login prior user {}; current user is {}.",
                   user->username, Config::Get(Config::RA_USERNAME));
      rc_client_logout(client);
      return;
    }
  }
  INFO_LOG_FMT(ACHIEVEMENTS, "Successfully logged in {} to RetroAchievements server.",
               user->username);
  std::lock_guard lg{AchievementManager::GetInstance().GetLock()};
  Config::SetBaseOrCurrent(Config::RA_API_TOKEN, user->token);
  AchievementManager::GetInstance().FetchPlayerBadge();
}

void AchievementManager::FetchBoardInfo(AchievementId leaderboard_id)
{
  u32* callback_data_1 = new u32(leaderboard_id);
  u32* callback_data_2 = new u32(leaderboard_id);
  rc_client_begin_fetch_leaderboard_entries(m_client, leaderboard_id, 1, 4,
                                            LeaderboardEntriesCallback, callback_data_1);
  rc_client_begin_fetch_leaderboard_entries_around_user(
      m_client, leaderboard_id, 4, LeaderboardEntriesCallback, callback_data_2);
}

void AchievementManager::LeaderboardEntriesCallback(int result, const char* error_message,
                                                    rc_client_leaderboard_entry_list_t* list,
                                                    rc_client_t* client, void* userdata)
{
  if (result != RC_OK)
  {
    WARN_LOG_FMT(ACHIEVEMENTS, "Failed to fetch leaderboard entries.");
    return;
  }

  u32 leaderboard_id = *reinterpret_cast<u32*>(userdata);
  delete userdata;
  auto& leaderboard = AchievementManager::GetInstance().m_leaderboard_map[leaderboard_id];
  for (size_t ix = 0; ix < list->num_entries; ix++)
  {
    std::lock_guard lg{AchievementManager::GetInstance().GetLock()};
    const auto& response_entry = list->entries[ix];
    auto& map_entry = leaderboard.entries[response_entry.index];
    map_entry.username.assign(response_entry.user);
    memcpy(map_entry.score.data(), response_entry.display, FORMAT_SIZE);
    map_entry.rank = response_entry.rank;
  }
  AchievementManager::GetInstance().m_update_callback({.leaderboards = {leaderboard_id}});
}

void AchievementManager::LoadGameCallback(int result, const char* error_message,
                                          rc_client_t* client, void* userdata)
{
  if (result != RC_OK)
  {
    WARN_LOG_FMT(ACHIEVEMENTS, "Failed to load data for current game.");
    return;
  }

  auto* game = rc_client_get_game_info(client);
  if (!game)
  {
    ERROR_LOG_FMT(ACHIEVEMENTS, "Failed to retrieve game information from client.");
    return;
  }
  INFO_LOG_FMT(ACHIEVEMENTS, "Loaded data for game ID {}.", game->id);

  AchievementManager::GetInstance().FetchGameBadges();
  AchievementManager::GetInstance().m_system = &Core::System::GetInstance();
  AchievementManager::GetInstance().m_update_callback({.all = true});
  // Set this to a value that will immediately trigger RP
  AchievementManager::GetInstance().m_last_rp_time =
      std::chrono::steady_clock::now() - std::chrono::minutes{2};
}

void AchievementManager::DisplayWelcomeMessage()
{
  std::lock_guard lg{m_lock};
  const u32 color =
      rc_client_get_hardcore_enabled(m_client) ? OSD::Color::YELLOW : OSD::Color::CYAN;
  if (Config::Get(Config::RA_BADGES_ENABLED))
  {
    OSD::AddMessage("", OSD::Duration::VERY_LONG, OSD::Color::GREEN,
                    DecodeBadgeToOSDIcon(m_game_badge.badge));
  }
  auto info = rc_client_get_game_info(m_client);
  if (!info)
  {
    ERROR_LOG_FMT(ACHIEVEMENTS, "Attempting to welcome player to game not running.");
    return;
  }
  OSD::AddMessage(info->title, OSD::Duration::VERY_LONG, OSD::Color::GREEN);
  rc_client_user_game_summary_t summary;
  rc_client_get_user_game_summary(m_client, &summary);
  OSD::AddMessage(fmt::format("You have {}/{} achievements worth {}/{} points",
                              summary.num_unlocked_achievements, summary.num_core_achievements,
                              summary.points_unlocked, summary.points_core),
                  OSD::Duration::VERY_LONG, color);
  if (summary.num_unsupported_achievements > 0)
  {
    OSD::AddMessage(
        fmt::format("{} achievements unsupported", summary.num_unsupported_achievements),
        OSD::Duration::VERY_LONG, OSD::Color::RED);
  }
  OSD::AddMessage(
      fmt::format("Hardcore mode is {}", rc_client_get_hardcore_enabled(m_client) ? "ON" : "OFF"),
      OSD::Duration::VERY_LONG, color);
  OSD::AddMessage(fmt::format("Leaderboard submissions are {}",
                              Config::Get(Config::RA_LEADERBOARDS_ENABLED) ? "ON" : "OFF"),
                  OSD::Duration::VERY_LONG, color);
}

void AchievementManager::HandleAchievementTriggeredEvent(const rc_client_event_t* client_event)
{
  OSD::AddMessage(fmt::format("Unlocked: {} ({})", client_event->achievement->title,
                              client_event->achievement->points),
                  OSD::Duration::VERY_LONG,
                  (rc_client_get_hardcore_enabled(AchievementManager::GetInstance().m_client)) ?
                      OSD::Color::YELLOW :
                      OSD::Color::CYAN,
                  (Config::Get(Config::RA_BADGES_ENABLED)) ?
                      DecodeBadgeToOSDIcon(AchievementManager::GetInstance()
                                               .m_unlocked_badges[client_event->achievement->id]
                                               .badge) :
                      nullptr);
}

void AchievementManager::HandleLeaderboardStartedEvent(const rc_client_event_t* client_event)
{
  OSD::AddMessage(fmt::format("Attempting leaderboard: {} - {}", client_event->leaderboard->title,
                              client_event->leaderboard->description),
                  OSD::Duration::VERY_LONG, OSD::Color::GREEN);
  AchievementManager::GetInstance().FetchBoardInfo(client_event->leaderboard->id);
}

void AchievementManager::HandleLeaderboardFailedEvent(const rc_client_event_t* client_event)
{
  OSD::AddMessage(fmt::format("Failed leaderboard: {}", client_event->leaderboard->title),
                  OSD::Duration::VERY_LONG, OSD::Color::RED);
  AchievementManager::GetInstance().FetchBoardInfo(client_event->leaderboard->id);
}

void AchievementManager::HandleLeaderboardSubmittedEvent(const rc_client_event_t* client_event)
{
  OSD::AddMessage(fmt::format("Scored {} on leaderboard: {}",
                              client_event->leaderboard->tracker_value,
                              client_event->leaderboard->title),
                  OSD::Duration::VERY_LONG, OSD::Color::YELLOW);
  AchievementManager::GetInstance().FetchBoardInfo(client_event->leaderboard->id);
}

void AchievementManager::HandleLeaderboardTrackerUpdateEvent(const rc_client_event_t* client_event)
{
  auto& active_leaderboards = AchievementManager::GetInstance().m_active_leaderboards;
  for (auto& leaderboard : active_leaderboards)
  {
    if (leaderboard.id == client_event->leaderboard_tracker->id)
    {
      strncpy(leaderboard.display, client_event->leaderboard_tracker->display,
              RC_CLIENT_LEADERBOARD_DISPLAY_SIZE);
    }
  }
}

void AchievementManager::HandleLeaderboardTrackerShowEvent(const rc_client_event_t* client_event)
{
  AchievementManager::GetInstance().m_active_leaderboards.push_back(
      *client_event->leaderboard_tracker);
}

void AchievementManager::HandleLeaderboardTrackerHideEvent(const rc_client_event_t* client_event)
{
  auto& active_leaderboards = AchievementManager::GetInstance().m_active_leaderboards;
  std::erase_if(active_leaderboards, [client_event](const auto& leaderboard) {
    return leaderboard.id == client_event->leaderboard_tracker->id;
  });
}

void AchievementManager::HandleAchievementChallengeIndicatorShowEvent(
    const rc_client_event_t* client_event)
{
  if (Config::Get(Config::RA_BADGES_ENABLED))
  {
    auto& unlocked_badges = AchievementManager::GetInstance().m_unlocked_badges;
    if (const auto unlocked_iter = unlocked_badges.find(client_event->achievement->id);
        unlocked_iter != unlocked_badges.end())
    {
      AchievementManager::GetInstance().m_active_challenges[client_event->achievement->badge_name] =
          DecodeBadgeToOSDIcon(unlocked_iter->second.badge);
    }
  }
}

void AchievementManager::HandleAchievementChallengeIndicatorHideEvent(
    const rc_client_event_t* client_event)
{
  AchievementManager::GetInstance().m_active_challenges.erase(
      client_event->achievement->badge_name);
}

void AchievementManager::HandleAchievementProgressIndicatorShowEvent(
    const rc_client_event_t* client_event)
{
  OSD::AddMessage(fmt::format("{} {}", client_event->achievement->title,
                              client_event->achievement->measured_progress),
                  OSD::Duration::SHORT, OSD::Color::GREEN,
                  (Config::Get(Config::RA_BADGES_ENABLED)) ?
                      DecodeBadgeToOSDIcon(AchievementManager::GetInstance()
                                               .m_unlocked_badges[client_event->achievement->id]
                                               .badge) :
                      nullptr);
}

void AchievementManager::HandleGameCompletedEvent(const rc_client_event_t* client_event,
                                                  rc_client_t* client)
{
  auto* user_info = rc_client_get_user_info(client);
  auto* game_info = rc_client_get_game_info(client);
  if (!user_info || !game_info)
  {
    WARN_LOG_FMT(ACHIEVEMENTS, "Received Game Completed event when game not running.");
    return;
  }
  bool hardcore = rc_client_get_hardcore_enabled(client);
  OSD::AddMessage(fmt::format("Congratulations! {} has {} {}", user_info->display_name,
                              hardcore ? "mastered" : "completed", game_info->title),
                  OSD::Duration::VERY_LONG, hardcore ? OSD::Color::YELLOW : OSD::Color::CYAN,
                  (Config::Get(Config::RA_BADGES_ENABLED)) ?
                      DecodeBadgeToOSDIcon(AchievementManager::GetInstance().m_game_badge.badge) :
                      nullptr);
}

static std::unique_ptr<OSD::Icon> DecodeBadgeToOSDIcon(const AchievementManager::Badge& badge)
{
  if (badge.empty())
    return nullptr;

  auto icon = std::make_unique<OSD::Icon>();
  if (!Common::LoadPNG(badge, &icon->rgba_data, &icon->width, &icon->height))
  {
    ERROR_LOG_FMT(ACHIEVEMENTS, "Error decoding badge.");
    return nullptr;
  }
  return icon;
}

void AchievementManager::Request(const rc_api_request_t* request,
                                 rc_client_server_callback_t callback, void* callback_data,
                                 rc_client_t* client)
{
  std::string url = request->url;
  std::string post_data = request->post_data;
  AchievementManager::GetInstance().m_queue.EmplaceItem([url = std::move(url),
                                                         post_data = std::move(post_data),
                                                         callback = std::move(callback),
                                                         callback_data = std::move(callback_data)] {
    const Common::HttpRequest::Headers USER_AGENT_HEADER = {{"User-Agent", "Dolphin/Placeholder"}};

    Common::HttpRequest http_request;
    Common::HttpRequest::Response http_response;
    if (!post_data.empty())
    {
      http_response = http_request.Post(url, post_data, USER_AGENT_HEADER,
                                        Common::HttpRequest::AllowedReturnCodes::All);
    }
    else
    {
      http_response =
          http_request.Get(url, USER_AGENT_HEADER, Common::HttpRequest::AllowedReturnCodes::All);
    }

    rc_api_server_response_t server_response;
    if (http_response.has_value() && http_response->size() > 0)
    {
      server_response.body = reinterpret_cast<const char*>(http_response->data());
      server_response.body_length = http_response->size();
      server_response.http_status_code = http_request.GetLastResponseCode();
    }
    else
    {
      constexpr char error_message[] = "Failed HTTP request.";
      server_response.body = error_message;
      server_response.body_length = sizeof(error_message);
      server_response.http_status_code = RC_API_SERVER_RESPONSE_RETRYABLE_CLIENT_ERROR;
    }

    callback(&server_response, callback_data);
  });
}

u32 AchievementManager::MemoryPeeker(u32 address, u8* buffer, u32 num_bytes, rc_client_t* client)
{
  if (buffer == nullptr)
    return 0u;
  auto& system = Core::System::GetInstance();
  Core::CPUThreadGuard threadguard(system);
  for (u32 num_read = 0; num_read < num_bytes; num_read++)
  {
    auto value = system.GetMMU().HostTryReadU8(threadguard, address + num_read,
                                               PowerPC::RequestedAddressSpace::Physical);
    if (!value.has_value())
      return num_read;
    buffer[num_read] = value.value().value;
  }
  return num_bytes;
}

void AchievementManager::FetchBadge(AchievementManager::BadgeStatus* badge, u32 badge_type,
                                    const AchievementManager::BadgeNameFunction function,
                                    const UpdatedItems callback_data)
{
  if (!m_client || !HasAPIToken() || !Config::Get(Config::RA_BADGES_ENABLED))
  {
    m_update_callback(callback_data);
    return;
  }

  m_image_queue.EmplaceItem([this, badge, badge_type, function = std::move(function),
                             callback_data = std::move(callback_data)] {
    std::string name_to_fetch;
    {
      std::lock_guard lg{m_lock};
      name_to_fetch = function(*this);
      if (name_to_fetch.empty())
        return;
    }
    rc_api_fetch_image_request_t icon_request = {.image_name = name_to_fetch.c_str(),
                                                 .image_type = badge_type};
    Badge fetched_badge;
    rc_api_request_t api_request;
    Common::HttpRequest http_request;
    if (rc_api_init_fetch_image_request(&api_request, &icon_request) != RC_OK)
    {
      ERROR_LOG_FMT(ACHIEVEMENTS, "Invalid request for image {}.", name_to_fetch);
      return;
    }
    auto http_response = http_request.Get(api_request.url);
    if (http_response.has_value() && http_response->size() <= 0)
    {
      WARN_LOG_FMT(ACHIEVEMENTS, "RetroAchievements connection failed on image request.\n URL: {}",
                   api_request.url);
      rc_api_destroy_request(&api_request);
      m_update_callback(callback_data);
      return;
    }

    rc_api_destroy_request(&api_request);
    fetched_badge = std::move(*http_response);

    INFO_LOG_FMT(ACHIEVEMENTS, "Successfully downloaded badge id {}.", name_to_fetch);
    std::lock_guard lg{m_lock};
    if (function(*this).empty() || name_to_fetch != function(*this))
    {
      INFO_LOG_FMT(ACHIEVEMENTS, "Requested outdated badge id {}.", name_to_fetch);
      return;
    }
    badge->badge = std::move(fetched_badge);
    badge->name = std::move(name_to_fetch);

    m_update_callback(callback_data);
  });
}

void AchievementManager::EventHandler(const rc_client_event_t* event, rc_client_t* client)
{
  switch (event->type)
  {
  case RC_CLIENT_EVENT_ACHIEVEMENT_TRIGGERED:
    HandleAchievementTriggeredEvent(event);
    break;
  case RC_CLIENT_EVENT_LEADERBOARD_STARTED:
    HandleLeaderboardStartedEvent(event);
    break;
  case RC_CLIENT_EVENT_LEADERBOARD_FAILED:
    HandleLeaderboardFailedEvent(event);
    break;
  case RC_CLIENT_EVENT_LEADERBOARD_SUBMITTED:
    HandleLeaderboardSubmittedEvent(event);
    break;
  case RC_CLIENT_EVENT_LEADERBOARD_TRACKER_UPDATE:
    HandleLeaderboardTrackerUpdateEvent(event);
    break;
  case RC_CLIENT_EVENT_LEADERBOARD_TRACKER_SHOW:
    HandleLeaderboardTrackerShowEvent(event);
    break;
  case RC_CLIENT_EVENT_LEADERBOARD_TRACKER_HIDE:
    HandleLeaderboardTrackerHideEvent(event);
    break;
  case RC_CLIENT_EVENT_ACHIEVEMENT_CHALLENGE_INDICATOR_SHOW:
    HandleAchievementChallengeIndicatorShowEvent(event);
    break;
  case RC_CLIENT_EVENT_ACHIEVEMENT_CHALLENGE_INDICATOR_HIDE:
    HandleAchievementChallengeIndicatorHideEvent(event);
    break;
  case RC_CLIENT_EVENT_ACHIEVEMENT_PROGRESS_INDICATOR_SHOW:
  case RC_CLIENT_EVENT_ACHIEVEMENT_PROGRESS_INDICATOR_UPDATE:
    HandleAchievementProgressIndicatorShowEvent(event);
    break;
  case RC_CLIENT_EVENT_ACHIEVEMENT_PROGRESS_INDICATOR_HIDE:
    // OnScreenDisplay messages disappear over time, so this is unnecessary
    // unless the display algorithm changes in the future.
    break;
  case RC_CLIENT_EVENT_GAME_COMPLETED:
    HandleGameCompletedEvent(event, client);
    break;
  default:
    INFO_LOG_FMT(ACHIEVEMENTS, "Event triggered of unhandled type {}", event->type);
    break;
  }
}

#endif  // USE_RETRO_ACHIEVEMENTS
