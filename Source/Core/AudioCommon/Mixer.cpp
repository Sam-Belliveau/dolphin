// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "AudioCommon/Mixer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "AudioCommon/Enums.h"
#include "Common/ChunkFile.h"
#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"
#include "Common/MathUtil.h"
#include "Common/Swap.h"
#include "Core/Config/MainSettings.h"
#include "Core/Core.h"
#include "Core/System.h"

static u32 DPL2QualityToFrameBlockSize(AudioCommon::DPL2Quality quality)
{
  switch (quality)
  {
  case AudioCommon::DPL2Quality::Lowest:
    return 512;
  case AudioCommon::DPL2Quality::Low:
    return 1024;
  case AudioCommon::DPL2Quality::Highest:
    return 4096;
  default:
    return 2048;
  }
}

Mixer::Mixer(u32 BackendSampleRate)
    : m_output_sample_rate(BackendSampleRate),
      m_surround_decoder(BackendSampleRate,
                         DPL2QualityToFrameBlockSize(Config::Get(Config::MAIN_DPL2_QUALITY)))
{
  m_config_changed_callback_id = Config::AddConfigChangedCallback([this] { RefreshConfig(); });
  RefreshConfig();

  INFO_LOG_FMT(AUDIO_INTERFACE, "Mixer is initialized");
}

Mixer::~Mixer()
{
  Config::RemoveConfigChangedCallback(m_config_changed_callback_id);
}

void Mixer::DoState(PointerWrap& p)
{
  m_dma_mixer.DoState(p);
  m_streaming_mixer.DoState(p);
  m_wiimote_speaker_mixer.DoState(p);
  m_skylander_portal_mixer.DoState(p);
  for (auto& mixer : m_gba_mixers)
    mixer.DoState(p);
}

// Executed from sound stream thread
void Mixer::MixerFifo::Mix(s16* samples, std::size_t num_samples)
{
  constexpr DT_s FADE_IN_RC = DT_s(0.008);
  constexpr DT_s FADE_OUT_RC = DT_s(0.064);
  constexpr DT_s DC_BALANCER_RC = DT_s(0.01);

  // We need at least a double because the index jump has 24 bits of fractional precision.
  const double out_sample_rate = m_mixer->m_output_sample_rate;
  double in_sample_rate =
      static_cast<double>(FIXED_SAMPLE_RATE_DIVIDEND) / m_input_sample_rate_divisor;

  const double emulation_speed = m_mixer->m_config_emulation_speed;
  if (0 < emulation_speed && emulation_speed != 1.0)
    in_sample_rate *= emulation_speed;

  const double index_jump = in_sample_rate / out_sample_rate;

  const std::size_t queue_size = m_mixer->m_config_audio_buffer_ms * out_sample_rate / 1000;

  // These fade in / out multiplier are tuned to match a constant
  // fade speed regardless of the input or the output sample rate.
  const float fade_in_mul = -std::expm1(-DT_s(1.0) / (out_sample_rate * FADE_IN_RC));
  const float fade_out_mul = -std::expm1(-DT_s(1.0) / (out_sample_rate * FADE_OUT_RC));
  const float dc_balance_mul = std::exp(-DT_s(1.0) / (out_sample_rate * DC_BALANCER_RC));

  const StereoPair volume{m_LVolume.load() / 256.0f, m_RVolume.load() / 256.0f};

  std::int64_t signed_num_samples = static_cast<std::int64_t>(num_samples);

  const std::size_t head = m_queue_head.load(std::memory_order_acquire);
  std::size_t tail = m_queue_tail.load(std::memory_order_acquire);

  constexpr std::size_t search_width = 64;

  while (signed_num_samples-- > 0)
  {
    // Checks to see if the queue has gotten too long.
    if (queue_size < ((head - tail) & MAX_QUEUE_SIZE_MASK))
    {
      // Jump the playhead to half the queue size behind the head.
      const std::size_t gap = (queue_size >> 1) + 1;
      tail = (head - gap) & MAX_QUEUE_SIZE_MASK;
    }

    // Checks to see if the queue is empty.
    m_queue_tail_frac += index_jump;
    std::size_t tail_jump = static_cast<std::size_t>(m_queue_tail_frac);
    m_queue_tail_frac -= tail_jump;

    tail = (tail + tail_jump) & MAX_QUEUE_SIZE_MASK;
    if (((tail + search_width + 1) & MAX_QUEUE_SIZE_MASK) == head)
    {
      // 1) Compute the quarter‐buffer offsets in modulo space:
      std::size_t quarter = queue_size >> 2;
      std::size_t tstart =
          (head + MAX_QUEUE_SIZE_MASK + 1 - 3 * quarter) & MAX_QUEUE_SIZE_MASK;  // head - 3q
      std::size_t tend =
          (head + MAX_QUEUE_SIZE_MASK + 1 - quarter) & MAX_QUEUE_SIZE_MASK;  // head -  q

      float best_dot = std::numeric_limits<float>::lowest();
      std::size_t best_tail = (head + MAX_QUEUE_SIZE_MASK + 1 - 2 * quarter) & MAX_QUEUE_SIZE_MASK;

      if (m_mixer->m_config_fill_audio_gaps)
      {
        // 2) Precompute existing window’s magnitude:
        double existing_mag = 1e-9;
        for (std::size_t i = 0; i < (search_width << 1) + 1; ++i)
        {
          const StereoPair& ex = m_queue[(tail + i - search_width) & MAX_QUEUE_SIZE_MASK];
          existing_mag += ex.dot(ex);
        }

        // 3) Now iterate p = tstart … tend (wrap‐aware):
        for (std::size_t p = tstart; p != tend; p = (p + 1) & MAX_QUEUE_SIZE_MASK)
        {
          // Gather (search_width*2+1) samples at p
          double potential_mag = 1e-9, dot_sum = 0.0;
          for (std::size_t i = 0; i < (search_width << 1) + 1; ++i)
          {
            const StereoPair& pot = m_queue[(p + i - search_width) & MAX_QUEUE_SIZE_MASK];
            const StereoPair& ex = m_queue[(tail + i - search_width) & MAX_QUEUE_SIZE_MASK];
            potential_mag += pot.dot(pot);
            dot_sum += pot.dot(ex);
          }
          float corr = static_cast<float>(dot_sum / std::sqrt(existing_mag * potential_mag));
          if (corr > best_dot)
          {
            best_dot = corr;
            best_tail = p;
          }
        }

        m_queue_looping.store(true, std::memory_order_relaxed);
      }
      
      m_dc_balance = m_dc_balance + m_queue[tail & MAX_QUEUE_SIZE_MASK] -
                     m_queue[best_tail & MAX_QUEUE_SIZE_MASK];

      tail = best_tail;
    }

    const float t1 = m_queue_tail_frac;
    const float t2 = t1 * t1;
    const float t3 = t2 * t1;

    const StereoPair s0 = m_queue[(tail - 5) & MAX_QUEUE_SIZE_MASK];
    const StereoPair s1 = m_queue[(tail - 4) & MAX_QUEUE_SIZE_MASK];
    const StereoPair s2 = m_queue[(tail - 3) & MAX_QUEUE_SIZE_MASK];
    const StereoPair s3 = m_queue[(tail - 2) & MAX_QUEUE_SIZE_MASK];
    const StereoPair s4 = m_queue[(tail - 1) & MAX_QUEUE_SIZE_MASK];
    const StereoPair s5 = m_queue[(tail - 0) & MAX_QUEUE_SIZE_MASK];

    StereoPair sample = (s0 * StereoPair{(+0.0f + 1.0f * t1 - 2.0f * t2 + 1.0f * t3) / 12.0f} +
                         s1 * StereoPair{(+0.0f - 8.0f * t1 + 15.0f * t2 - 7.0f * t3) / 12.0f} +
                         s2 * StereoPair{(+3.0f + 0.0f * t1 - 7.0f * t2 + 4.0f * t3) / 3.0f} +
                         s3 * StereoPair{(+0.0f + 2.0f * t1 + 5.0f * t2 - 4.0f * t3) / 3.0f} +
                         s4 * StereoPair{(+0.0f - 1.0f * t1 - 6.0f * t2 + 7.0f * t3) / 12.0f} +
                         s5 * StereoPair{(+0.0f + 0.0f * t1 + 1.0f * t2 - 1.0f * t3) / 12.0f});

    // Apply Fade In / Fade Out depending on if we are looping
    if (m_queue_looping.load(std::memory_order_relaxed))
      m_fade_volume += fade_out_mul * (0.0f - m_fade_volume);
    else
      m_fade_volume += fade_in_mul * (1.0f - m_fade_volume);

    // Apply the fade volume and the regular volume to the sample
    sample = (sample + m_dc_balance) * volume * StereoPair{m_fade_volume};
    m_dc_balance = m_dc_balance * dc_balance_mul;

    // This quantization method prevents accumulated error but does not do noise shaping.
    sample.l += samples[0] - m_quantization_error.l;
    samples[0] = MathUtil::SaturatingCast<s16>(std::lround(sample.l));
    m_quantization_error.l = std::clamp(samples[0] - sample.l, -1.0f, 1.0f);

    sample.r += samples[1] - m_quantization_error.r;
    samples[1] = MathUtil::SaturatingCast<s16>(std::lround(sample.r));
    m_quantization_error.r = std::clamp(samples[1] - sample.r, -1.0f, 1.0f);

    samples += 2;
  }

  m_queue_tail.store(tail, std::memory_order_release);
}

std::size_t Mixer::Mix(s16* samples, std::size_t num_samples)
{
  if (!samples)
    return 0;

  memset(samples, 0, num_samples * 2 * sizeof(s16));

  m_dma_mixer.Mix(samples, num_samples);
  m_streaming_mixer.Mix(samples, num_samples);
  m_wiimote_speaker_mixer.Mix(samples, num_samples);
  m_skylander_portal_mixer.Mix(samples, num_samples);
  for (auto& mixer : m_gba_mixers)
    mixer.Mix(samples, num_samples);

  return num_samples;
}

std::size_t Mixer::MixSurround(float* samples, std::size_t num_samples)
{
  if (!num_samples)
    return 0;

  memset(samples, 0, num_samples * SURROUND_CHANNELS * sizeof(float));

  std::size_t needed_frames = m_surround_decoder.QueryFramesNeededForSurroundOutput(num_samples);

  constexpr std::size_t max_samples = 0x8000;
  ASSERT_MSG(AUDIO, needed_frames <= max_samples,
             "needed_frames would overflow m_scratch_buffer: {} -> {} > {}", num_samples,
             needed_frames, max_samples);

  std::array<s16, max_samples> buffer;
  std::size_t available_frames = Mix(buffer.data(), static_cast<std::size_t>(needed_frames));
  if (available_frames != needed_frames)
  {
    ERROR_LOG_FMT(AUDIO,
                  "Error decoding surround frames: needed {} frames for {} samples but got {}",
                  needed_frames, num_samples, available_frames);
    return 0;
  }

  m_surround_decoder.PutFrames(buffer.data(), needed_frames);
  m_surround_decoder.ReceiveFrames(samples, num_samples);

  return num_samples;
}

void Mixer::MixerFifo::PushSamples(const s16* samples, std::size_t num_samples)
{
  const std::size_t tail = m_queue_tail.load(std::memory_order_acquire);
  std::size_t head = m_queue_head.load(std::memory_order_acquire);

  while (num_samples-- > 0)
  {
    const s16 l = m_little_endian ? samples[1] : Common::swap16(samples[1]);
    const s16 r = m_little_endian ? samples[0] : Common::swap16(samples[0]);
    samples += 2;

    // Check if we run out of space in the circular queue. (rare)
    std::size_t next_head = (head + 1) % MAX_QUEUE_SIZE_MASK;
    if (next_head == tail) [[unlikely]]
    {
      WARN_LOG_FMT(AUDIO,
                   "Granule Queue has completely filled and audio samples are being dropped. "
                   "This should not happen unless the audio backend has stopped requesting audio.");
      break;
    }

    m_queue[head] = StereoPair(l, r);
    head = next_head;
  }

  m_queue_head.store(head, std::memory_order_release);
  m_queue_looping.store(false, std::memory_order_relaxed);
}

void Mixer::PushSamples(const s16* samples, std::size_t num_samples)
{
  m_dma_mixer.PushSamples(samples, num_samples);
  if (m_log_dsp_audio)
  {
    const s32 sample_rate_divisor = m_dma_mixer.GetInputSampleRateDivisor();
    auto volume = m_dma_mixer.GetVolume();
    m_wave_writer_dsp.AddStereoSamplesBE(samples, static_cast<u32>(num_samples),
                                         sample_rate_divisor, volume.first, volume.second);
  }
}

void Mixer::PushStreamingSamples(const s16* samples, std::size_t num_samples)
{
  m_streaming_mixer.PushSamples(samples, num_samples);
  if (m_log_dtk_audio)
  {
    const s32 sample_rate_divisor = m_streaming_mixer.GetInputSampleRateDivisor();
    auto volume = m_streaming_mixer.GetVolume();
    m_wave_writer_dtk.AddStereoSamplesBE(samples, static_cast<u32>(num_samples),
                                         sample_rate_divisor, volume.first, volume.second);
  }
}

void Mixer::PushWiimoteSpeakerSamples(const s16* samples, std::size_t num_samples,
                                      u32 sample_rate_divisor)
{
  // Max 20 bytes/speaker report, may be 4-bit ADPCM so multiply by 2
  static constexpr std::size_t MAX_SPEAKER_SAMPLES = 20 * 2;
  std::array<s16, MAX_SPEAKER_SAMPLES * 2> samples_stereo;

  ASSERT_MSG(AUDIO, num_samples <= MAX_SPEAKER_SAMPLES,
             "num_samples would overflow samples_stereo: {} > {}", num_samples,
             MAX_SPEAKER_SAMPLES);
  if (num_samples <= MAX_SPEAKER_SAMPLES)
  {
    m_wiimote_speaker_mixer.SetInputSampleRateDivisor(sample_rate_divisor);

    for (std::size_t i = 0; i < num_samples; ++i)
    {
      samples_stereo[i * 2] = samples[i];
      samples_stereo[i * 2 + 1] = samples[i];
    }

    m_wiimote_speaker_mixer.PushSamples(samples_stereo.data(), num_samples);
  }
}

void Mixer::PushSkylanderPortalSamples(const u8* samples, std::size_t num_samples)
{
  // Skylander samples are always supplied as 64 bytes, 32 x 16 bit samples
  // The portal speaker is 1 channel, so duplicate and play as stereo audio
  static constexpr std::size_t MAX_PORTAL_SPEAKER_SAMPLES = 32;
  std::array<s16, MAX_PORTAL_SPEAKER_SAMPLES * 2> samples_stereo;

  ASSERT_MSG(AUDIO, num_samples <= MAX_PORTAL_SPEAKER_SAMPLES,
             "num_samples is not less or equal to 32: {} > {}", num_samples,
             MAX_PORTAL_SPEAKER_SAMPLES);

  if (num_samples <= MAX_PORTAL_SPEAKER_SAMPLES)
  {
    for (std::size_t i = 0; i < num_samples; ++i)
    {
      s16 sample = static_cast<u16>(samples[i * 2 + 1]) << 8 | static_cast<u16>(samples[i * 2]);
      samples_stereo[i * 2] = sample;
      samples_stereo[i * 2 + 1] = sample;
    }

    m_skylander_portal_mixer.PushSamples(samples_stereo.data(), num_samples);
  }
}

void Mixer::PushGBASamples(std::size_t device_number, const s16* samples, std::size_t num_samples)
{
  m_gba_mixers[device_number].PushSamples(samples, num_samples);
}

void Mixer::SetDMAInputSampleRateDivisor(u32 rate_divisor)
{
  m_dma_mixer.SetInputSampleRateDivisor(rate_divisor);
}

void Mixer::SetStreamInputSampleRateDivisor(u32 rate_divisor)
{
  m_streaming_mixer.SetInputSampleRateDivisor(rate_divisor);
}

void Mixer::SetGBAInputSampleRateDivisors(std::size_t device_number, u32 rate_divisor)
{
  m_gba_mixers[device_number].SetInputSampleRateDivisor(rate_divisor);
}

void Mixer::SetStreamingVolume(u32 lvolume, u32 rvolume)
{
  m_streaming_mixer.SetVolume(std::clamp<u32>(lvolume, 0x00, 0xff),
                              std::clamp<u32>(rvolume, 0x00, 0xff));
}

void Mixer::SetWiimoteSpeakerVolume(u32 lvolume, u32 rvolume)
{
  m_wiimote_speaker_mixer.SetVolume(lvolume, rvolume);
}

void Mixer::SetGBAVolume(std::size_t device_number, u32 lvolume, u32 rvolume)
{
  m_gba_mixers[device_number].SetVolume(lvolume, rvolume);
}

void Mixer::StartLogDTKAudio(const std::string& filename)
{
  if (!m_log_dtk_audio)
  {
    bool success = m_wave_writer_dtk.Start(filename, m_streaming_mixer.GetInputSampleRateDivisor());
    if (success)
    {
      m_log_dtk_audio = true;
      m_wave_writer_dtk.SetSkipSilence(false);
      NOTICE_LOG_FMT(AUDIO, "Starting DTK Audio logging");
    }
    else
    {
      m_wave_writer_dtk.Stop();
      NOTICE_LOG_FMT(AUDIO, "Unable to start DTK Audio logging");
    }
  }
  else
  {
    WARN_LOG_FMT(AUDIO, "DTK Audio logging has already been started");
  }
}

void Mixer::StopLogDTKAudio()
{
  if (m_log_dtk_audio)
  {
    m_log_dtk_audio = false;
    m_wave_writer_dtk.Stop();
    NOTICE_LOG_FMT(AUDIO, "Stopping DTK Audio logging");
  }
  else
  {
    WARN_LOG_FMT(AUDIO, "DTK Audio logging has already been stopped");
  }
}

void Mixer::StartLogDSPAudio(const std::string& filename)
{
  if (!m_log_dsp_audio)
  {
    bool success = m_wave_writer_dsp.Start(filename, m_dma_mixer.GetInputSampleRateDivisor());
    if (success)
    {
      m_log_dsp_audio = true;
      m_wave_writer_dsp.SetSkipSilence(false);
      NOTICE_LOG_FMT(AUDIO, "Starting DSP Audio logging");
    }
    else
    {
      m_wave_writer_dsp.Stop();
      NOTICE_LOG_FMT(AUDIO, "Unable to start DSP Audio logging");
    }
  }
  else
  {
    WARN_LOG_FMT(AUDIO, "DSP Audio logging has already been started");
  }
}

void Mixer::StopLogDSPAudio()
{
  if (m_log_dsp_audio)
  {
    m_log_dsp_audio = false;
    m_wave_writer_dsp.Stop();
    NOTICE_LOG_FMT(AUDIO, "Stopping DSP Audio logging");
  }
  else
  {
    WARN_LOG_FMT(AUDIO, "DSP Audio logging has already been stopped");
  }
}

void Mixer::RefreshConfig()
{
  m_config_emulation_speed = Config::Get(Config::MAIN_EMULATION_SPEED);
  m_config_fill_audio_gaps = Config::Get(Config::MAIN_AUDIO_FILL_GAPS);
  m_config_audio_buffer_ms = Config::Get(Config::MAIN_AUDIO_BUFFER_SIZE);
}

void Mixer::MixerFifo::DoState(PointerWrap& p)
{
  p.Do(m_input_sample_rate_divisor);
  p.Do(m_LVolume);
  p.Do(m_RVolume);
}

void Mixer::MixerFifo::SetInputSampleRateDivisor(u32 rate_divisor)
{
  m_input_sample_rate_divisor = rate_divisor;
}

u32 Mixer::MixerFifo::GetInputSampleRateDivisor() const
{
  return m_input_sample_rate_divisor;
}

void Mixer::MixerFifo::SetVolume(u32 lvolume, u32 rvolume)
{
  m_LVolume.store(lvolume + (lvolume >> 7));
  m_RVolume.store(rvolume + (rvolume >> 7));
}

std::pair<s32, s32> Mixer::MixerFifo::GetVolume() const
{
  return std::make_pair(m_LVolume.load(), m_RVolume.load());
}
