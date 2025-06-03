/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2024 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_TIME_HPP
#define JLM_UTIL_TIME_HPP

#include <jlm/util/common.hpp>

#include <chrono>

namespace jlm::util
{

class Timer final
{
public:
  constexpr Timer()
      : ElapsedTimeInNanoseconds_(0),
        IsRunning_(false)
  {}

  Timer(const Timer & other) = delete;
  Timer(Timer && other) = default;
  Timer &
  operator=(const Timer & other) = delete;
  Timer &
  operator=(Timer && other) = default;

  [[nodiscard]] bool
  IsRunning() const noexcept
  {
    return IsRunning_;
  }

  /**
   * Discards any time counted thus far.
   * If the timer is currently running, it stops.
   */
  void
  reset() noexcept
  {
    ElapsedTimeInNanoseconds_ = 0;
    IsRunning_ = false;
  }

  /**
   * Starts the timer, without resetting any previously counted time.
   * A no-op if the timer is already running.
   */
  void
  start() noexcept
  {
    if (IsRunning_)
      return;
    Start_ = std::chrono::high_resolution_clock::now();
    IsRunning_ = true;
  }

  /**
   * Stops the timer. The timer can be resumed again by calling start().
   * If the timer was already stopped, this is a no-op.
   */
  void
  stop() noexcept
  {
    if (!IsRunning_)
      return;
    auto end = std::chrono::high_resolution_clock::now();
    ElapsedTimeInNanoseconds_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - Start_).count();
    IsRunning_ = false;
  }

  /**
   * Retrieves the total time the timer has been running since the last reset.
   * Requires the timer to not be running.
   * @return total timed runtime in wall clock nanoseconds
   */
  [[nodiscard]] size_t
  ns() const
  {
    JLM_ASSERT(!IsRunning_);
    return ElapsedTimeInNanoseconds_;
  }

private:
  size_t ElapsedTimeInNanoseconds_;
  bool IsRunning_;
  std::chrono::time_point<std::chrono::high_resolution_clock> Start_;
};

}

#endif
