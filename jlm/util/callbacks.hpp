/*
 * Copyright 2015 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_CALLBACKS_HPP
#define JLM_UTIL_CALLBACKS_HPP

#include <atomic>
#include <functional>
#include <memory>

namespace jlm::util
{

class Callback final
{
public:
  class callback_impl
  {
  public:
    virtual ~callback_impl() noexcept;

    inline constexpr callback_impl()
        : refcount(1)
    {}

    virtual void
    disconnect() noexcept = 0;

    std::atomic<int> refcount;
  };

  inline ~Callback()
  {
    disconnect();
  }

  inline explicit Callback(callback_impl * impl) noexcept
      : impl_(impl)
  {
    impl->refcount.fetch_add(1, std::memory_order_relaxed);
  }

  inline Callback() noexcept
      : impl_(nullptr)
  {}

  inline Callback(Callback && other) noexcept
      : impl_(nullptr)
  {
    std::swap(impl_, other.impl_);
  }

  inline Callback &
  operator=(Callback && other) noexcept
  {
    reset();
    std::swap(impl_, other.impl_);
    return *this;
  }

  inline void
  disconnect() noexcept
  {
    if (impl_)
    {
      impl_->disconnect();
      reset();
    }
  }

private:
  void
  reset() noexcept
  {
    if (impl_ && impl_->refcount.fetch_sub(1, std::memory_order_release))
    {
      delete impl_;
      impl_ = 0;
    }
  }

  callback_impl * impl_;
};

template<typename... Args>
class Notifier;

template<typename... Args>
class NotifierProxy final
{
public:
  typedef std::function<void(Args...)> function_type;

  inline Callback
  connect(function_type fn)
  {
    return notifier_.connect(fn);
  }

private:
  explicit NotifierProxy(Notifier<Args...> & n) noexcept
      : notifier_(n)
  {}

  Notifier<Args...> & notifier_;

  friend class Notifier<Args...>;
};

template<typename... Args>
class Notifier final
{
public:
  typedef std::function<void(Args...)> function_type;

private:
  class callback_impl final : public Callback::callback_impl
  {
  public:
    ~callback_impl() noexcept override = default;

    callback_impl(Notifier * n, function_type fn)
        : notifier_(n),
          fn_(std::move(fn))
    {}

    void
    disconnect() noexcept override
    {
      if (!notifier_)
      {
        return;
      }

      if (prev_)
      {
        prev_->next_ = next_;
      }
      else
      {
        notifier_->first_ = next_;
      }
      if (next_)
      {
        next_->prev_ = prev_;
      }
      else
      {
        notifier_->last_ = prev_;
      }

      notifier_ = 0;
    }

    Notifier * notifier_;
    function_type fn_;

    callback_impl * prev_;
    callback_impl * next_;
  };

public:
  ~Notifier() noexcept
  {
    while (first_)
    {
      first_->disconnect();
    }
  }

  constexpr Notifier() noexcept
      : first_(nullptr),
        last_(nullptr)
  {}

  inline void
  operator()(Args... args) const
  {
    callback_impl * current = first_;
    while (current)
    {
      current->fn_(args...);
      current = current->next_;
    }
  }

  inline Callback
  connect(function_type fn)
  {
    callback_impl * c = new callback_impl(this, std::move(fn));

    c->prev_ = last_;
    c->next_ = nullptr;
    if (last_)
    {
      last_->next_ = c;
    }
    else
    {
      first_ = c;
    }
    last_ = c;

    return Callback(c);
  }

  NotifierProxy<Args...>
  proxy() noexcept
  {
    return NotifierProxy<Args...>(*this);
  }

private:
  callback_impl * first_;
  callback_impl * last_;
};

}

#endif
