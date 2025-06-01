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
class notifier;

template<typename... Args>
class notifier_proxy final
{
public:
  typedef std::function<void(Args...)> function_type;

  inline Callback
  connect(function_type fn)
  {
    return notifier_.connect(fn);
  }

private:
  notifier_proxy(notifier<Args...> & n) noexcept
      : notifier_(n)
  {}

  notifier<Args...> & notifier_;

  friend class notifier<Args...>;
};

template<typename... Args>
class notifier final
{
public:
  typedef std::function<void(Args...)> function_type;

private:
  class callback_impl final : public Callback::callback_impl
  {
  public:
    virtual ~callback_impl() noexcept
    {}

    inline callback_impl(notifier * n, function_type fn)
        : notifier_(n),
          fn_(std::move(fn))
    {}

    virtual void
    disconnect() noexcept
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

    notifier * notifier_;
    function_type fn_;

    callback_impl * prev_;
    callback_impl * next_;
  };

public:
  inline ~notifier() noexcept
  {
    while (first_)
    {
      first_->disconnect();
    }
  }

  inline constexpr notifier() noexcept
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

  inline notifier_proxy<Args...>
  proxy() noexcept
  {
    return notifier_proxy<Args...>(*this);
  }

private:
  callback_impl * first_;
  callback_impl * last_;
};

}

#endif
