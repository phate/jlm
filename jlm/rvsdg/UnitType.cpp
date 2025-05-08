#include <jlm/rvsdg/UnitType.hpp>

namespace jlm::rvsdg
{

UnitType::~UnitType() noexcept = default;

std::string
UnitType::debug_string() const
{
  return "Unit";
}

bool
UnitType::operator==(const Type & other) const noexcept
{
  return dynamic_cast<const UnitType *>(&other) != nullptr;
}

std::size_t
UnitType::ComputeHash() const noexcept
{
  return typeid(UnitType).hash_code();
}

std::shared_ptr<const UnitType>
UnitType::Create()
{
  static const UnitType instance;
  return std::shared_ptr<const UnitType>(std::shared_ptr<void>(), &instance);
}

}
