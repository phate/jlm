/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_FILE_HPP
#define JLM_UTIL_FILE_HPP

#include <jlm/util/common.hpp>
#include <jlm/util/strfmt.hpp>

#include <filesystem>
#include <string>

namespace jlm::util
{

class filepath final
{
public:
  filepath(const std::string & path)
      : path_(path)
  {}

  filepath(const filepath & other)
      : path_(other.path_)
  {}

  filepath(filepath && other) noexcept
      : path_(std::move(other.path_))
  {}

  filepath &
  operator=(const filepath & other)
  {
    if (this == &other)
      return *this;

    path_ = other.path_;
    return *this;
  }

  filepath &
  operator=(filepath && other) noexcept
  {
    if (this == &other)
      return *this;

    path_ = std::move(other.path_);
    return *this;
  }

  /**
   * \brief Returns the base name of the file without the path.
   *
   * Example:
   *    jlm::file f("/tmp/archive.tar.gz");
   *    auto base = f.base(); // base = "archive"
   */
  [[nodiscard]] std::string
  base() const noexcept
  {
    auto fn = name();
    auto pos = fn.find_first_of(".");
    if (pos == std::string::npos)
      return fn;

    return fn.substr(0, pos);
  }

  /**
   * \brief Returns the name of the file, excluding the path.
   *
   * Example:
   *    jlm::file f("/tmp/archive.tar.gz");
   *    auto name = f.name(); // name = "archive.tar.gz"
   *
   * @return The name of the file
   */
  [[nodiscard]] std::string
  name() const noexcept
  {
    auto pos = path_.find_last_of("/");
    if (pos == std::string::npos)
      return path_;

    return path_.substr(pos + 1, path_.size() - pos);
  }

  /**
   * \brief Returns the complete suffix (extension) of the file.
   *
   * Example:
   *    jlm::file f("/tmp/archive.tar.gz");
   *    auto ext = f.complete_suffix(); // ext = "tar.gz"
   */
  [[nodiscard]] std::string
  complete_suffix() const noexcept
  {
    auto fn = name();
    auto pos = fn.find_first_of(".");
    if (pos == std::string::npos)
      return fn;

    return fn.substr(pos + 1, fn.size() - pos);
  }

  /**
   * \brief Returns the suffix (extension) of the file.
   *
   * Example:
   *    jlm::file f("/tmp/archive.tar.gz");
   *    auto ext = f.suffix(); // ext = "gz"
   */
  [[nodiscard]] std::string
  suffix() const noexcept
  {
    auto fn = name();
    auto pos = fn.find_last_of(".");
    if (pos == std::string::npos)
      return fn;

    return fn.substr(pos + 1, fn.size() - pos);
  }

  /**
   * \brief Returns the path to the file or directory's parent directory.
   *
   * If the current path does not contain a parent directory, either "/" or "" is returned,
   * depending on whether this is an absolute or relative path.
   *
   * This function does not respect ".." and instead treats it like any other folder,
   * just like the GNU coreutil "dirname"
   *
   * Examples:
   *    "/tmp/archive.tar.gz" => "/tmp/"
   *    "/tmp/jlm/" => "/tmp/"
   *    "dir/file.txt" => "dir/"
   *    "test.txt" => ""
   * Special cases:
   *    "/" => "/"
   *    "." => ""
   *    "" => ""
   */
  [[nodiscard]] std::string
  path() const noexcept
  {
    if (path_.empty())
      return "";
    if (path_ == "/")
      return "/";

    // Ignore a potential trailing '/'
    auto pos = path_.find_last_of("/", path_.size() - 2);

    // If no / was found, path_ is a file in the current working directory
    if (pos == std::string::npos)
      return "";

    return path_.substr(0, pos + 1);
  }

  /**
   * Creates a new filepath "this / other".
   *
   * If other is an absolute path, the "this"-part is completely ignored.
   *
   * Examples:
   *  "/tmp/" join "a.txt"    => "/tmp/a.txt"
   *  "a/b" join "c/d"        => "a/b/c/d"
   *  "a/b" join "/tmp/x"     => "/tmp/x"
   *
   * @param other the second part of the path
   * @return the joined file path
   */
  [[nodiscard]] util::filepath
  join(const util::filepath & other) const
  {
    std::filesystem::path t(to_str());
    t.append(other.to_str());
    return t.string();
  }

  /**
   * \brief Determines whether the filepath exists
   *
   * @return True if the filepath exists, otherwise false.
   */
  [[nodiscard]] bool
  Exists() const noexcept
  {
    auto fileStatus = std::filesystem::status(path_);
    return std::filesystem::exists(fileStatus);
  }

  /** \brief Determines whether filepath is a directory.
   *
   * @return True if the filepath is a directory, otherwise false.
   */
  [[nodiscard]] bool
  IsDirectory() const noexcept
  {
    auto fileStatus = std::filesystem::status(path_);
    return std::filesystem::is_directory(fileStatus);
  }

  /** \brief Determines whether filepath is a file.
   *
   * @return True if the filepath is a file, otherwise false.
   */
  [[nodiscard]] bool
  IsFile() const noexcept
  {
    auto fileStatus = std::filesystem::status(path_);
    return std::filesystem::is_regular_file(fileStatus);
  }

  /**
   * Creates a directory with the given filepath.
   *
   * @throws jlm::util::error if the filepath aleady exists, if the parent directory does not exist,
   * or if any other filesystem error occurs
   */
  void
  CreateDirectory() const
  {
    if (Exists())
      throw error("filepath already exists: " + path_);

    filepath baseDir(path());
    if (!baseDir.IsDirectory())
      throw error("parent directory is not a directory: " + baseDir.to_str());

    std::error_code ec;
    std::filesystem::create_directory(path_, ec);

    if (ec.value() != 0)
      throw error("cannot create directory '" + path_ + "': " + ec.message());
  }

  [[nodiscard]] std::string
  to_str() const noexcept
  {
    return path_;
  }

  [[nodiscard]] bool
  operator==(const filepath & other) const noexcept
  {
    return path_ == other.path_;
  }

  [[nodiscard]] bool
  operator==(const std::string & f) const noexcept
  {
    return path_ == f;
  }

  /** \brief Generates a unique file in a given \p directory with a prefix and suffix.
   *
   * @param directory The directory in which the file is created.
   * @param fileNamePrefix The file name prefix.
   * @param fileNameSuffix The file name suffix.
   *
   * @return A unique file
   */
  static jlm::util::filepath
  CreateUniqueFileName(
      const jlm::util::filepath & directory,
      const std::string & fileNamePrefix,
      const std::string & fileNameSuffix)
  {
    auto randomString = CreateRandomAlphanumericString(6);
    filepath filePath(directory.to_str() + "/" + fileNamePrefix + randomString + fileNameSuffix);

    JLM_ASSERT(!filePath.Exists());
    return filePath;
  }

private:
  std::string path_;
};

class file final
{
public:
  file(const filepath & path)
      : fd_(NULL),
        path_(path)
  {}

  ~file()
  {
    close();
  }

  file(const file &) = delete;

  file(file && other)
      : fd_(other.fd_),
        path_(std::move(other.path_))
  {
    other.fd_ = NULL;
  }

  file &
  operator=(const file &) = delete;

  file &
  operator=(file && other)
  {
    if (this == &other)
      return *this;

    fd_ = other.fd_;
    path_ = std::move(other.path_);
    other.fd_ = NULL;

    return *this;
  }

  void
  close() noexcept
  {
    if (fd_)
      fclose(fd_);

    fd_ = NULL;
  }

  void
  open(const char * mode)
  {
    fd_ = fopen(path_.to_str().c_str(), mode);
    if (!fd_)
      throw error("Cannot open file " + path_.to_str());
  }

  bool
  is_open() const noexcept
  {
    return fd_ != NULL;
  }

  FILE *
  fd() const noexcept
  {
    return fd_;
  }

  const filepath &
  path() const noexcept
  {
    return path_;
  }

private:
  FILE * fd_;
  filepath path_;
};

}

#endif
