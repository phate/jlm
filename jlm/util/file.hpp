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

class FilePath final
{
public:
  explicit FilePath(const std::string & path)
      : path_(path)
  {}

  FilePath(const FilePath & other)
      : path_(other.path_)
  {}

  FilePath(FilePath && other) noexcept
      : path_(std::move(other.path_))
  {}

  FilePath &
  operator=(const FilePath & other)
  {
    if (this == &other)
      return *this;

    path_ = other.path_;
    return *this;
  }

  FilePath &
  operator=(FilePath && other) noexcept
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
   * Emulates the behavior of the GNU coreutil "dirname".
   *
   * If the current path does not contain a parent directory, either "/" or "." is returned,
   * depending on whether this is an absolute or relative path.
   *
   * This function does not respect ".." and instead treats it like any other folder.
   *
   * Examples:
   *    "/tmp/archive.tar.gz" => "/tmp"
   *    "/tmp/jlm/" => "/tmp"
   *    "/a" => "/"
   *    "a/.." => "a"
   *    "dir/file.txt" => "dir"
   *    "test.txt" => "."
   * Special cases:
   *    "/" => "/"
   *    "." => "."
   *    "" => "."
   */
  [[nodiscard]] FilePath
  Dirname() const noexcept
  {
    if (path_.empty())
      return FilePath(".");
    if (path_ == "/")
      return FilePath("/");

    // Find the last '/' char, ignoring a trailing '/'
    size_t lastSlash = std::string::npos;
    if (path_.size() >= 1)
      lastSlash = path_.find_last_of('/', path_.size() - 2);

    // If no '/' was found, path_ is a file in the current working directory, or is "." itself
    if (lastSlash == std::string::npos)
      return FilePath(".");

    // The only '/' was at the very beginning of the path. We must keep it.
    if (lastSlash == 0)
      return FilePath("/");

    // Return the path of the parent directory, without a trailing '/'
    return FilePath(path_.substr(0, lastSlash));
  }

  /**
   * Creates a new file path \p this / \p other.
   *
   * If \p other is an absolute path, the \p this-part is completely ignored.
   * What constitutes an absolute path is platform specific.
   *
   * Any "." or ".." in paths are kept as is.
   * Except if the \p this-part is equal to ".", in which case \p other is returned directly.
   *
   * Examples:
   *  "/tmp/" join "a.txt"    => "/tmp/a.txt"
   *  "a/b" join "c/d"        => "a/b/c/d"
   *  "a/b" join "/tmp/x"     => "/tmp/x"
   *  "." join "e.txt"        => "e.txt"
   *  "" join "e.txt"         => "e.txt"
   *  "a/." join "../e.txt"   => "a/./../e.txt"
   *
   * @param other the second part of the path
   * @return the joined file path
   */
  [[nodiscard]] FilePath
  Join(const std::string & other) const
  {
    std::filesystem::path t(path_ == "." ? "" : path_.c_str());
    t.append(other);
    return FilePath(t.string());
  }

  [[nodiscard]] FilePath
  Join(const FilePath & other) const
  {
    return Join(other.to_str());
  }

  /**
   * Creates a new file path by adding the given suffix
   * @return the new file path
   */
  [[nodiscard]] FilePath
  WithSuffix(const std::string & suffix) const
  {
    return FilePath(path_ + suffix);
  }

  /**
   * \brief Determines whether the file path exists
   *
   * @return True if the file path exists, otherwise false.
   */
  [[nodiscard]] bool
  Exists() const noexcept
  {
    auto fileStatus = std::filesystem::status(path_);
    return std::filesystem::exists(fileStatus);
  }

  /** \brief Determines whether file path is a directory.
   *
   * @return True if the file path is a directory, otherwise false.
   */
  [[nodiscard]] bool
  IsDirectory() const noexcept
  {
    auto fileStatus = std::filesystem::status(path_);
    return std::filesystem::is_directory(fileStatus);
  }

  /** \brief Determines whether file path is a file.
   *
   * @return True if the file path is a file, otherwise false.
   */
  [[nodiscard]] bool
  IsFile() const noexcept
  {
    auto fileStatus = std::filesystem::status(path_);
    return std::filesystem::is_regular_file(fileStatus);
  }

  /**
   * Creates the directory represented by this file path object.
   * The parent directory must already exist.
   * The directory can also exist already, in which case this is a no-op.
   *
   * @throws jlm::util::error if an error occurs
   */
  void
  CreateDirectory() const
  {
    if (IsFile())
      throw Error("file already exists: " + path_);

    FilePath baseDir(Dirname());
    if (!baseDir.IsDirectory())
      throw Error("parent directory is not a directory: " + baseDir.to_str());

    std::error_code ec;
    std::filesystem::create_directory(path_, ec);

    if (ec.value() != 0)
      throw Error("could not create directory '" + path_ + "': " + ec.message());
  }

  [[nodiscard]] const std::string &
  to_str() const noexcept
  {
    return path_;
  }

  [[nodiscard]] bool
  operator==(const FilePath & other) const noexcept
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
  static FilePath
  CreateUniqueFileName(
      const FilePath & directory,
      const std::string & fileNamePrefix,
      const std::string & fileNameSuffix)
  {
    auto randomString = CreateRandomAlphanumericString(6);
    FilePath filePath(directory.to_str() + "/" + fileNamePrefix + randomString + fileNameSuffix);

    JLM_ASSERT(!filePath.Exists());
    return filePath;
  }

  /**
   * @return a directory suitable for temporary files
   */
  [[nodiscard]] static FilePath
  TempDirectoryPath()
  {
    return FilePath(std::filesystem::temp_directory_path().string());
  }

private:
  std::string path_;
};

class File final
{
public:
  explicit File(const FilePath & path)
      : fd_(NULL),
        path_(path)
  {}

  ~File()
  {
    close();
  }

  File(const File &) = delete;

  File(File && other) noexcept
      : fd_(other.fd_),
        path_(std::move(other.path_))
  {
    other.fd_ = NULL;
  }

  File &
  operator=(const File &) = delete;

  File &
  operator=(File && other) noexcept
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
      throw Error("Cannot open file " + path_.to_str());
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

  const FilePath &
  path() const noexcept
  {
    return path_;
  }

private:
  FILE * fd_;
  FilePath path_;
};

}

#endif
