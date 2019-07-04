/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_FILE_HPP
#define JLM_UTIL_FILE_HPP

#include <string>

namespace jlm {

class filepath final {
public:
	inline
	filepath(const std::string & path)
	: path_(path)
	{}

	/**
	* \brief Returns the base name of the file without the path.
	*
	* Example:
	*    jlm::file f("/tmp/archive.tar.gz");
	*    auto base = f.base(); // base = "archive"
	*/
	inline std::string
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
	inline std::string
	name() const noexcept
	{
		auto pos = path_.find_last_of("/");
		if (pos == std::string::npos)
			return path_;

		return path_.substr(pos+1, path_.size()-pos);
	}

	/**
	* \brief Returns the complete suffix (extension) of the file.
	*
	* Example:
	*    jlm::file f("/tmp/archive.tar.gz");
	*    auto ext = f.complete_suffix(); // ext = "tar.gz"
	*/
	inline std::string
	complete_suffix() const noexcept
	{
		auto fn = name();
		auto pos = fn.find_first_of(".");
		if (pos == std::string::npos)
			return fn;

		return fn.substr(pos+1, fn.size()-pos);
	}

	/**
	* \brief Returns a file's path, excluding the file name.
	*
	* Example:
	*    jlm::file f("/tmp/archive.tar.gz");
	*    auto path = f.path(); // path = "/tmp/"
	*/
	std::string
	path() const noexcept
	{
		auto pos = path_.find_last_of("/");
		if (pos == std::string::npos)
			return "";

		return path_.substr(0, pos+1);
	}

	inline std::string
	to_str() const noexcept
	{
		return path_;
	}

	inline bool
	operator==(const jlm::filepath & other) const noexcept
	{
		return path_ == other.path_;
	}

	inline bool
	operator==(const std::string & f) const noexcept
	{
		return path_ == f;
	}

private:
	std::string path_;
};

}

#endif
