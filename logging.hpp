#ifndef _LOGGING_HPP_
#define _LOGGING_HPP_

#include <iostream>
#include <string>
#include <fstream>
#include <mutex>

namespace logging
{

class log
{
public:
    log(void) {}
    log(const std::string &log_filename)
    {
        open(log_filename);
    }
    log(const char *log_filename)
    {
        open(log_filename);
    }
    ~log()
    {
        log_file.close();
    }
    void open(const std::string &log_filename)
    {
        log_file.open(log_filename);
    }
    void open(const char *log_filename)
    {
        log_file.open(log_filename);
    }

    void debug(const std::string &message)
    {
#ifdef _DEBUG_ENABLE
        std::lock_guard<std::mutex> _lock(mtx);
        log_file << "[DEBUG]: " << message << std::endl;
#endif
    }
    void debug(const char *message)
    {
#ifdef _DEBUG_ENABLE
        std::lock_guard<std::mutex> _lock(mtx);
        log_file << "[DEBUG]: " << message << std::endl;
#endif
    }
    void info(const std::string &message)
    {
        std::lock_guard<std::mutex> _lock(mtx);
        log_file << "[INFO]: " << message << std::endl;
    }
    void info(const char *message)
    {
        std::lock_guard<std::mutex> _lock(mtx);
        log_file << "[INFO]: " << message << std::endl;
    }
    void error(const std::string &message)
    {
        std::lock_guard<std::mutex> _lock(mtx);
        std::cerr << "[ERROR]: " << message << std::endl;
        log_file << "[ERROR]: " << message << std::endl;
    }
    void error(const char *message)
    {
        std::lock_guard<std::mutex> _lock(mtx);
        std::cerr << "[ERROR]: " << message << std::endl;
        log_file << "[ERROR]: " << message << std::endl;
    }
    void dump(const std::string &message)
    {
#ifdef _DUMP_ENABLE
        std::lock_guard<std::mutex> _lock(mtx);
        log_file << message << std::endl;
#endif
    }
    void dump(const char *message)
    {
#ifdef _DUMP_ENABLE
        std::lock_guard<std::mutex> _lock(mtx);
        log_file << message << std::endl;
#endif
    }

private:
    std::ofstream log_file;
    std::mutex mtx;
};

}

#endif /* _LOGGING_HPP_ */
