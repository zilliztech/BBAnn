#pragma once

class TimeRecorder {
    using stdclock = std::chrono::high_resolution_clock;

 public:
    explicit TimeRecorder(const std::string& header);

    ~TimeRecorder();  // trace = 0, debug = 1, info = 2, warn = 3, error = 4, critical = 5

    double
    RecordSection(const std::string& msg);

    double
    ElapseFromBegin(const std::string& msg);

    static std::string
    GetTimeSpanStr(double span);

 private:
    void
    PrintTimeRecord(const std::string& msg, double span);

 private:
    std::string header_;
    stdclock::time_point start_;
    stdclock::time_point last_;
};
