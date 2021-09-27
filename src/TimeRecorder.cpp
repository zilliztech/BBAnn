#include "util/TimeRecorder.h"
#include <iostream>

TimeRecorder::TimeRecorder(const std::string &header) : header_(header) {
  start_ = last_ = stdclock::now();
}

TimeRecorder::~TimeRecorder() {}

std::string TimeRecorder::GetTimeSpanStr(double span) {
  std::string str_sec = std::to_string(span * 0.000001) +
                        ((span > 1000000) ? " seconds" : " second");
  std::string str_ms = std::to_string(span * 0.001) + " ms";

  return str_sec + " [" + str_ms + "]";
}

void TimeRecorder::PrintTimeRecord(const std::string &msg, double span) {
  std::string str_log;
  if (!header_.empty())
    str_log += header_ + ": ";
  str_log += msg;
  str_log += " (";
  str_log += TimeRecorder::GetTimeSpanStr(span);
  str_log += ")";

  std::cout << str_log << std::endl;
}

double TimeRecorder::RecordSection(const std::string &msg) {
  stdclock::time_point curr = stdclock::now();
  double span =
      (std::chrono::duration<double, std::micro>(curr - last_)).count();
  last_ = curr;

  PrintTimeRecord(msg, span);
  return span;
}

double TimeRecorder::ElapseFromBegin(const std::string &msg) {
  stdclock::time_point curr = stdclock::now();
  double span =
      (std::chrono::duration<double, std::micro>(curr - start_)).count();

  PrintTimeRecord(msg, span);
  return span;
}
