#include "lib/bbannlib2.h"

int main() {
    bbann::BBAnnParameters para;
    auto index = std::make_unique<bbann::BBAnnIndex2<float, float>> (MetricType::L2);
    bbann::BBAnnIndex2<float,float>::BuildIndex(para);
    return 0;
}