#pragma once



// type definations
//

enum class MetricType {
    None = 0,
    L2 = 1,
    IP = 2,
};

enum class DataType {
    None = 0,
    INT8 = 1,
    FLOAT = 2,
};

enum class QuantizerType {
    None = 0,
    PQ = 1,
    PQRES = 2,
};

enum class LevelType {
    FIRST_LEVEL = 0,
    SECOND_LEVEL = 1,
    THIRTH_LEVEL = 2,
    BALANCE_LEVEL = 3,
    FINAL_LEVEL =4,
};
