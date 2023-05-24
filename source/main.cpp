#include "MicroBit.h"

#include "tflite_codal.h"
#include "model.h"

// The Micro:bit control object
MicroBit uBit;

TfLiteCodal * tf;

// Out main function, run at startup
int main() {
    uBit.init();

    DMESGF("Hello World!");
    
    // init ML
    tf = new TfLiteCodal();
    tf->initialise(model_tflite, 6000);

    // test inference with example data

    // this should be 0:
    //2040, -2040, 1417.7388672553543, 0, 2040, -984, 824.3757332978387, 0, 2040, -2040, 1521.0920462644638, 0, 2129.6180555555557

    // this should be 1:
    // 140, 124, 3.6154803433579334, 0, -12, -28, 3.961766964913097, 0, -996, -1016, 4.211320472937928, 0, 1015.73125

    float test_data[13] = {140, 124, 3.6154803433579334, 0, -12, -28, 3.961766964913097, 0, -996, -1016, 4.211320472937928, 0, 1015.73125};
    float * results = (float *) tf->inferArray(test_data, tf->TensorType::TT_FLOAT, 13);
    if (results[0] == 1) {
        DMESGF("Results 0");
    }
    if (results[1] == 1) {
        DMESGF("Results 1");
    }

    // Will never return, but here so the compiler is happy :)
    return 0;
}