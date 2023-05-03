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
    float test_data[13] = {3.018380407186306, -1.2952721069160453, 2.0557226994800146, 3.1962880785956864, 1.4162036382612526, -2.1728292442178345, 2.599114291471402, -0.5348871489053622, 1.3463181981695382, -1.225127368913248, 1.3437687771313476, -0.29443619762014206, 1.8072526207682738};
    float * results = (float *) tf->inferArray(test_data, tf->TensorType::TT_FLOAT, 13);
    if (results[0]>results[1] && results[0]>results[2])
        DMESGF("Result: %d", 1);
    else if (results[1]>results[0] && results[1]>results[2])
        DMESGF("Result: %d", 2);
    else
        DMESGF("Result: %d", 3);

    // Will never return, but here so the compiler is happy :)
    return 0;
}