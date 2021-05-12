#ifndef CONFIG_H_
#define CONFIG_H_

// The number of labels (without negative)
#define label_num 3

struct Config {

  // This must be the same as seq_length in the src/model_train/config.py
  int seq_length;

  // The number of expected consecutive inferences for each gesture type.
  int consecutiveInferenceThresholds[label_num] ;

 /* char* output_message[label_num] = {
        "30",
        "35",
        "40"
  };*/
};


#endif // CONFIG_H_
