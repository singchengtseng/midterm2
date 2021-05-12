#include "mbed.h"
#include "mbed_rpc.h"

#include "stm32l475e_iot01_accelero.h"

#include "uLCD_4DGL.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "math.h"
#include "accelerometer_handler.h"

#include "magic_wand_model_data.h"
#include "config.h"


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"



#define PI 3.14159
 

//uLCD_4DGL uLCD(D1, D0, D2);
DigitalIn btn(USER_BUTTON);
BufferedSerial pc(USBTX, USBRX);
DigitalOut led1(LED1);
DigitalOut led2(LED2);
DigitalOut led3(LED3);

Thread t1;
Thread t2;
Thread t_model;

float threshold_angle; 

bool flag=true;

struct Config config={64,{20, 10,250}};


const char* host = "172.22.1.91";

volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;

const char* topic = "Mbed";

WiFiInterface *wifi = WiFiInterface::get_default_instance();
NetworkInterface* net = wifi;
MQTTNetwork mqttNetwork(net);
MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);

int choose=0;

/////////////////////////////////////tensorflow/////////////////////////////////////////////////////

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}


int model_run(){


  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

   
    // Produce an output
    if (gesture_index < label_num) {
      //error_reporter->Report(config.output_message[gesture_index]);
       choose=gesture_index;
    }
  }

}


//////////////////////////////////tensorflow MODEL//////////////////////////////////////////////////////////////




//////////////////////////////////////MQTT/////////////////////////////////////////


void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
    printf(msg);
    ThisThread::sleep_for(1000ms);
    char payload[300];
    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char*)message.payload);
    printf(payload);
    ++arrivedcount;
}




void publish_message(MQTT::Client<MQTTNetwork, Countdown>* Client,int Angle) {
    message_num++;
    MQTT::Message message;
    char buff[100];
    sprintf(buff, "message publish:%d",Angle);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = Client->publish(topic, message);

    printf("rc:  %d\r\n", rc);
    printf("Publish message: %s\r\n", buff);
}

void close_mqtt() {
    closed = true;
}



int connect_to_mqtt(){
    
    if (!wifi) {
            printf("ERROR: No WiFiInterface found.\r\n");
            return -1;
    }


    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
            printf("\nConnection error: %d\r\n", ret);
            return -1;
    }


    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);

    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting
//(host, 1883);//connect to mqtt broker
    int rc = mqttNetwork.connect(sockAddr);
    if (rc != 0) {
            printf("Connection error.");
            return -1;
    }
    printf("Successfully connected!\r\n");

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";

    if ((rc = client.connect(data)) != 0){
            printf("Fail to connect MQTT\r\n");
    }
    int num = 0;
    while (num != 5) {
            client.yield(100);
            ++num;
    }

    while (1) {
            if (closed) break;
            client.yield(500);
            ThisThread::sleep_for(500ms);
    }

    printf("Ready to close MQTT Network......\n");

    if ((rc = client.unsubscribe(topic)) != 0) {
            printf("Failed: rc from unsubscribe was %d\n", rc);
    }
    if ((rc = client.disconnect()) != 0) {
    printf("Failed: rc from disconnect was %d\n", rc);
    }

    mqttNetwork.disconnect();
    printf("Successfully closed!\n");
    

}

////////////////////////////////////////mqtt/////////////

///////////////////////RPCfunction//////////////////

void gesture_ui(Arguments *in, Reply *out);
void angle_detection(Arguments *in, Reply *out);
void stop_gesture_ui(Arguments *in, Reply *out);

RPCFunction rpcfunc1(&gesture_ui, "gesture_ui");
RPCFunction rpcfunc2(&angle_detection, "angle_detection");
RPCFunction rpcfunc3(&stop_gesture_ui, "stop_ui_gesture");



int main(){

Thread tt;

tt.start(connect_to_mqtt);



    char buf[256], outbuf[256];


    FILE *devin = fdopen(&pc, "r");

    FILE *devout = fdopen(&pc, "w");

    while(1) {

    memset(buf, 0, 256);

    for (int i = 0; ; i++) {

        char recv = fgetc(devin);

        if (recv == '\n') {

            printf("\r\n");

            break;

        }

        buf[i] = fputc(recv, devout);

    }

    //Call the static call method on the RPC class

    RPC::call(buf, outbuf);

    printf("%s\r\n", outbuf);

   }

}





void blink(){
    led2=0;
    led1=1;
    ThisThread::sleep_for(500ms);
    led1=0;
    led3=1;
    ThisThread::sleep_for(500ms);
    led3=0;
    led2=1; //led2=1  ------------------->function is working right now
}

void select_threshold_angle(){
    int selection[]={30,35,40};
     
    while(flag){

        //using gesture to select threshold_angle and display on ulcd
        
        threshold_angle=selection[choose];
        //printf("%d",selection[choose]);
        if(!btn){
            //publish the threshold_angle selected to MQTT broker
            publish_message(&client,threshold_angle);
           // break;
        }
        ThisThread::sleep_for(500ms);
    }
}

void thread_func2(){
    int16_t pDataXYZ[3] = {0};
    float angle_detect=0;
    int i=0;

    int period=500;//(ms)
    float ref_x=0;float ref_y=0;float ref_z=0;
    blink();
    //flag reference vector
    BSP_ACCELERO_AccGetXYZ(pDataXYZ);
    ref_x=pDataXYZ[0];ref_y=pDataXYZ[1];ref_z=pDataXYZ[2];
    blink();
    //ThisThread::sleep_for(100ms);
    while(i<10){
        BSP_ACCELERO_AccGetXYZ(pDataXYZ);
        angle_detect=(pDataXYZ[0]*ref_x+pDataXYZ[1]*ref_y+pDataXYZ[2]*ref_z)/(sqrt(pow(ref_x,2)+pow(ref_y,2)+pow(ref_z,2))*sqrt(pow(pDataXYZ[0],2)+pow(pDataXYZ[1],2)+pow(pDataXYZ[2],2)));
        double temp=(sqrt(1-(angle_detect*angle_detect))/angle_detect);
        angle_detect = atan(temp)*180/PI;
        if(angle_detect>threshold_angle){
            publish_message(&client,pDataXYZ[0]); //ss<<angle_detect; ss.str();
            publish_message(&client,pDataXYZ[1]);
            publish_message(&client,pDataXYZ[2]);
            i++;
        }
        ThisThread::sleep_for(period);
    }

}



void gesture_ui(Arguments *in, Reply *out){
    blink();
    t_model.start(model_run);
    t1.start(select_threshold_angle);
    
    //t1.join();
}

void angle_detection(Arguments *in, Reply *out){
    blink();
    t2.start(thread_func2);
    //t2.join():
}


void stop_gesture_ui(Arguments *in, Reply *out){
    flag=false;
}

//////////////////////////rpc/////////////////////////

/*
void gesture_ui(Arguments *in, Reply *out){}
void stop_gesture_ui(Arguments *in, Reply *out){}
void angle_detection(Arguments *in, Reply *out){}
*/