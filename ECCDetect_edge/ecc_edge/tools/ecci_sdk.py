import paho.mqtt.client as mqtt
import threading
import os
import ast
import re
import _pickle as cPickle
from queue import Queue
import io
import logging
import uuid
import json

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SUB_OBJECTS = os.getenv("ECCI_SUB_OBJECTS",default = None)
PUB_TARGETS = ast.literal_eval(os.getenv("ECCI_PUB_TARGETS",default = "{}"))

BROKER_IP = os.getenv("ECCI_LOCAL_BROKER_IP")
BROKER_PORT = int(os.getenv("ECCI_LOCAL_BROKER_PORT", default="1883"))
BROKER_ID = os.getenv("ECCI_LOCAL_BROKER_ID")
BROKER_USERNAME = os.getenv("ECCI_LOCAL_BROKER_USERNAME")
BROKER_PASSWORD = os.getenv("ECCI_LOCAL_BROKER_PASSWORD")

APP_ID = os.getenv("ECCI_APP_ID")
# CONTAINER_NAME = re.findall(r"^edgeai_.*?_(.*)", os.getenv("ECCI_CONTAINER_NAME"))[0]
CONTAINER_NAME = os.getenv("ECCI_CONTAINER_NAME")
BRIDGE_MOUNTPOINTS = ast.literal_eval(os.getenv("ECCI_BRIDGE_MOUNTPOINTS",default="[]"))

# ---------------------------任务3-4新增---------------------------
ECCI_APP_TYPE = os.getenv("ECCI_APP_TYPE")  #controller/component
# 发送就绪命令给controller的target_name:相应的broker_id PS：len()==1/0
ECCI_APP_CONTROLLER_CONTAINER = ast.literal_eval(os.getenv("ECCI_APP_CONTROLLER_CONTAINER",default="{}"))
# 在label策略下存在，表示pub_targets中同镜像不同容器名的种类，可用于选择某类应用component中的部分容器
ECCI_APP_PUB_TARGETS_GROUPS = ast.literal_eval(os.getenv("ECCI_APP_CONTROLLER_CONTAINER",default="[]"))

ECCI_AGENT_ID = os.getenv("ECCI_AGENT_ID")


def gen_listen_component(pub_dict):
    target,target_broker_ip = pub_dict
    print("########### broker_ip is",target_broker_ip)
    return target,False

# 当该容器类型为controller时，其需要pub的就是需要其等待就绪的容器
if ECCI_APP_TYPE == "controller":
    LISTEN_CONTAINERS = dict(map(gen_listen_component, PUB_TARGETS.items()))
    logger.info(f"LISTEN_CONTAINERS="+str(LISTEN_CONTAINERS))
# ---------------------------任务3-4新增---------------------------


CONTAINER_TYPE = os.getenv("ECCI_CONTAINER_TYPE")

TOPIC_PREFIX = f"ECCI/{BROKER_ID}/{APP_ID}"

def gen_sub_topic(pub_dict):
    target,target_broker_ip = pub_dict
    return target,f'ECCI/{target_broker_ip}/{APP_ID}/app/{CONTAINER_NAME}/{target}'
    # return target,f'ECCI/{target_broker_ip}/{APP_ID}/app/{target}'
if PUB_TARGETS:
    PUB_TOPICS = dict(map(gen_sub_topic,PUB_TARGETS.items()))

class Client:

    def __init__(self):
        self._broker_ip = BROKER_IP
        self._broker_port = BROKER_PORT
        self.start_event = threading.Event()
        self.exit_event = threading.Event()
        self._sub_data_sender_queue = Queue()
        self._sub_data_payload_queue = Queue()
        self._sub_cmd_sender_queue = Queue()
        self._sub_cmd_payload_queue = Queue()
        self._mqtt_client = None

    def wait_for_ready(self):
        self.start_event.wait()

    def get_sub_data_sender_queue(self):
        return self._sub_data_sender_queue
    
    def get_sub_data_payload_queue(self):
        return self._sub_data_payload_queue

    def get_sub_cmd_sender_queue(self):
        return self._sub_cmd_sender_queue
    
    def get_sub_cmd_payload_queue(self):
        return self._sub_cmd_payload_queue

    def _on_client_connect(self, mqtt_client, userdata, flags, rc):
        logger.info("Connected with result code "+str(rc))
        
        if BRIDGE_MOUNTPOINTS:
            for mountpoint in BRIDGE_MOUNTPOINTS:
                sub_prefix = f"{mountpoint}{TOPIC_PREFIX}"
                self._mqtt_client.subscribe(f"{sub_prefix}/app/+/{CONTAINER_NAME}", qos=2)
                self._mqtt_client.subscribe(f"{sub_prefix}/plugin/+/{CONTAINER_NAME}", qos=2)
                print("666666666666 subscribe topic",f"{sub_prefix}/app/+/{CONTAINER_NAME}")
        self._mqtt_client.subscribe(f"{TOPIC_PREFIX}/app/+/{CONTAINER_NAME}", qos=2)
        print("777777777777 subscribe topic:",f"{TOPIC_PREFIX}/app/+/{CONTAINER_NAME}")
        self._mqtt_client.subscribe(f"{TOPIC_PREFIX}/plugin/+/{CONTAINER_NAME}", qos=2)

        if SUB_OBJECTS != None:
            sub_objects = ast.literal_eval(SUB_OBJECTS)
            for sub_object  in sub_objects:
                self._mqtt_client.subscribe(f"{TOPIC_PREFIX}/app/+/{sub_object}", qos=2)
                print("888888888888 subscribe topic:",f"{TOPIC_PREFIX}/app/+/{sub_object}")
                self._mqtt_client.subscribe(f"{TOPIC_PREFIX}/plugin/+/{sub_object}", qos=2)

        # after sub all topic , send ready
        if not self.start_event.isSet():
            self.send_ready()


    def _on_client_message(self, mqtt_client, userdata, msg):
        rev_msg = cPickle.loads(msg.payload)
        sender = self._topic_parse(msg.topic)
        if rev_msg['type'] == "cmd":
            self._sub_cmd_sender_queue.put(sender)
            self._sub_cmd_payload_queue.put(rev_msg['contents'])
            try:
                if rev_msg['contents']['cmd'] == "start":
                    logger.debug(f"{ECCI_APP_TYPE} received start message")
                    if ECCI_APP_TYPE == "controller":
                        payload = {"type":"cmd","contents":{"cmd":"start"}}
                        self.publish(message=payload)
                        logger.debug(f"{ECCI_APP_TYPE} send start message")
                    self.start_event.set()
                elif rev_msg['contents']['cmd'] == "exit":
                    self.exit_event.set()
                elif rev_msg['contents']['cmd'] == "ready":
                    self.ready_cntr_collect(sender)
                elif rev_msg['contents']['cmd'] == "req_ready": #use yuehai v2
                    self.send_ready()
            except KeyError as e:
                logger.critical(e)
                pass
        elif rev_msg['type'] == "data":
            self._sub_data_sender_queue.put(sender)
            self._sub_data_payload_queue.put(rev_msg['contents'])
        else:
            logger.error("error message = "+str(rev_msg))

    def _on_client_publish(self, mqtt_client, userdata, mid):
        logger.info("publish success")
        pass

    # 用于controller 采集所有的ready消息，判断本地所有component是否就绪或者全局下的container(包括component和controller)所有是否就绪
    def ready_cntr_collect(self, sender):
        if sender in LISTEN_CONTAINERS:
            LISTEN_CONTAINERS[sender] = True
            logger.info(f"{sender} sender ready message")
            # 所有listen containers均以发送ready 消息
            if all(list(LISTEN_CONTAINERS.values())):
                # 向上级controller发送ready或者向下发送Start 命令
                if ECCI_APP_CONTROLLER_CONTAINER:
                    # 本地控制器
                    for container_name,broker_id in ECCI_APP_CONTROLLER_CONTAINER.items():
                        pass
                    topic=f"ECCI/{broker_id}/{APP_ID}/app/{CONTAINER_NAME}/{container_name}"
                    logger.debug(topic)
                    payload = {"type":"cmd","contents":{"cmd":"ready"}}
                    self._mqtt_client.publish(topic,cPickle.dumps(payload),qos=2)
                    logger.debug("local controller send ready to global controller")
                else:
                    # 全局控制器
                    # 下发start 命令
                    self.start_event.set()
                    payload = {"type":"cmd","contents":{"cmd":"start"}}
                    self.publish(message=payload)
                    logger.debug(f"{ECCI_APP_TYPE} send start message")

                    # 新增发送app ready给ECCI，改变应用状态-4.8
                    # 从应用组件部署中到应用运行中(10->2)
                    # 或从应用组件启动中到应用运行中(11->2)
                    payload = {"app_id":APP_ID,"container_name":CONTAINER_NAME,"agent_id":ECCI_AGENT_ID,"ready_type":"app"}
                    # logger.debug(f"toEdgeAI/{BROKER_ID}/con_cntr_ready")
                    self._mqtt_client.publish(f"toEdgeAI/{BROKER_ID}/con_cntr_ready",json.dumps(payload),qos=2)
                    logger.debug("app controller and component ready, app status will change")


        else:
            logger.error(f"{sender} is not in "+str(LISTEN_CONTAINERS))

    def find_target(self, targets):
        target_list=[]
        if isinstance(targets,str):
            for pub_target in list(PUB_TARGETS.keys()):
                if pub_target.startswith(targets):
                    target_list.append(pub_target)
        elif isinstance(targets,list):
            for target in targets:
                for pub_target in list(PUB_TARGETS.keys()):
                    if pub_target.startswith(target):
                        target_list.append(pub_target)
        return target_list

    def publish(self, message, targets=None, retain=False):
        no_match = True
        try:
            if targets == None:
                for topic in list(PUB_TOPICS.values()):
                    self._mqtt_client.publish(topic=topic, payload=cPickle.dumps(message), qos=2, retain=retain)
                no_match = False
            elif isinstance(targets,str) or isinstance(targets,list):
                target_list = self.find_target(targets)
                if len(target_list) != 0:
                    no_match = False
                for target_item in target_list:
                    self._mqtt_client.publish(topic=PUB_TOPICS[target_item], payload=cPickle.dumps(message), qos=2, retain=retain)
            else:
                logger.critical(f"Illegal targets format which is {str(type(targets))}")
            if no_match:
                logger.critical(f"refused publish, {targets} does not exist in the {PUB_TOPICS}")
        except Exception as e:
            logger.error(e)

    def _topic_parse(self, topic):
        if BRIDGE_MOUNTPOINTS:
            for mountpoint in BRIDGE_MOUNTPOINTS:
                pattern = f"^{mountpoint}"
                if re.match(pattern,topic):
                    sender = re.findall(r'^(?:'+mountpoint+')?/?ECCI/.*/.*/.*/(.*)/.*',topic)[0]
                    return sender
        sender = re.findall(r'^ECCI/.*/.*/.*/(.*)/.*',topic)[0]
        return sender

    def send_ready(self):
        # 若为controller 应用容器，那么其就绪是通过brokermonitor发给ECCI系统
        if ECCI_APP_TYPE == "controller":
            logger.debug("controller send ready")
            # topic=f"toEdgeAI/{broker_id}/{APP_ID}/app/{CONTAINER_NAME}/{container_name}"
            payload = {"app_id":APP_ID,"container_name":CONTAINER_NAME,"agent_id":ECCI_AGENT_ID,"ready_type":ECCI_APP_TYPE}
            # logger.debug(f"toEdgeAI/{BROKER_ID}/con_cntr_ready")
            self._mqtt_client.publish(f"toEdgeAI/{BROKER_ID}/con_cntr_ready",json.dumps(payload),qos=2)
            # logger.debug("controller send ready")

            #------v2--用于restart==always执行------
            payload_v2 = {"type":"cmd","contents":{"cmd":"req_ready"}}
            self.publish(payload_v2)
            pass
        # 若为cpmponet 应用容器，那么其就绪是通过broker发给controller
        elif ECCI_APP_TYPE == "component":
            # logger.debug("component send ready")
            for container_name,broker_id in ECCI_APP_CONTROLLER_CONTAINER.items():
                pass
            topic=f"ECCI/{broker_id}/{APP_ID}/app/{CONTAINER_NAME}/{container_name}"
            logger.debug(topic)
            payload = {"type":"cmd","contents":{"cmd":"ready"}}
            self._mqtt_client.publish(topic,cPickle.dumps(payload),qos=2)
            logger.debug("component send ready")   

    def send_complete(self, ECCI_PERIOD_TYPE):
        #发送train/detect信号给ECCI系统
        logger.debug("send complete to ecci")
        payload = {"app_id":APP_ID,"container_name":CONTAINER_NAME,"agent_id":ECCI_AGENT_ID,"ready_type":ECCI_PERIOD_TYPE}
        self._mqtt_client.publish(f"toEdgeAI/{BROKER_ID}/con_cntr_ready",json.dumps(payload),qos=2)

    def send_data(self, msg_data, ECCI_ALGORITHM_TYPE):
        #发送train/detect信号给ECCI系统
        logger.debug("send complete to ecci")
        payload = {"type":"data", "algorithm_type":ECCI_ALGORITHM_TYPE, "data":msg_data, "app_id":APP_ID}
        print("===sdk send_data payload===",payload)
        try:
            self._mqtt_client.publish(f"toEdgeAI/{BROKER_ID}/send_data",json.dumps(payload),qos=2)
        except Exception as e:
            print("error = ",e)
        #若为train，则发送complete，通过brokermonitor发给ECCI系统
        # if ECCI_PERIOD_TYPE == "train":
        #     logger.debug("train send complete")
        #     payload = {"app_id":APP_ID,"container_name":CONTAINER_NAME,"agent_id":ECCI_AGENT_ID,"ready_type":ECCI_PERIOD_TYPE}
        #     self._mqtt_client.publish(f"toEdgeAI/{BROKER_ID}/con_cntr_ready",json.dumps(payload),qos=2)
        # #若为detect，则发送complish，通过brokermonitor发送给ECCI系统
        # elif ECCI_PERIOD_TYPE == "detect":
        #     for container_name,broker_id in ECCI_APP_CONTROLLER_CONTAINER.items():
        #         pass
        #     payload = {"app_id":APP_ID,"container_name":CONTAINER_NAME,"agent_id":ECCI_AGENT_ID,"ready_type":ECCI_PERIOD_TYPE}
        #     self._mqtt_client.publish(f"toEdgeAI/{BROKER_ID}/con_cntr_ready",json.dumps(payload),qos=2)
        #     logger.debug("detect send complish")   

    def initialize(self):
        
        try:
            # self._mqtt_client = mqtt.Client(client_id="jglgsy123", clean_session=False)
            self._mqtt_client = mqtt.Client()
            self._mqtt_client.enable_logger()
            self._mqtt_client.on_message = self._on_client_message
            self._mqtt_client.on_connect = self._on_client_connect
            self._mqtt_client.on_publish = self._on_client_publish
            if BROKER_USERNAME and BROKER_PASSWORD:
                self._mqtt_client.username_pw_set(BROKER_USERNAME,BROKER_PASSWORD)
            self._mqtt_client.connect(self._broker_ip, self._broker_port)
            self._mqtt_client.loop_forever()
        except TypeError:
            logger.error('Connect to mqtt broker error')
            return
if __name__ == "__main__":
    ecci_client = Client()
    from threading import Thread
    mqtt_thread = Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()