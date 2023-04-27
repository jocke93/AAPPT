import json
import os
import shutil
import sys
import time
from multiprocessing import Process

import paramiko
import stomp
from paramiko.client import SSHClient
from scp import SCPClient

from file_util import recreate_dir
from processing import process_test_pages


def connect_and_subscribe(connection):
    global params
    try:
        connection.connect(params['StompUsername'], params['StompPassword'], wait=True)
        connection.subscribe(destination=fr'/queue/{params["StompDown"]}', id=0, ack='client-individual')
        # headers={'activemq.prefetchSize': 1})
    except Exception as e:
        raise e


def attempt_connection(connection):
    global flag
    flag = True
    while flag:
        try:
            connect_and_subscribe(connection)
            flag = False
        except Exception:
            minutes = params['StompRetryMinutes']
            print(f'MQ Server unavailable. Retrying again in {minutes} min...')
            time.sleep(minutes * 60)


class Listener(stomp.ConnectionListener):
    def __init__(self, connection):
        self.conn = connection

    def on_error(self, frame):
        print(f'Received error: {frame.body}')

    def on_message(self, frame):
        print(f'Received message: {frame.body}')
        message_id = frame.headers['message-id']

        results_folder = ''
        json_path = ''
        test_name = ''
        test_path = ''
        global params
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=params['SftpHostname'], port=params['SftpPort'],
                        username=params['SftpUsername'], password=params['SftpPassword'])
            scp = SCPClient(ssh.get_transport())
            msg_json = json.loads(frame.body)

            if len(sys.argv) > 1 and sys.argv[1] == '1':
                raise Exception()

            test_name = msg_json["Path"][msg_json["Path"].rfind(r"/") + 1:]
            test_path = msg_json["Path"][:msg_json["Path"].rfind(r"/") + 1]
            json_path = test_name.split('.')[0] + '.json'

            scp.get(msg_json['Path'])
            scp.get(msg_json['Configuration'], json_path)
            scp.close()

            print(json_path)
            print(test_name)

            with open(json_path, 'r') as json_file:
                test_json = json.load(json_file)

            results_folder = f'results{os.sep}{test_name[:test_name.rfind(".")]}'
            args = [test_json,
                    test_name,
                    msg_json['LowerThreshold'],
                    msg_json['UpperThreshold'],
                    params['TestDpi'],
                    results_folder,
                    {'name': f'{test_name}', 'file': f'logs{os.sep}LOG_{test_name[:test_name.rfind(".")]}.txt'}
                    ]
            ret_code = process_test_pages(*args)

            if ret_code not in [4096]:
                scp = SCPClient(ssh.get_transport())
                scp.put(f'{test_name[:test_name.rfind(".")]}.json',
                        f'{test_path}'f'{test_name[:test_name.rfind(".")]}.json')

                for root, dirs, files in os.walk(results_folder):
                    for image in files:
                        scp.put(f'{results_folder}{os.sep}{image}', f'{test_path}{image}')
                    break

                scp.close()
                msg = json.dumps({'path': test_path, 'status': ret_code})
                self.conn.send(body=msg, destination=fr'/queue/{params["StompUp"]}')
            else:
                msg = {'Path': test_path, 'Status': ret_code, 'Error': 'Problem sa testom.'}
                self.conn.send(body=msg, destination=fr'/queue/{params["StompUp"]}')
        except Exception as e:
            ret_code = 8192
            print(f'Exception:{e}')
            msg = {'Path': test_path, 'Status': ret_code, 'Error': 'Problem u komunikaciji.'}
            self.conn.send(body=msg, destination=fr'/queue/{params["StompUp"]}')
        finally:
            global flag
            flag = True

        if len(results_folder) and os.path.isdir(results_folder):
            shutil.rmtree(results_folder)

        if len(test_name) and os.path.isfile(test_name):
            os.remove(test_name)

        if len(json_path) and os.path.isfile(json_path):
            os.remove(json_path)

        self.conn.ack(message_id, 0)
        print('Message send')

    def on_connected(self, frame):
        print('Connected')

    def on_disconnected(self):
        print('Disconnected')
        attempt_connection(self.conn)


def worker_job():
    conn = stomp.Connection(host_and_ports=[(params['StompHost'], params['StompPort'])],
                            heartbeats=(params['StompHeartbeats'], params['StompHeartbeats']))
    conn.set_listener('', Listener(conn))

    attempt_connection(conn)

    global flag
    flag = True
    while flag:
        try:
            time.sleep(60)
        except Exception:
            attempt_connection(conn)


flag = True
if __name__ == '__main__':
    params = {}
    with open('config.json', 'r') as json_file:
        params_json = json.load(json_file)
        print(json.dumps(params_json, indent=2))

    # recreate_dir('logs')
    # recreate_dir('results')
    # processes = [Process(target=worker_job) for _ in range(params['NumOfWorkers'])]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()
