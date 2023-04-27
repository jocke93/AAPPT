import json
import os
import re
import time

from cv2 import cv2

from file_util import *
from drawing_util import draw_contours, draw_rectangles, draw_circles
from file_util import recreate_dir
from image_util import morph_image
from test_processing import get_page_qr_code, update_question_answers, update_circles_contours, \
    update_circles_filled_status, write_results, get_page_bar_code, get_borders, update_question, \
    BarcodeNotFoundException, update_answer_dict


def adjust_questions_answers_coordinates(questions, borders, page_num, factor):
    top_left = borders['TL']['r'][0], borders['TL']['r'][1]
    for zq in questions:
        for zone in zq['Zones']:
            if zone['Page'] == page_num:
                zone['details'] = None
                zone_rect = [int(zone['X'] * factor), int(zone['Y'] * factor),
                             int(zone['Width'] * factor), int(zone['Height'] * factor)]
                zone_rect = [top_left[0] + zone_rect[0], top_left[1] + zone_rect[1],
                             zone_rect[2], zone_rect[3]]
                zone['X'], zone['Y'], zone['Width'], zone['Height'] = zone_rect
        for answer in zq['Answers']:
            if 'Zone' in answer:
                answer['details'] = None
                if answer['Zone'] == page_num:
                    answer_rect = [int(answer['X'] * factor), int(answer['Y'] * factor),
                                   int(answer['Width'] * factor), int(answer['Height'] * factor)]
                    answer_rect = [top_left[0] + answer_rect[0], top_left[1] + answer_rect[1],
                                   answer_rect[2], answer_rect[3]]
                    answer['X'], answer['Y'], answer['Width'], answer['Height'] = answer_rect


def drawing_codes(image_write, student, page_key):
    # drawing BC rectangle
    if student['bc']:
        sid = student['bc']['id']
        bc_rect = student['bc']['rect']
        cv2.putText(image_write, text=f'SID {sid}',
                    org=(bc_rect[0], bc_rect[1] - bc_rect[3] // 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
        draw_rectangles(image_write, [bc_rect], (0, 0, 255), 5)

    # drawing QR rectangle
    if student['test'][page_key]:
        test_id = student['test'][page_key]['id']
        qr_rect = student['test'][page_key]['qr_rect']
        page_num = student['test'][page_key]['page']
        cv2.putText(image_write, text=f'id{test_id} p{page_num}',
                    org=(qr_rect[0], qr_rect[1] - qr_rect[3] // 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
        draw_rectangles(image_write, [qr_rect], (0, 0, 255), 5)


def drawing_zones_answers(image_write, zones, answers):
    # drawing questions contours
    draw_rectangles(image_write,
                    [[z['X'], z['Y'], z['Width'], z['Height']] for z in zones if 'details' in z and z['details']],
                    (0, 255, 0), 3)

    # drawing answers contours
    for a in answers:
        if 'details' in a and a['details'] or True:
            draw_rectangles(image_write,
                            [[a['X'], a['Y'], a['Width'], a['Height']]],
                            (255, 0, 0), 3)

        if a['details'] and a['details']['circles']:
            # drawing circles contours
            draw_circles(image_write,
                         [c for row in a['details']['circles']['c'] for c in row],
                         (0, 0, 255), 2)

            if a['details']['circles']['c_f']:
                # drawing filled status
                for i, row in enumerate(a['details']['circles']['c_f']):
                    for j, c_fill in enumerate(row):
                        (x, y), r = a['details']['circles']['c'][i][j]
                        # is_black = c_fill['is_black']
                        perc = c_fill['perc']
                        # cv2.putText(image_write, 'B' if is_black else 'W', org=(x - 2 * r, y),
                        # fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5, color=(0, 0, 255),
                        # thickness=3, lineType=cv2.LINE_AA)
                        cv2.putText(image_write, fr'{perc:.0f}',
                                    org=(x + r, y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5,
                                    # color=(255, 0, 255) if is_black else (255, 255, 0), thickness=3,
                                    color=(0, 0, 255), thickness=3,
                                    lineType=cv2.LINE_AA)


def process_test_pages(test_json, test_file, lower_threshold, upper_threshold, dpi, results_folder, logger):
    test_logger = None
    test_start_time = None
    ret_code = 0
    try:
        test_start_time = time.time()
        test_logger = setup_logger(logger['name'], logger['file'])
        recreate_dir(results_folder)
        # images_folder = fr'{results_folder}{os.sep}_images'
        # recreate_dir(images_folder)
        # lines_folder = fr'{results_folder}{os.sep}_lines'
        # recreate_dir(lines_folder)
        # test = dict()
        # student = {'bc': None, 'test': test}
        loaded, test_tif = cv2.imreadmulti(test_file)
        if not loaded:
            raise Exception('Failed to load test tif!')
        for page_iter, image in enumerate(test_tif):
            page_num = page_iter + 1
            page_key = f'_P_{page_num}'
            # test[page_key] = None
            questions = []
            try:
                # image = image[:, int(image.shape[1] * 0.05):int(image.shape[1] * 0.95)]

                # # FETCHING BARCODE
                # if student['bc'] is None and False:
                #     try:
                #         student['bc'], image = get_page_bar_code(image, test_logger)
                #     except BarcodeNotFoundException:
                #         pass

                # # FETCHING QR-CODE
                # test[page_key], image = get_page_qr_code(image, test_logger)

                # # Check page number
                # if test[page_key]['page'] != page_iter + 1 and False:
                #     raise Exception('QR page num and page_iter don\'t match!')

                # FIND QUESTIONS FOR CURRENT PAGE
                for question in test_json['Questions']:
                    if 'Zones' in question and any([z['Page'] == page_num for z in question['Zones']]):
                        if 'Images' not in question:
                            question['Images'] = [f'{page_key}.jpg']
                        questions.append(question)

                gray_inv, thresh_inv, image_lines = morph_image(image, test_logger)
                # cv2.imwrite(fr'{lines_folder}/T{page_iter + 1}.jpg', thresh_inv)
                # cv2.imwrite(fr'{lines_folder}/L{page_iter + 1}.jpg', image_lines)

                # FIND BORDERS, QUESTIONS ROI RECT AND POTENTIAL QUESTIONS
                borders, questions_roi_rect, possible_questions_contours = get_borders(image_lines, test_logger)

                # ADJUST QUESTIONS AND ANSWERS ZONES COORDINATES TO TOP LEFT BORDER
                adjust_questions_answers_coordinates(questions, borders, page_num, factor=dpi // 150)

                # image_write = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # drawing_codes(image_write, student, page_key)
                # draw_contours(image_write, [b['c'] for b in borders.values()], (0, 0, 255), 3)
                for question in questions:
                    zones = [z for z in question['Zones'] if z['Page'] == page_num]
                    answers = [a for a in question['Answers'] if 'Zone' in a and a['Zone'] == page_num]

                    ret_code |= update_question(question, zones,
                                                questions_roi_rect, possible_questions_contours,
                                                test_logger, test_json)

                    ret_code |= update_question_answers(image_lines, question, zones, answers, test_logger, test_json)

                    ret_code |= update_circles_contours(thresh_inv, question, answers, test_logger, test_json)

                    ret_code |= update_circles_filled_status(gray_inv, question, answers, lower_threshold,
                                                             upper_threshold, test_logger, test_json)

                    # drawing_zones_answers(image_write, zones, answers)

                    ret_code |= write_results(image, question, zones, answers, results_folder, test_logger, test_json)

                # cv2.imwrite(fr'{images_folder}{os.sep}{page_key}.jpg', image_write)

            except Exception as e:
                cv2.imwrite(fr'{results_folder}{os.sep}{page_key}.jpg', image)
                test_logger.exception(f'Page {page_key}: {e}')
                ret_code |= 512
                key = f'Page.{page_num}'
                if 'Errors' not in test_json:
                    test_json['Errors'] = {key: []}
                elif key not in test_json['Errors']:
                    test_json['Errors'][key] = []
                test_json['Errors'][key].append('Problem pri pronalasku graničnika.')
            finally:
                for question in questions:
                    try:
                        question['Zones'] = [z for z in question['Zones'] if z['Page'] != page_num]
                        if not len(question['Zones']):
                            question.pop('Zones', None)
                            if len(question['Images']) > 1:
                                question['Images'].pop(0)
                            # for a in [a for a in question['Answers'] if 'Zone' in a and a['Zone'] == 0]:
                            for a in [a for a in question['Answers'] if 'Zone' in a]:
                                update_answer_dict(a)
                    except Exception as e:
                        test_logger.exception(f'ERROR Question {question["Number"]} : {e}')
                        ret_code |= 1024
                        key = f'Question.{question["Number"]}'
                        if 'Errors' not in test_json:
                            test_json['Errors'] = {key: []}
                        elif key not in test_json['Errors']:
                            test_json['Errors'][key] = []
                        test_json['Errors'][key].append('Problem čišćenja rezultata.')

                # If at least one question zone was not found
                try:
                    if not questions or any([f'{page_key}.jpg' in q['Images'] for q in questions]):
                        cv2.imwrite(fr'{results_folder}{os.sep}{page_key}.jpg', image)
                except Exception as e:
                    ret_code |= 2048
                    key = f'Page.{page_num}'
                    if 'Errors' not in test_json:
                        test_json['Errors'] = {key: []}
                    elif key not in test_json['Errors']:
                        test_json['Errors'][key] = []
                    test_json['Errors'][key].append('Problem snimanja slike stranice.')

                test_logger.info(f'Tiff page {page_num} DONE.')

        # if student['bc'] is None:
        #     raise Exception('Bar-code not found!')

        json_keys = {'SerialNumber', 'Questions', 'Errors'}
        for key in list(test_json):
            if key not in json_keys:
                test_json.pop(key)
        json_name = re.split(r'[\\.]', test_file)[-2]
        test_json['SerialNumber'] = json_name.split('_')[0]
        with open(f'{json_name}.json', 'w') as file:
            json.dump(test_json, file, ensure_ascii=False)

    except Exception as e:
        test_logger.exception(f'Test {test_file}: {e}')
        ret_code |= 4096
    finally:
        if test_logger:
            test_logger.info(f'Test {test_file} DONE.')
            if test_start_time:
                test_end_time = time.time()
                test_logger.info(f'Time: {((test_end_time - test_start_time) * 1000):.2f} ms')

            handler = test_logger.handlers.pop(-1)
            handler.close()

        return ret_code


def process_tests(json_file=fr'.{os.sep}input-mat.json', parallel_processing=False,
                  tests_folder=fr'.{os.sep}tests', results_folder=fr'.{os.sep}results', logs_folder=fr'.{os.sep}logs'):
    logger = None
    start_time = None
    try:
        start_time = time.time()
        recreate_dir(logs_folder)
        logger = setup_logger('LOG', f'{logs_folder}{os.sep}LOG.txt')
        tests_logs = f'{logs_folder}{os.sep}tests_logs'
        recreate_dir(tests_logs)
        recreate_dir(results_folder)
        test_args = dict()
        with open(json_file, 'r') as file:
            test_json = json.load(file)
        for root, dirs, files in os.walk(tests_folder):
            for test in files:
                test_args[test] = [test_json,
                                   f'{tests_folder}{os.sep}{test}',
                                   300,
                                   f'{results_folder}{os.sep}{test[:test.rfind(".")]}',
                                   {'name': f'{test}', 'file': f'{tests_logs}{os.sep}LOG_{test[:test.rfind(".")]}.txt'}
                                   ]
            break

        if parallel_processing:
            from multiprocessing import Pool
            with Pool(len(test_args)) as pool:
                results = []
                for test in test_args:
                    res = pool.apply_async(process_test_pages, test_args[test])
                    results.append(res)
                [result.wait() for result in results]
        else:
            for test in test_args:
                process_test_pages(*test_args[test])
    except Exception as e:
        if logger:
            logger.exception(e)
        else:
            raise e
    finally:
        if logger and start_time:
            end_time = time.time()
            logger.info(f'TOTAL TIME: {(end_time - start_time):.2f} sec')


# split_tif_to_images('test'.tif', fp=1, lp=1)
if __name__ == '__main__':
    process_tests(parallel_processing=False)
