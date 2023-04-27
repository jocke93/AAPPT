import math
import os
import re

import numpy as np
from cv2 import cv2
from numpy import mean
from pyzbar import pyzbar

from image_util import read_border_templates, scale_image, rotate_image, show_image
from shape_util import same_row_rect, same_col_rect, is_square_contour, is_circle_contour, \
    get_grid_cells, get_grid_circles, count_circle_filled_pixels, is_rect_inside_another, rect_overlapping_perc


class BarcodeNotFoundException(Exception):
    pass


def get_page_bar_code(image, test_logger):
    # QR:12105531010300018873039#02
    def check_bar_code_format(code):
        return re.match('(.*)', code)

    test_logger.info('===== PHASE 0 ==== (Bar-code)')
    h, w = image.shape[:2]
    x_low, x_high, y_low, y_high = int(w * 0.05), int(w * 0.45), int(h * 0.05), int(h * 0.2)
    roi = image[y_low:y_high, x_low:x_high]
    bar_code = pyzbar.decode(roi)
    if not len(bar_code):
        image = rotate_image(image, 180)
        roi = image[y_low:y_high, x_low:x_high]
        bar_code = pyzbar.decode(roi)
        if not len(bar_code):
            raise BarcodeNotFoundException()

    for bc in bar_code:
        if bc.type != 'CODE128':
            continue
        bc_data = bc.data.decode('utf-8')
        if match := check_bar_code_format(bc_data):
            student_id = match.groups()[0]
            test_logger.info(f'BC: Student {student_id}')
            bc_rect = [bc.rect[0] + x_low, bc.rect[1] + y_low, bc.rect[2], bc.rect[3]]
            tolerance = [int(bc_rect[2] * 0.1), int(bc_rect[3] * 0.1)]
            bc_rect = [bc_rect[0] - tolerance[0], bc_rect[1] - tolerance[1],
                       bc_rect[2] + 2 * tolerance[0], bc_rect[3] + 2 * tolerance[1]]
            return {'id': student_id, 'rect': bc_rect}, image
        else:
            raise Exception('Bar-code not in proper format!')
    raise BarcodeNotFoundException()


def get_page_qr_code(image, test_logger):
    # QR:12105531010300018873039#02
    def check_qr_code_format(code):
        return re.match(r'(\d{23})(?:[#$])(\d{2})', code)

    def get_angle_deg(qrcode):
        poly = sorted(qrcode.polygon, key=lambda p: p[0])
        poly = sorted(poly[:2], key=lambda p: p[1]) + sorted(poly[2:], key=lambda p: p[1], reverse=True)
        angle = math.atan2(poly[1].y - poly[0].y, poly[1].x - poly[0].x)
        angle_deg = np.rad2deg(angle)
        return angle_deg - 90

    test_logger.info('===== PHASE 1 ==== (QR)')
    h, w = image.shape[:2]
    x_low, x_high, y_low, y_high = int(w * 0.35), int(w * 0.65), int(h * 0.05), int(h * 0.18)
    roi = image[y_low:y_high, x_low:x_high]
    qr_codes = pyzbar.decode(roi)
    if not len(qr_codes):
        image = rotate_image(image, 180)
        roi = image[y_low:y_high, x_low:x_high]
        qr_codes = pyzbar.decode(roi)
        if not len(qr_codes):
            raise Exception('QR code not found!')

    for qr in qr_codes:
        if qr.type != 'QRCODE':
            continue
        qrcode_data = qr.data.decode('utf-8')
        if match := check_qr_code_format(qrcode_data):
            test_id, page_num = match.groups()
            test_logger.info(f'QR: Test-{test_id}, Page: {page_num}')
            rot_angle = get_angle_deg(qr)
            if rot_angle:
                image = rotate_image(image, rot_angle)

            # recalibrate QR rectangle after image rotation
            qr_points = np.array([[p[0], p[1]] for p in qr.polygon])
            ones = np.ones(shape=(len(qr_points), 1))
            points_ones = np.hstack([qr_points, ones])
            m_inv = cv2.getRotationMatrix2D(tuple(np.array(image.shape[1::-1]) / 2), rot_angle, 1.0)
            qr_points = m_inv.dot(points_ones.T).T
            qr_points = qr_points.astype(int) + (x_low, y_low)
            qr_rect = cv2.boundingRect(qr_points)
            tolerance = int(max(qr_rect[2], qr_rect[3]) * 0.1)
            qr_rect = [qr_rect[0] - tolerance, qr_rect[1] - tolerance,
                       qr_rect[2] + 2 * tolerance, qr_rect[3] + 2 * tolerance]
            return {'id': test_id, 'qr_rect': qr_rect, 'page': int(page_num)}, image
        else:
            raise Exception('QR code not in proper format!')
    raise Exception('QR code not found!')


def get_borders(image, test_logger):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[1])  # sort by y coordinate ascending

    test_logger.info('===== PHASE 2 ==== (Borders)')
    test_logger.info(f'Contours found: {len(contours)}')
    # Processing 4 borders contours
    border_images = read_border_templates()
    borders = dict.fromkeys(border_images.keys())
    for b in borders:
        borders[b] = {'c': None, 'r': None, 'val': 0}

    borders_indexes = dict.fromkeys(borders.keys())
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        dim = w if w < h else h
        region = image[y: y + dim, x: x + dim]
        region = scale_image(region, border_images['TL'].shape[0] / region.shape[0])
        max_val = 0
        max_key = None
        for key in borders.keys():
            result = cv2.matchTemplate(region, border_images[key], cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(result)
            if val > max_val:
                max_val = val
                max_key = key

        if max_key is None:
            continue
        elif max_val > borders[max_key]['val']:
            borders[max_key]['c'] = c
            borders[max_key]['r'] = [x, y, w, h]
            borders[max_key]['val'] = max_val
            borders_indexes[max_key] = i
            # print(max_val)
            # show_image('b', region, 1, True)

    for key in borders:
        if borders[key]['c'] is None:
            raise Exception('Could not find borders!')

    # print(same_row_rect(borders['TL']['r'], borders['TR']['r']))
    # print(same_row_rect(borders['BL']['r'], borders['BR']['r']))
    # print(same_col_rect(borders['TL']['r'], borders['BL']['r']))
    # print(same_col_rect(borders['TR']['r'], borders['BR']['r']))
    # print(borders['TL']['r'])
    # print(borders['TR']['r'])
    # print(borders['BL']['r'])
    # print(borders['BR']['r'])
    if not (same_row_rect(borders['TL']['r'], borders['TR']['r']) and
            same_row_rect(borders['BL']['r'], borders['BR']['r']) and
            same_col_rect(borders['TL']['r'], borders['BL']['r']) and
            same_col_rect(borders['TR']['r'], borders['BR']['r'])):
        raise Exception('Could not find borders!')

    roi_width = borders['TR']['r'][0] - borders['TL']['r'][0] + borders['TL']['r'][2]
    roi_height = borders['BL']['r'][1] - borders['TL']['r'][1] + borders['TL']['r'][1]
    questions_roi_rect = [borders['TL']['r'][0], borders['TL']['r'][1], roi_width, roi_height]

    potential_questions = [c for i, c in enumerate(contours) if i not in borders_indexes.values()]
    return borders, questions_roi_rect, potential_questions


def update_question(question, zones, questions_roi_rect, potential_questions, test_logger, test_json):
    test_logger.info('===== PHASE 3 ==== (Questions regions)')
    # Processing question contours
    ret_code = 0
    for zone in zones:
        try:
            for i, c in enumerate(potential_questions):
                if is_square_contour(c):
                    rect = cv2.boundingRect(c)
                    # potential question, >= 80% w
                    if is_rect_inside_another(rect, questions_roi_rect) and rect[2] > questions_roi_rect[2] * 0.8:
                        z_rect = [zone['X'], zone['Y'], zone['Width'], zone['Height']]
                        if rect_overlapping_perc(rect, z_rect) > 0.85:
                            zone['X'], zone['Y'], zone['Width'], zone['Height'] = rect
                            zone['details'] = True
                            del potential_questions[i]
                            break
            break
        except Exception as e:
            zone['details'] = False
            test_logger.exception(f'ERROR Question {question["Number"]} : {e}')
            ret_code |= 256
            key = f'Question.{question["Number"]}'
            if 'Errors' not in test_json:
                test_json['Errors'] = {key: []}
            elif key not in test_json['Errors']:
                test_json['Errors'][key] = []
            test_json['Errors'][key].append('Problem pri ukrštanju zone pitanja sa zonerom.')
        finally:
            test_logger.info(f'Question {question["Number"]} | found: {zone["details"]}')
    return ret_code


def update_question_answers(image, question, zones, answers, test_logger, test_json):
    test_logger.info('===== PHASE 4 ==== (Answers regions)')
    ret_code = 0
    for zone in zones:
        possible_answers = []
        try:
            if not zone['details']:
                continue
            x, y, w, h = zone['X'], zone['Y'], zone['Width'], zone['Height']
            roi = image[y:y + h, x: x + w]
            contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_list = sorted([(i, c, h) for i, (c, h) in enumerate(zip(contours, hierarchy[0]))],
                                   key=lambda el: cv2.contourArea(el[1]), reverse=True)
            # only works if contours sorted from largest to smallest by size
            question_internal_contour = contours_list[1]

            possible_answer_region_ids = dict()
            for e in contours_list:
                if is_square_contour(e[1]):  # is square
                    rect = list(cv2.boundingRect(e[1]))
                    rect[0] += x
                    rect[1] += y
                    if e[2][3] not in possible_answer_region_ids:  # if parent id not in ...
                        possible_answer_region_ids[e[2][3]] = [rect]
                    else:
                        possible_answer_region_ids[e[2][3]].append(rect)

            answer_regions_ids = set()
            for e in contours_list:
                if e[0] in possible_answer_region_ids and e[0] not in answer_regions_ids and \
                        is_square_contour(e[1]) and e[2][3] == question_internal_contour[0]:
                    # is square and parent is question
                    try:
                        grid_cells = get_grid_cells(possible_answer_region_ids[e[0]])
                    except Exception as e:
                        test_logger.info(f'{e}')
                        continue
                    rect = list(cv2.boundingRect(e[1]))
                    rect[0] += x
                    rect[1] += y
                    answer_regions_ids.add(e[0])
                    possible_answers.append({'rect': rect,
                                             'cells': grid_cells,
                                             'dim': (len(grid_cells), len(grid_cells[0]))})
        except Exception as e:
            test_logger.exception(f'ERROR Question {question["Number"]} : {e}')
            ret_code |= 32
            key = f'Question.{question["Number"]}'
            if 'Errors' not in test_json:
                test_json['Errors'] = {key: []}
            elif key not in test_json['Errors']:
                test_json['Errors'][key] = []
            test_json['Errors'][key].append('Problem pri pronalasku potencijalnih zona odgovora.')
            for answer in answers:
                answer['details'] = None
            continue

        for answer in [a for a in answers if a['X']]:
            try:
                check = False
                answer_rect = [answer['X'], answer['Y'], answer['Width'], answer['Height']]
                for i, possible in enumerate(possible_answers):
                    # print(answer['Zone'])
                    # print(possible['rect'], answer_rect, rect_overlapping_perc(possible['rect'], answer_rect))
                    if rect_overlapping_perc(possible['rect'], answer_rect) > 0.85:
                        answer['X'], answer['Y'], answer['Width'], answer['Height'] = possible['rect']
                        answer['details'] = possible
                        del possible_answers[i]
                        test_logger.info(f'Question {question["Number"]} | Answer {answer["Number"]} '
                                         f'| Candidate answer table {possible["dim"]}')
                        check = True
                        break
                if not check:
                    ret_code |= 64
                    key = f'Question.{question["Number"]}.Answer.{answer["Number"]}'
                    if 'Errors' not in test_json:
                        test_json['Errors'] = {key: []}
                    elif key not in test_json['Errors']:
                        test_json['Errors'][key] = []
                    test_json['Errors'][key].append('Problem pronalaska preklapanja '
                                                    'potencijalne mreže zone odgovora sa zonerom.')
            except Exception as e:
                test_logger.exception(f'ERROR Question {question["Number"]} | Answer {answer["Number"]} : {e}')
                ret_code |= 128
                key = f'Question.{question["Number"]}.Answer.{answer["Number"]}'
                if 'Errors' not in test_json:
                    test_json['Errors'] = {key: []}
                elif key not in test_json['Errors']:
                    test_json['Errors'][key] = []
                test_json['Errors'][key].append('Problem pri ukrštanju zone odgovora sa zonerom.')
                answer['details'] = None
    return ret_code


def update_circles_contours(image, question, answers, test_logger, test_json):
    test_logger.info('===== PHASE 5 ==== (Circles regions)')
    ret_code = 0
    for answer in answers:
        try:
            if not answer['details']:
                continue
            test_logger.info(f'Checking: Question {question["Number"]} | Answer {answer["Number"]} | '
                             f'table {answer["details"]["dim"]} ...')
            answer['details'].update({'circles': {'c': None}})
            x, y, w, h = answer['details']['rect']
            roi = image[y:y + h, x: x + w]
            contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_list = sorted([(ident, c, h) for ident, (c, h) in enumerate(zip(contours, hierarchy[0]))],
                                   key=lambda el: cv2.contourArea(el[1]), reverse=True)

            if not len(contours_list):
                answer['details'] = None
                raise Exception()

            rows_cnt, cols_cnt = answer['details']['dim']
            table_cells_count = rows_cnt * cols_cnt
            thresh_cnt = table_cells_count // 2 if table_cells_count // 2 > 0 else 1
            if rows_cnt > 2 or cols_cnt > 2:
                thresh_cnt = max(thresh_cnt, (rows_cnt - 1) * (cols_cnt - 1))

            try:
                table_cells_circles = []
                parent_rect_cont = contours_list[0]
                parent_rect = cv2.boundingRect(parent_rect_cont[1])
                # largest contour without parent
                if is_square_contour(parent_rect_cont[1]) and parent_rect_cont[2][3] == -1 and \
                        parent_rect[2] > 0.9 * w and parent_rect[3] > 0.9 * h:
                    table_cells = [
                        (ident, cont, hie)
                        for ident, cont, hie in contours_list
                        if is_square_contour(cont) and hie[3] == parent_rect_cont[0]
                    ]
                    if len(table_cells) != table_cells_count:
                        raise Exception(f'Expected {table_cells_count}, got {len(table_cells)} cells instead.')
                    try:
                        get_grid_cells([cv2.boundingRect(c) for _, c, _ in table_cells])
                    except Exception as e:
                        raise Exception(f'Table is not grid: {e}')

                    table_cells = dict(zip(
                        [ident for ident, _, _ in table_cells],
                        [cv2.boundingRect(cont) for _, cont, _ in table_cells])
                    )
                    for _, cont, hie in contours_list:
                        if hie[3] in table_cells:
                            rect = list(table_cells[hie[3]])
                            rect[0] += x
                            rect[1] += y
                            table_cells_circles.append((cont, cv2.minEnclosingCircle(cont), rect))
                            table_cells.pop(hie[3])
                else:
                    raise Exception('No parentless rectangular contour. ')
            except Exception as e:
                test_logger.info(f'{e} Going to give another try...')
                table_cells = [cell for row in answer['details']['cells'] for cell in row]
                # LARGEST CONTOURS INSIDE TABLE CELLS
                table_cells_circles = []
                for _, cont, _ in contours_list:
                    rect = cv2.boundingRect(cont)
                    rect = [rect[0] + x, rect[1] + y, rect[2], rect[3]]
                    for i, cell in enumerate(table_cells):
                        if is_rect_inside_another(rect, cell):
                            del table_cells[i]
                            table_cells_circles.append((cont, cv2.minEnclosingCircle(cont), cell))
                            break

            if not len(table_cells_circles):
                answer['details'] = None
                raise Exception()

            # CIRCLE-LIKE CONTOURS CENTERED INSIDE TABLE CELL
            table_cells_circles_copy = table_cells_circles[:]
            table_cells_circles = []
            for cont, circle, cell in table_cells_circles_copy:
                if not is_circle_contour(cont, 0.5):
                    continue
                if cell[0] + cell[2] * 0.4 <= circle[0][0] + x <= cell[0] + cell[2] * 0.6 and \
                        cell[1] + cell[3] * 0.4 <= circle[0][1] + y <= cell[1] + cell[3] * 0.6:
                    table_cells_circles.append((cont, circle, cell))

            if not len(table_cells_circles):
                answer['details'] = None
                raise Exception()

            # PROPER CIRCLES MEAN RADIUS (0.85 default factor)
            radius_list = [circle[1] for cont, circle, _ in table_cells_circles if is_circle_contour(cont)][:thresh_cnt]
            if len(radius_list):
                mean_r = mean(radius_list)
            else:
                mean_r = mean([circle[1] for _, circle, _ in table_cells_circles])
            low_bound_r, high_bound_r = 0.9 * mean_r, 1.5 * mean_r
            table_cells_circles = [c for c in table_cells_circles if low_bound_r <= c[1][1] <= high_bound_r]
            min_r = min([c[1][1] for c in table_cells_circles])

            table_cells_circles_copy = table_cells_circles[:]
            table_cells_circles = []
            for _, c, cell in table_cells_circles_copy:
                if c[1] > min_r * 1.05:  # RECALIBRATE
                    table_cells_circles.append(((cell[0] + cell[2] // 2, cell[1] + cell[3] // 2), int(min_r)))
                else:
                    table_cells_circles.append(((int(c[0][0] + x), int(c[0][1] + y)), int(c[1])))
            table_cells_circles = table_cells_circles[:thresh_cnt]

            if not len(table_cells_circles) or len(table_cells_circles) != thresh_cnt:
                answer['details'] = None
                raise Exception()

            test_logger.info(f'Potential circles (before grid check): {len(table_cells_circles)}')
            try:
                circles_grid = get_grid_circles(table_cells_circles)
                dim = (answer['Columns'], answer['Rows']) \
                    if answer['Transposed'] else (answer['Rows'], answer['Columns'])
                if dim != (len(circles_grid), len(circles_grid[0])):
                    raise Exception('Circles grid zoner validation failed!')
            except Exception as e:
                test_logger.info(f'{e}')
                ret_code |= 8
                key = f'Question.{question["Number"]}.Answer.{answer["Number"]}'
                if 'Errors' not in test_json:
                    test_json['Errors'] = {key: []}
                elif key not in test_json['Errors']:
                    test_json['Errors'][key] = []
                test_json['Errors'][key].append('Problem validacije mreze krugova.')
                answer['details'] = None
                continue
            test_logger.info(f'Circles: {len(table_cells_circles)} | {thresh_cnt} of {table_cells_count} needed')
            answer['details']['circles']['c'] = circles_grid
            answer['details']['circles']['mean_r'] = mean_r
            answer['details']['circles']['dim'] = dim
            test_logger.info(f'CONFIRMED - Grid dimension: {answer["details"]["circles"]["dim"]}')
        except Exception as e:
            test_logger.exception(f'ERROR Question {question["Number"]} | Answer {answer["Number"]} : {e}')
            ret_code |= 16
            key = f'Question.{question["Number"]}.Answer.{answer["Number"]}'
            if 'Errors' not in test_json:
                test_json['Errors'] = {key: []}
            elif key not in test_json['Errors']:
                test_json['Errors'][key] = []
            test_json['Errors'][key].append('Problem validacije zone krugova.')
            answer['details'] = None
    return ret_code


def update_circles_filled_status(image, question, answers, lower_threshold, upper_threshold, test_logger, test_json):
    test_logger.info('===== PHASE 6 ==== (Filled circles)')
    ret_code = 0
    radius_factor = 0.75
    for answer in answers:
        try:
            if not answer['details'] or not len(answer['details']['circles']['c']):
                continue
            test_logger.info(f'Checking: Question {question["Number"]} | Answer {answer["Number"]} | '
                             f'table {answer["details"]["dim"]} ...')

            answer['details']['circles']['c_f'] = []
            perc_list = []
            for row in answer['details']['circles']['c']:
                perc_list.append([])
                for c in row:
                    x, y, r = c[0][0], c[0][1], c[1]
                    circle_seg = image[y - r:y + r, x - r:x + r]
                    black_pixels = count_circle_filled_pixels(circle_seg, radius_factor)
                    r *= radius_factor
                    area = r * r * math.pi
                    perc = 1. * black_pixels / area
                    perc_list[-1].append(perc)
                    # black_pixels_alg = cv2.countNonZero(circle_seg)
                    # area = circle_seg.shape[0] * circle_seg.shape[1]
                    # perc = black_pixels / area / math.pi * 4 * 100

            for circ_i in range(len(perc_list)):
                answer['details']['circles']['c_f'].append([])
                for circ_j in range(len(perc_list[0])):
                    perc = perc_list[circ_i][circ_j]
                    perc *= 100
                    is_black = None if lower_threshold < perc < upper_threshold \
                        else True if perc > upper_threshold \
                        else False
                    answer['details']['circles']['c_f'][-1].append({'is_black': is_black, 'perc': round(perc, 2)})

            test_logger.info(f'Filled matrix: {answer["details"]["circles"]["c_f"]}')
        except Exception as e:
            test_logger.exception(f'ERROR Question {question["Number"]} | Answer {answer["Number"]} : {e}')
            ret_code |= 4
            key = f'Question.{question["Number"]}.Answer.{answer["Number"]}'
            if 'Errors' not in test_json:
                test_json['Errors'] = {key: []}
            elif key not in test_json['Errors']:
                test_json['Errors'][key] = []
            test_json['Errors'][key].append('Problem popunjenosti kruga.')
            answer['details'] = None
    return ret_code


def update_answer_dict(ans):

    result_answer_dict = {
        'Number': ans['Number'],
        'ManualReviewNeeded': False
        if 'details' in ans and ans['details'] and 'circles' in ans['details'] and 'c_f' in ans['details']['circles'] and
           not any([c['is_black'] is None for row in ans['details']['circles']['c_f'] for c in row])
        else True,
        'DetectedValues': [[c['is_black'] in {True, None} for c in row] for row in ans['details']['circles']['c_f']]
        if 'details' in ans and ans['details'] and 'circles' in ans['details'] and 'c_f' in ans['details']['circles']
        else None
    }
    ans.clear()
    ans.update(result_answer_dict)


def write_results(image, question, zones, answers, results_folder, test_logger, test_json):
    test_logger.info('===== PHASE 7 ==== (Results)')
    ret_code = 0

    def add_image_name(q, img_name):
        if 'Images' not in q:
            q['Images'] = [img_name]
        else:
            q['Images'].append(img_name)

    for zone in zones:
        try:
            if not zone['details']:
                continue
            x, y, w, h = zone['X'], zone['Y'], zone['Width'], zone['Height']
            region = image[y:y + h, x:x + w]
            image_name = f'Q{question["Number"]}_P{zone["Page"]}.jpg'
            add_image_name(question, image_name)
            cv2.imwrite(f'{results_folder}{os.sep}{image_name}', region)
            test_logger.info(f'Question {question["Number"]} | Page {zone["Page"]} | Zoner: {not zone["details"]}')
            for answer in answers:
                try:
                    x, y, w, h = answer['X'], answer['Y'], answer['Width'], answer['Height']
                    region = image[y:y + h, x:x + w]
                    image_name = f'Q{question["Number"]}_A{answer["Number"]}.jpg'
                    cv2.imwrite(f'{results_folder}{os.sep}{image_name}', region)
                    add_image_name(question, image_name)
                    test_logger.info(f'Question {question["Number"]} | Answer {answer["Number"]} | '
                                     f'Zoner: {not answer["details"]}')
                    update_answer_dict(answer)
                except Exception as e:
                    test_logger.exception(f'ERROR Question {question["Number"]} | Answer {answer["Number"]} : {e}')
                    ret_code |= 1
                    key = f'Question.{question["Number"]}.Answer.{answer["Number"]}'
                    if 'Errors' not in test_json:
                        test_json['Errors'] = {key: []}
                    elif key not in test_json['Errors']:
                        test_json['Errors'][key] = []
                    test_json['Errors'][key].append('Problem upisa zone odgovora.')
        except Exception as e:
            test_logger.exception(f'ERROR Question {question["Number"]} : {e}')
            ret_code |= 2
            key = f'Question.{question["Number"]}'
            if 'Errors' not in test_json:
                test_json['Errors'] = {key: []}
            elif key not in test_json['Errors']:
                test_json['Errors'][key] = []
            test_json['Errors'][key].append('Problem upisa zone pitanja.')
    return ret_code
