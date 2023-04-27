import os
import shutil
import logging


# def split_pdf_to_images(test_path, output=fr'.{os.sep}images', img_dpi=300, fp=2, lp=16):
#     from pdf2image import convert_from_path
#
#     try:
#         if os.path.isdir(output):
#             shutil.rmtree(output)
#         os.mkdir(output)
#         pages = convert_from_path(pdf_path=test_path, dpi=img_dpi, first_page=fp, last_page=lp,
#                                   poppler_path=r'C:\Users\Jocke\Downloads\poppler-21.10.0\Library\bin')
#         for i, page in enumerate(pages):
#             page.save(fr'{output}' + os.sep + f'{(i + fp):03}.png', 'PNG')
#     except Exception as e:
#         raise Exception(f'PDF split ERROR: {e}')
#     finally:
#         print('PDF split DONE.')


# def split_tif_to_images(test_path, output=fr'.{os.sep}images', fp=2, lp=16):
#     from PIL import Image, ImageSequence
#
#     try:
#         tif_image = Image.open(test_path)
#         for i, page in enumerate(ImageSequence.Iterator(tif_image)):
#             if i + 1 in range(fp, lp + 1):
#                 page.save(fr'{output}\{(i + 1):03}.png', 'PNG')
#     except Exception as e:
#         print('Tiff split ERROR: ', e)
#     finally:
#         print('Tiff split DONE.')


def recreate_dir(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
