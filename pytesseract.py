#!/usr/bin/env python

'''
Python-tesseract. For more information: https://github.com/madmaze/pytesseract
'''
tesseract_cmd = 'C:\\Tesseract-OCR\\tesseract'
try:
    import Image
except ImportError:
    from PIL import Image

import os
import sys
import subprocess
from pkgutil import find_loader
import tempfile
import shlex
from glob import iglob
from distutils.version import LooseVersion

numpy_installed = find_loader('numpy') is not None
if numpy_installed:
    from numpy import ndarray

# CHANGE THIS IF TESSERACT IS NOT IN YOUR PATH, OR IS NAMED DIFFERENTLY
tesseract_cmd = 'C:\\Bhaskar_backup\\Bhaskar\\Softwares\\Tesseract-OCR\\tesseract' #'C:\\Tesseract-OCR\\tesseract' #'C:\\Tesseract-OCR\\tesseract'
print("tesseract_cmd")
RGB_MODE = 'RGB'
OSD_KEYS = {
    'Page number': ('page_num', int),
    'Orientation in degrees': ('orientation', int),
    'Rotate': ('rotate', int),
    'Orientation confidence': ('orientation_conf', float),
    'Script': ('script', str),
    'Script confidence': ('script_conf', float)
}


class Output:
    STRING = "string"
    BYTES = "bytes"
    DICT = "dict"


class TesseractError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.args = (status, message)


class TesseractNotFoundError(OSError):
    def __init__(self):
        super(TesseractNotFoundError, self).__init__(
            tesseract_cmd + " is not installed or it's not in your path"
        )


class TSVNotSupported(Exception):
    def __init__(self):
        super(TSVNotSupported, self).__init__(
            'tsv output not supported. It requires Tesseract >= 3.05'
        )


def run_once(func):
    def wrapper(*args, **kwargs):
        if wrapper._result is wrapper:
            wrapper._result = func(*args, **kwargs)
        return wrapper._result

    wrapper._result = wrapper
    return wrapper


def get_errors(error_string):
    return u' '.join(
        line for line in error_string.decode('utf-8').splitlines()
    ).strip()


def cleanup(temp_name):
    ''' Tries to remove files by filename wildcard path. '''
    for filename in iglob(temp_name + '*' if temp_name else temp_name):
        try:
            os.remove(filename)
        except OSError:
            pass


def prepare(image):
    if isinstance(image, Image.Image):
        return image

    if numpy_installed and isinstance(image, ndarray):
        return Image.fromarray(image)

    raise TypeError('Unsupported image object')


def save_image(image):
    image = prepare(image)

    img_extension = image.format
    if image.format not in {'JPEG', 'PNG', 'TIFF', 'BMP', 'GIF'}:
        img_extension = 'PNG'

    if not image.mode.startswith(RGB_MODE):
        image = image.convert(RGB_MODE)

    if 'A' in image.getbands():
        # discard and replace the alpha channel with white background
        background = Image.new(RGB_MODE, image.size, (255, 255, 255))
        background.paste(image, (0, 0), image)
        image = background

    temp_name = tempfile.mktemp(prefix='tess_')
    input_file_name = temp_name + os.extsep + img_extension
    image.save(input_file_name, format=img_extension, **image.info)
    return temp_name, input_file_name


def subprocess_args(include_stdout=True):
    # See https://github.com/pyinstaller/pyinstaller/wiki/Recipe-subprocess
    # for reference and comments.

    kwargs = {
        'stdin': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'startupinfo': None,
        'env': None
    }

    if hasattr(subprocess, 'STARTUPINFO'):
        kwargs['startupinfo'] = subprocess.STARTUPINFO()
        kwargs['startupinfo'].dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs['env'] = os.environ

    if include_stdout:
        kwargs['stdout'] = subprocess.PIPE

    return kwargs


def run_tesseract(input_filename,
                  output_filename_base,
                  extension,
                  lang,
                  config='',
                  nice=0):
    command = []

    if not sys.platform.startswith('win32') and nice != 0:
        command += ('nice', '-n', str(nice))

    command += (tesseract_cmd, input_filename, output_filename_base)

    if lang is not None:
        command += ('-l', lang)

    command += shlex.split(config)

    if extension != 'box':
        command.append(extension)

    proc = subprocess.Popen(command, **subprocess_args())
    status_code, error_string = proc.wait(), proc.stderr.read()
    proc.stderr.close()

    if status_code:
        raise TesseractError(status_code, get_errors(error_string))

    return True


def run_and_get_output(image,
                       extension,
                       lang=None,
                       config='',
                       nice=None,
                       return_bytes=False):

    temp_name, input_filename = '', ''
    try:
        temp_name, input_filename = save_image(image)
        kwargs = {
            'input_filename': input_filename,
            'output_filename_base': temp_name + '_out',
            'extension': extension,
            'lang': lang,
            'config': config,
            'nice': nice
        }
        try:
            run_tesseract(**kwargs)
            filename = kwargs['output_filename_base'] + os.extsep + extension
            with open(filename, 'rb') as output_file:
                if return_bytes:
                    return output_file.read()
                return output_file.read().decode('utf-8').strip()
        except OSError:
            raise TesseractNotFoundError()
    finally:
        cleanup(temp_name)


def file_to_dict(tsv, cell_delimiter, str_col_idx):
    result = {}
    rows = [row.split(cell_delimiter) for row in tsv.split('\n')]
    if not rows:
        return result

    header = rows.pop(0)
    if len(rows[-1]) < len(header):
        # Fixes bug that occurs when last text string in TSV is null, and
        # last row is missing a final cell in TSV file
        rows[-1].append('')

    if str_col_idx < 0:
        str_col_idx += len(header)

    for i, head in enumerate(header):
        result[head] = [
            int(row[i]) if i != str_col_idx else row[i] for row in rows
        ]

    return result


def is_valid(val, _type):
    if _type is int:
        return val.isdigit()

    if _type is float:
        try:
            float(val)
            return True
        except ValueError:
            return False

    return True


def osd_to_dict(osd):
    return {
        OSD_KEYS[kv[0]][0]: OSD_KEYS[kv[0]][1](kv[1]) for kv in (
            line.split(': ') for line in osd.split('\n')
        ) if len(kv) == 2 and is_valid(kv[1], OSD_KEYS[kv[0]][1])
    }


@run_once
def get_tesseract_version():
    '''
    Returns a string containing the Tesseract version.
    '''
    try:
        return LooseVersion(
            subprocess.check_output(
                [tesseract_cmd, '--version'], stderr=subprocess.STDOUT
            ).decode('utf-8').split()[1]
        )
    except OSError:
        raise TesseractNotFoundError()


def image_to_string(image,
                    lang=None,
                    config='',
                    nice=0,
                    boxes=False,
                    output_type=Output.STRING):
    '''
    Returns the result of a Tesseract OCR run on the provided image to string
    '''
    if boxes:
        # Added for backwards compatibility
        print('\nWarning: Argument \'boxes\' is deprecated and will be removed'
              ' in future versions. Use function image_to_boxes instead.\n')
        return image_to_boxes(image, lang, config, nice, output_type)

    if output_type == Output.DICT:
        return {'text': run_and_get_output(image, 'txt', lang, config, nice)}
    elif output_type == Output.BYTES:
        return run_and_get_output(image, 'txt', lang, config, nice, True)

    return run_and_get_output(image, 'txt', lang, config, nice)


def image_to_boxes(image,
                   lang=None,
                   config='',
                   nice=0,
                   output_type=Output.STRING):
    '''
    Returns string containing recognized characters and their box boundaries
    '''
    config += ' batch.nochop makebox'

    if output_type == Output.DICT:
        box_header = 'char left bottom right top page\n'
        return file_to_dict(
            box_header + run_and_get_output(
                image, 'box', lang, config, nice), ' ', 0)
    elif output_type == Output.BYTES:
        return run_and_get_output(image, 'box', lang, config, nice, True)

    return run_and_get_output(image, 'box', lang, config, nice)


def image_to_data(image,
                  lang=None,
                  config='',
                  nice=0,
                  output_type=Output.STRING):
    '''
    Returns string containing box boundaries, confidences,
    and other information. Requires Tesseract 3.05+
    '''

    # TODO: we can use decoration for this check
#    if get_tesseract_version() < '3.05':
#        raise TSVNotSupported()

    if output_type == Output.DICT:
        return file_to_dict(
            run_and_get_output(image, 'tsv', lang, config, nice), '\t', -1)
    elif output_type == Output.BYTES:
        return run_and_get_output(image, 'tsv', lang, config, nice, True)

    return run_and_get_output(image, 'tsv', lang, config, nice)


def image_to_osd(image,
                 lang=None,
                 config='',
                 nice=0,
                 output_type=Output.STRING):
    '''
    Returns string containing the orientation and script detection (OSD)
    '''
    config += ' --psm 0'

    if output_type == Output.DICT:
        return osd_to_dict(
            run_and_get_output(image, 'osd', lang, config, nice))
    elif output_type == Output.BYTES:
        return run_and_get_output(image, 'osd', lang, config, nice, True)

    return run_and_get_output(image, 'osd', lang, config, nice)


def main():
    if len(sys.argv) == 2:
        filename, lang = sys.argv[1], None
    elif len(sys.argv) == 4 and sys.argv[1] == '-l':
        filename, lang = sys.argv[3], sys.argv[2]
    else:
        sys.stderr.write('Usage: python pytesseract.py [-l lang] input_file\n')
        exit(2)

    try:
        print(image_to_string(Image.open(filename), lang=lang))
    except IOError:
        sys.stderr.write('ERROR: Could not open file "%s"\n' % filename)
        exit(1)


if __name__ == '__main__':
    main()
