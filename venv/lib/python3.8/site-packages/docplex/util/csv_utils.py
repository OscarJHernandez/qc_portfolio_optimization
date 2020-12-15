# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2020
# --------------------------------------------------------------------------
'''Some csv utilities
'''
import os


def encode_csv_string(text):
    """ Encode a string to be used in CSV file

    Args:
        text:  String to encode
    Returns:
        Encoded string, including starting and ending double quote
    """
    res = ['"']
    for c in text:
        res.append(c)
        if c == '"':
            res.append('"')
    res.append('"')
    return ''.join(res)


def write_csv_line(output, line, encoding):
    line = ','.join([encode_csv_string('%s' % c) for c in line])
    output.write(line.encode(encoding))
    output.write('\n'.encode(encoding))


def write_csv(env, table, fields, name):
    # table must be a named tuple
    encoding = 'utf-8'
    with env.get_output_stream(name) as ostr:
        write_csv_line(ostr, fields, encoding)
        for line in table:
            write_csv_line(ostr, line, encoding)


def write_table_as_csv(env, table, name, field_names):
    '''Writes a kpis dataframe as file which name is specified.
    The data type depends of extension of name.

    This uses the specfied env to write data as attachments
    '''
    _, ext = os.path.splitext(name)
    ext = ext.lower()
    if ext == '.csv':
        encoding = 'utf-8'
        with env.get_output_stream(name) as ostr:
            write_csv_line(ostr, field_names, encoding)
            for line in table:
                write_csv_line(ostr, line, encoding)
    else:
        # right now, only csv is supported
        raise ValueError('file format not supported for KPIs file: %s' % ext)
