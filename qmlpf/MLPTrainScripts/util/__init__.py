#useful utilities
import signal,time, logging, os
LOGGING_LEVEL=logging.DEBUG

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def my_logger(name):
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


log=my_logger(__name__)


def alarm_handler(signum, frame):
    raise TimeoutError
def input_with_timeout(prompt, timeout=30):
    """ get input with timeout

    :param prompt: the prompt to print
    :param timeout: timeout in seconds, or None to disable

    :returns: the input
    :raises: TimeoutError if times out
    """
    # set signal handler
    if timeout is not None:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout) # produce SIGALRM in `timeout` seconds
    try:
        time.sleep(.5) # get input to be printed after logging
        return input(prompt)
    except TimeoutError as to:
        raise to
    finally:
        if timeout is not None:
            signal.alarm(0) # cancel alarm

def yes_or_no(question, default='y', timeout=None):
    """ Get y/n answer with default choice and optional timeout

    :param question: prompt
    :param default: the default choice, i.e. 'y' or 'n'
    :param timeout: the timeout in seconds, default is None

    :returns: True or False
    """
    if default is not None and (default!='y' and default!='n'):
        log.error(f'bad option for default: {default}')
        quit(1)
    y='Y' if default=='y' else 'y'
    n='N' if default=='n' else 'n'
    while "the answer is invalid":
        try:
            to_str='' if timeout is None or os.name=='nt' else f'(Timeout {default} in {timeout}s)'
            if os.name=='nt':
                log.warning('cannot use timeout signal on windows')
                time.sleep(.1) # make the warning come out first
                reply=str(input(f'{question} {to_str} ({y}/{n}): ')).lower().strip()
            else:
                reply = str(input_with_timeout(f'{question} {to_str} ({y}/{n}): ',timeout=timeout)).lower().strip()
        except TimeoutError:
            log.warning(f'timeout expired, returning default={default} answer')
            reply=''
        if len(reply)==0 or reply=='':
            return True if default=='y' else False
        elif reply[0].lower() == 'y':
            return True
        if reply[0].lower() == 'n':
            return False
