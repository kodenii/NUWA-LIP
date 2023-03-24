# coding=utf-8
# Copyleft 2019 project LXRT.
import platform

from wutils import *

np.set_printoptions(suppress=True)


@iterable_class
class BasicArgs:
    # default settings
    resume = True
    use_tqdm = True
    # specific settings

    if len(platform.node()) == 12:
        logger.info("Detected Docker Node %s." % platform.node())
        root_dir = '/data/'
        debug = False
    else:
        raise ValueError("Unknown Node %s." % platform.node())

    @staticmethod
    def get_log_dir(config_filename):
        task_name, filename = os.path.normpath(config_filename).split(os.path.sep)[-2:]
        method_name = filename.split('.')[0]
        # If the rootname is MCG1024EMB_E100TK500SP50, then change it to MCG1024EMB/E100TK500SP50
        method_path = method_name.replace('_', os.path.sep)
        log_dir = os.path.join(BasicArgs.root_dir, task_name, "logs", method_path)
        return task_name, method_name, log_dir
