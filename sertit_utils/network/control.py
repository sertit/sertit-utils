import os
import time
import logging
import traceback
from random import Random


logger = logging.getLogger(__name__)


def exponential_backoff(network_request, wait_time_slot: float, increase_factor: float, max_wait: float, max_retries: int,
                        desc: str, random_state=None):
    """
    Implementation of the standard Exponential Backoff algorithm (https://en.wikipedia.org/wiki/Exponential_backoff)

    This algorithm is useful in networking setups where one has to use a potentially unreliable resource over the
    network, where by unreliable we mean not always available.

    Every major service provided over the network can't guarantee 100% availability all the time. Therefore if a service
    fails one should retry a certain number of time and a maximum amount of times. This algorithm is designed to try
    using the services multiple times with exponentially increasing delays between the tries. The time delays are chosen
    at random to ensure better theoretical behaviour.

    No default value is provided on purpose

    Args:
        network_request (function):
            Python function taking no arguments which represents a network request which may fail. If it fails it must
            raise an exception.
        wait_time_slot (float): Smallest amount of time to wait between retries. Recommended range [0.1 - 10.0] seconds
        increase_factor (float):
            Exponent of the expected exponential increase in waiting time. In other words on average by how much should
            the delay in time increase. Recommended range [1.5 - 3.0]
        max_wait (float):
            Maximum of time one will wait for the network request to perform successfully. If the maximum amount of time
            is reached a timeout exception is thrown.
        max_retries (int):
            Number of total tries to perform the network request. Must be at least 2 and maximum
            `EXP_BACK_OFF_ABS_MAX_RETRIES` (or 100 if the environment value is not defined). Recommended range [5 - 25].
            If the value exceeds `EXP_BACK_OFF_ABS_MAX_RETRIES` (or 100 if the environment value is not defined). The
            value will be set to `EXP_BACK_OFF_ABS_MAX_RETRIES` (or 100 if the environment value is not defined).
        desc (str): Description of the network request being attempted
        random_state (int): Seed to the random number generator (optional)

    Returns:
    """

    abs_max_tries = int(os.environ.get('EXP_BACK_OFF_ABS_MAX_RETRIES', '100'))
    if abs_max_tries <= 2:
        raise Exception(f"Environment variable 'EXP_BACK_OFF_ABS_MAX_RETRIES' must be positive finite integer greater "
                        f"than 2")

    if not float(wait_time_slot) == wait_time_slot or wait_time_slot <= 0.0:
        raise TypeError(f"Variable 'wait_time_slot' (current value {wait_time_slot}) must be float strictly greater than 0.0")

    if not float(increase_factor) == increase_factor or increase_factor <= 0.0:
        raise TypeError(f"Variable 'increase_factor' (current value {increase_factor}) must be float strictly greater "
                        f"than 0.0")

    if not float(max_wait) == max_wait or max_wait <= 0.0:
        raise TypeError(f"Variable 'max_wait' (current value {max_wait}) must be float strictly greater "
                        f"than 0.0")

    if not int(max_retries) == max_retries or max_retries <= 2:
        raise TypeError(f"Variable 'max_retries' (current value {max_retries}) must be integer strictly greater "
                        f"than 2")

    if random_state is not None and not int(random_state) == random_state:
        raise TypeError(
            f"Variable 'random_state' (current value {random_state}) must be 'None' or integer value")

    if desc == '':
        raise TypeError(
            f"Variable 'desc' (current value {desc}) must be non empty string")

    real_max_tries = min(max_retries, abs_max_tries)
    rng = Random(random_state)

    cumulated_wait_time = 0.0

    # First try with no wait
    try:
        return network_request()
    except Exception as e:
        logger.error(f"Action '{desc}' failed with exception: {e}", exc_info=True)

    for i in range(real_max_tries - 2):  # Avoids infinite loop
        random_scale = rng.randint(0, 2**i - 1)
        curr_wait_time = wait_time_slot * increase_factor ** random_scale

        logger.info(f'Retrying action {desc} in {curr_wait_time} seconds...')

        cumulated_wait_time += curr_wait_time
        if cumulated_wait_time > max_wait:
            logger.error(f"Waited {cumulated_wait_time} seconds : Maximum wait time ({max_wait}) reached "
                         f"!!! Aborting !!!")
            raise TimeoutError(f"About to exceed maximum wait time {max_wait} for action {desc}")

        time.sleep(curr_wait_time)

        try:
            return network_request()
        except Exception as e:
            logger.error(f"Action '{desc}' failed with exception: {e}", exc_info=True)

    # Final blind try and hope for the best
    network_request()