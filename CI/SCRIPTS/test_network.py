""" Script testing the network.control """
import pytest
from sertit_utils.network.control import exponential_backoff


class NetworkRequestMocker:

    def __init__(self, number_of_failures):
        self.number_of_failures = number_of_failures + 1
        self.try_number = 1

    def network_request(self):
        div, mod = divmod(self.try_number, self.number_of_failures)
        self.try_number += 1
        if mod == 0:
            return "The One Ring"
        else:
            raise Exception("Resource is unavailable")


def test_log():
    mocker = NetworkRequestMocker(5)

    with pytest.raises(Exception):
        mocker.network_request()  # Must raise

    with pytest.raises(TimeoutError):
        exponential_backoff(
            network_request=lambda: mocker.network_request(),
            wait_time_slot=0.05, increase_factor=2, max_wait=0.1, max_retries=6,
            desc="Requesting mock resource", random_state=0
        )

    resource = exponential_backoff(
        network_request=lambda: mocker.network_request(),
        wait_time_slot=0.05, increase_factor=2, max_wait=10, max_retries=6,
        desc="Requesting mock resource", random_state=0
    )

    assert resource == "The One Ring"
