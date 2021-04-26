# -*- coding: utf-8 -*-
# Copyright 2021, SERTIT-ICube - France, https://sertit.unistra.fr/
# This file is part of sertit-utils project
#     https://github.com/sertit/sertit-utils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Script testing the network.control """
import pytest

from sertit.network import exponential_backoff


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


def test_network():
    mocker = NetworkRequestMocker(5)

    with pytest.raises(Exception):
        mocker.network_request()  # Must raise

    with pytest.raises(TimeoutError):
        exponential_backoff(
            network_request=lambda: mocker.network_request(),
            wait_time_slot=0.05,
            increase_factor=2,
            max_wait=0.1,
            max_retries=6,
            desc="Requesting mock resource",
            random_state=0,
        )

    resource = exponential_backoff(
        network_request=lambda: mocker.network_request(),
        wait_time_slot=0.05,
        increase_factor=2,
        max_wait=10,
        max_retries=6,
        desc="Requesting mock resource",
        random_state=0,
    )

    assert resource == "The One Ring"
